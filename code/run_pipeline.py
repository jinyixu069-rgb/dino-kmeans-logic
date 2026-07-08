# run_pipeline_gpu.py
# GPU 版 k-means 词袋 + Mahalanobis, K 扫描 (full-data logical AD)
#
# 相对上一版(位置增广)的改动:
#   - 去掉位置增广(augment_with_position 等),回到纯语义 k-means,已验证对结果无实质影响。
#   - --k 改为可接收多个值,一次扫多个 K,复用同一份缓存特征,不用重跑 DINO。
#   - 每个 K 都要重新聚类(词表大小变了,直方图维度也变了),独立跑、横向对比。
#
# 用法:
#   python run_pipeline_gpu.py --all --seeds 0 1 2 3 4 5 --k 64            # 复现 0.8331~0.8342 锚点
#   python run_pipeline_gpu.py --all --seeds 0 1 2 3 4 5 --k 32 64 128 256 # 扫 K

import os
import argparse
import numpy as np
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OBJS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

KMEANS_FIT_SUBSAMPLE = 200_000
KMEANS_ITERS = 100
KMEANS_BATCH = 4096
ASSIGN_CHUNK = 200_000


# ----------------------------- 数据加载 -----------------------------

def load_split_features(obj, split_name):
    d = os.path.join(CACHE_DIR, obj, split_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    return [np.load(os.path.join(d, f)).astype(np.float32) for f in files]


# ----------------------------- GPU k-means -----------------------------

def kmeans_gpu(X, k, seed, device, n_iters=KMEANS_ITERS, batch_size=KMEANS_BATCH,
              subsample=KMEANS_FIT_SUBSAMPLE):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    if n > subsample:
        idx = torch.randperm(n, generator=g)[:subsample]
        pool = X[idx]
    else:
        pool = X
    n_pool = pool.shape[0]

    init_idx = torch.randperm(n_pool, generator=g)[:k]
    centroids = pool[init_idx].clone()
    counts = torch.zeros(k, device=X.device)

    for _ in range(n_iters):
        bidx = torch.randint(0, n_pool, (min(batch_size, n_pool),), generator=g)
        batch = pool[bidx]
        assign = torch.cdist(batch, centroids).argmin(1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assign, batch)
        cnts = torch.bincount(assign, minlength=k).to(centroids.dtype)
        counts += cnts
        mask = cnts > 0
        if mask.any():
            lr = (cnts[mask] / counts[mask]).unsqueeze(1)
            centroids[mask] = (1 - lr) * centroids[mask] + lr * (sums[mask] / cnts[mask].unsqueeze(1))
    return centroids


def assign_gpu(X, centroids, chunk=ASSIGN_CHUNK):
    n = X.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=X.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        labels[s:e] = torch.cdist(X[s:e], centroids).argmin(1)
    return labels


def build_histogram_gpu(feats_list, centroids, k, device):
    counts_per_img = [f.shape[0] for f in feats_list]
    big = torch.cat([torch.from_numpy(f).to(device) for f in feats_list], dim=0)
    labels = assign_gpu(big, centroids)

    hists = torch.zeros((len(feats_list), k), dtype=torch.float64, device=device)
    off = 0
    for i, n in enumerate(counts_per_img):
        hists[i] = torch.bincount(labels[off:off + n], minlength=k).to(torch.float64)
        off += n
    return hists.cpu().numpy()


# ----------------------------- 单类别流程 -----------------------------

def run_one_category(obj, k, seed, device_str):
    device = torch.device(device_str)
    train_feats = load_split_features(obj, "train_good")
    good_feats = load_split_features(obj, "test_good")
    logical_feats = load_split_features(obj, "test_logical")

    train_tensor = torch.from_numpy(np.concatenate(train_feats, axis=0)).to(device)
    centroids = kmeans_gpu(train_tensor, k, seed, device)

    train_hist = build_histogram_gpu(train_feats, centroids, k, device)
    good_hist = build_histogram_gpu(good_feats, centroids, k, device)
    logical_hist = build_histogram_gpu(logical_feats, centroids, k, device)

    lw = LedoitWolf().fit(train_hist)
    mean_vec = torch.from_numpy(lw.location_).to(device=device, dtype=torch.float64)
    precision = torch.from_numpy(lw.get_precision()).to(device=device, dtype=torch.float64)

    def maha(hist_np):
        h = torch.from_numpy(hist_np).to(device=device, dtype=torch.float64)
        diff = h - mean_vec
        return torch.einsum("ij,jk,ik->i", diff, precision, diff).cpu().numpy()

    good_s, logical_s = maha(good_hist), maha(logical_hist)
    y_true = np.concatenate([np.zeros(len(good_s)), np.ones(len(logical_s))])
    y_score = np.concatenate([good_s, logical_s])
    return roc_auc_score(y_true, y_score)


# ----------------------------- 并行调度 -----------------------------

def _worker(task):
    obj, k, seed, gpu_id = task
    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        device_str = "cpu"
    auc = run_one_category(obj, k, seed, device_str)
    return obj, seed, k, auc


def run_all_jobs(objs, ks, seeds, device_pref, workers):
    jobs = [(obj, seed, k) for k in ks for seed in seeds for obj in objs]
    results = {k: {obj: {} for obj in objs} for k in ks}
    n_gpus = 0 if device_pref == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)

    if n_gpus >= 1:
        tagged = [(obj, k, seed, i % n_gpus) for i, (obj, seed, k) in enumerate(jobs)]
        max_workers = workers or n_gpus
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, t) for t in tagged]
            for fut in as_completed(futs):
                obj, seed, k, auc = fut.result()
                results[k][obj][seed] = auc
                print(f"[k={k}][seed={seed}] {obj:22s}: {auc:.4f}", flush=True)
    else:
        for (obj, seed, k) in jobs:
            _, _, _, auc = _worker((obj, k, seed, -1))
            results[k][obj][seed] = auc
            print(f"[k={k}][seed={seed}] {obj:22s}: {auc:.4f}", flush=True)

    ordered = {k: {obj: [results[k][obj][s] for s in seeds] for obj in objs} for k in ks}
    return ordered


# ----------------------------- 主入口 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", choices=OBJS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", default=[64],
                    help="视觉词数量列表,如 32 64 128 256")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    objs = OBJS if args.all else ([args.obj] if args.obj else None)
    if objs is None:
        print("请指定 --obj <category> 或 --all")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但无可用 GPU")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_all_jobs(objs, args.k, args.seeds, args.device, args.workers)

    n_gpus = 0 if args.device == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    lines = [
        f"# GPU k-means bag-of-words + Mahalanobis, K sweep (full-data logical AD)",
        f"# DINOv2-with-registers-giant, anisotropic resize 672, mean layers [-18,-12]",
        f"# k values={args.k}, seeds={args.seeds}, device={'%d GPU'%n_gpus if n_gpus else 'CPU'}, "
        f"time={datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for k in args.k:
        lines.append(f"=== k = {k} ===")
        lines.append(f"{'category':22s}  {'mean':>7s}  {'std':>6s}  per-seed")
        macro_ps = np.zeros(len(args.seeds))
        for obj in objs:
            arr = np.array(results[k][obj])
            lines.append(f"{obj:22s}  {arr.mean():.4f}  {arr.std():.4f}  "
                         f"[{', '.join(f'{a:.4f}' for a in arr)}]")
        if args.all:
            for si in range(len(args.seeds)):
                macro_ps[si] = np.mean([results[k][obj][si] for obj in objs])
            lines.append(f"{'MACRO':22s}  {macro_ps.mean():.4f}  {macro_ps.std():.4f}  "
                         f"[{', '.join(f'{m:.4f}' for m in macro_ps)}]")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_tag = "_".join(str(x) for x in args.k)
    out_path = os.path.join(RESULTS_DIR, f"kmeans_gpu_ksweep{k_tag}_{stamp}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()