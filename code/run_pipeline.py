# run_pipeline.py
# GPU 版 k-means 词袋 + 位置增广(路线一) + Mahalanobis (full-data logical AD)
#
# 相对金字塔版本的改动:
#   - 去掉空间金字塔(cell_ids_for_level / pyramid_histogram 等),直方图恢复成单一全局 K 维。
#   - 新增 --pos-weight λ1 λ2 ...:在聚类和最近邻分配前,把每个 patch 的归一化坐标
#     (row/47-0.5, col/47-0.5) 乘以 λ 拼接到特征向量末尾,让 k-means 在"语义+位置"联合
#     空间里聚类。λ=0 时,拼接的两维恒为 0,数学上严格等价于不拼接(已在沙盒验证)。
#   - 每个 λ 都需要重新做一次 k-means(因为距离空间变了),不能像金字塔那样在同一次聚类结果上
#     叠加,所以 λ 列表是"逐个独立跑、横向对比",不是"组合"。
#
# 用法:
#   python run_pipeline.py --all --seeds 0 1 2 3 4 5 --pos-weight 0            # 复现 0.8331 锚点
#   python run_pipeline.py --all --seeds 0 1 2 3 4 5 --pos-weight 0 0.5 1 2 5  # 扫 λ

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

DEFAULT_K = 64
KMEANS_FIT_SUBSAMPLE = 200_000
KMEANS_ITERS = 100
KMEANS_BATCH = 4096
ASSIGN_CHUNK = 200_000
GRID = 48   # 672 / 14 = 48,依赖特征提取时保留的行优先 patch 排列


# ----------------------------- 数据加载 -----------------------------

def load_split_features(obj, split_name):
    d = os.path.join(CACHE_DIR, obj, split_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    return [np.load(os.path.join(d, f)).astype(np.float32) for f in files]


# ----------------------------- 位置增广(路线一) -----------------------------

_POS_CACHE = {}  # grid -> [GRID*GRID, 2] 归一化坐标(未乘 lambda),复用避免重复计算


def get_normalized_coords(grid, device):
    key = (grid, str(device))
    if key not in _POS_CACHE:
        rows = torch.arange(grid, device=device).view(grid, 1).expand(grid, grid).reshape(-1)
        cols = torch.arange(grid, device=device).view(1, grid).expand(grid, grid).reshape(-1)
        r = rows.float() / (grid - 1) - 0.5
        c = cols.float() / (grid - 1) - 0.5
        _POS_CACHE[key] = torch.stack([r, c], dim=1)  # [GRID*GRID, 2], 值域 [-0.5, 0.5]
    return _POS_CACHE[key]


def augment_with_position(feat, lam, device):
    """feat: [N, D] (N 应为 GRID*GRID,一张图的 patch 特征)。返回 [N, D+2]。
    lam=0 时最后两维恒为 0,等价于不拼接(已验证)。"""
    n = feat.shape[0]
    if n != GRID * GRID:
        raise ValueError(f"expected {GRID*GRID} patches, got {n}; 位置坐标假设了统一的 {GRID}x{GRID} 网格")
    coords = get_normalized_coords(GRID, device) * lam  # [N, 2]
    return torch.cat([feat, coords], dim=1)


def augment_list(feats_list, lam, device):
    return [augment_with_position(torch.from_numpy(f).to(device), lam, device) for f in feats_list]


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


def build_histogram_gpu(aug_feats_list, centroids, k, device):
    """aug_feats_list: list of [N_i, D+2] tensor(已经过位置增广)。返回 [num_img, k] 直方图。"""
    counts_per_img = [f.shape[0] for f in aug_feats_list]
    big = torch.cat(aug_feats_list, dim=0)
    labels = assign_gpu(big, centroids)

    hists = torch.zeros((len(aug_feats_list), k), dtype=torch.float64, device=device)
    off = 0
    for i, n in enumerate(counts_per_img):
        hists[i] = torch.bincount(labels[off:off + n], minlength=k).to(torch.float64)
        off += n
    return hists.cpu().numpy()


# ----------------------------- 单类别流程 -----------------------------

def run_one_category(obj, k, seed, lam, device_str, report_scale=False):
    device = torch.device(device_str)
    train_feats = load_split_features(obj, "train_good")
    good_feats = load_split_features(obj, "test_good")
    logical_feats = load_split_features(obj, "test_logical")

    train_aug = augment_list(train_feats, lam, device)
    good_aug = augment_list(good_feats, lam, device)
    logical_aug = augment_list(logical_feats, lam, device)

    if report_scale:
        # 诊断: 原始特征各维典型尺度 vs 位置维尺度,帮助判断 lambda 的合理取值范围
        sample = train_aug[0]  # [N, D+2]
        feat_part = sample[:, :-2]
        pos_part = sample[:, -2:]
        print(f"[scale-debug][{obj}] feature dims: mean_abs={feat_part.abs().mean().item():.4f}, "
              f"std={feat_part.std().item():.4f} | position dims (after *lambda={lam}): "
              f"mean_abs={pos_part.abs().mean().item():.4f}, std={pos_part.std().item():.4f}",
              flush=True)

    train_tensor = torch.cat(train_aug, dim=0)
    centroids = kmeans_gpu(train_tensor, k, seed, device)

    train_hist = build_histogram_gpu(train_aug, centroids, k, device)
    good_hist = build_histogram_gpu(good_aug, centroids, k, device)
    logical_hist = build_histogram_gpu(logical_aug, centroids, k, device)

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
    obj, k, seed, lam, gpu_id, report_scale = task
    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        device_str = "cpu"
    auc = run_one_category(obj, k, seed, lam, device_str, report_scale)
    return obj, seed, lam, auc


def run_all_jobs(objs, k, seeds, lambdas, device_pref, workers):
    # 只对每个 (obj, lambda) 的第一次出现打印 scale 诊断,避免刷屏
    seen_scale = set()
    jobs = []
    for lam in lambdas:
        for seed in seeds:
            for obj in objs:
                report_scale = (obj, lam) not in seen_scale
                seen_scale.add((obj, lam))
                jobs.append((obj, seed, lam, report_scale))

    results = {lam: {obj: {} for obj in objs} for lam in lambdas}
    n_gpus = 0 if device_pref == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)

    if n_gpus >= 1:
        tagged = [(obj, k, seed, lam, i % n_gpus, rs)
                  for i, (obj, seed, lam, rs) in enumerate(jobs)]
        max_workers = workers or n_gpus
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, t) for t in tagged]
            for fut in as_completed(futs):
                obj, seed, lam, auc = fut.result()
                results[lam][obj][seed] = auc
                print(f"[lambda={lam}][seed={seed}] {obj:22s}: {auc:.4f}", flush=True)
    else:
        for (obj, seed, lam, rs) in jobs:
            _, _, _, auc = _worker((obj, k, seed, lam, -1, rs))
            results[lam][obj][seed] = auc
            print(f"[lambda={lam}][seed={seed}] {obj:22s}: {auc:.4f}", flush=True)

    ordered = {lam: {obj: [results[lam][obj][s] for s in seeds] for obj in objs} for lam in lambdas}
    return ordered


# ----------------------------- 主入口 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", choices=OBJS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pos-weight", type=float, nargs="+", default=[0.0],
                    help="位置权重 lambda 列表,如 0 0.5 1 2 5。0=纯语义锚点(等价原版)")
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
    results = run_all_jobs(objs, args.k, args.seeds, args.pos_weight, args.device, args.workers)

    n_gpus = 0 if args.device == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    lines = [
        f"# GPU k-means bag-of-words + position augmentation (route 1) + Mahalanobis (full-data logical AD)",
        f"# DINOv2-with-registers-giant, anisotropic resize 672, mean layers [-18,-12]",
        f"# k={args.k}, pos_weight(lambda)={args.pos_weight}, seeds={args.seeds}, "
        f"device={'%d GPU'%n_gpus if n_gpus else 'CPU'}, "
        f"time={datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for lam in args.pos_weight:
        lines.append(f"=== lambda = {lam} ===")
        lines.append(f"{'category':22s}  {'mean':>7s}  {'std':>6s}  per-seed")
        macro_ps = np.zeros(len(args.seeds))
        for obj in objs:
            arr = np.array(results[lam][obj])
            lines.append(f"{obj:22s}  {arr.mean():.4f}  {arr.std():.4f}  "
                         f"[{', '.join(f'{a:.4f}' for a in arr)}]")
        if args.all:
            for si in range(len(args.seeds)):
                macro_ps[si] = np.mean([results[lam][obj][si] for obj in objs])
            lines.append(f"{'MACRO':22s}  {macro_ps.mean():.4f}  {macro_ps.std():.4f}  "
                         f"[{', '.join(f'{m:.4f}' for m in macro_ps)}]")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lam_tag = "_".join(str(x) for x in args.pos_weight)
    out_path = os.path.join(RESULTS_DIR, f"kmeans_gpu_pos{lam_tag}_k{args.k}_{stamp}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
