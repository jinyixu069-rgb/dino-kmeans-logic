# run_pipeline_gpu.py
# GPU 版 k-means 词袋 + 空间金字塔 + Mahalanobis (full-data logical AD)
#
# 相对 CPU 版的改动:
#   1. k-means / 最近邻分配 / Mahalanobis 全在 GPU 上(torch),CPU 版的 sklearn 只保留 LedoitWolf。
#   2. 新增空间金字塔: --pyramid 1 2 4  表示 全局(1x1) + 2x2 + 4x4 三个级别,
#      每个 cell 各自统计 K 维词频后拼接。--pyramid 1 即退化为原全局版本(对照锚点)。
#   3. 多卡: 每个 (obj,seed) round-robin 分到不同 GPU (spawn 上下文, 避免 fork+CUDA 挂死)。
#      单卡/CPU: 串行 (这个任务单类很快, 不值得为单卡再上线程复杂度)。
#
# 用法:
#   python run_pipeline_gpu.py --all --seeds 0 1 2 3 4 5 --pyramid 1          # 复现 0.8342 锚点
#   python run_pipeline_gpu.py --all --seeds 0 1 2 3 4 5 --pyramid 1 2 4      # 金字塔
#   python run_pipeline_gpu.py --all --seeds 0 1 2 3 4 5 --pyramid 1 2 4 --k 128
#   python run_pipeline_gpu.py --all --pyramid 1 2 4 --device cpu             # 调试用

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
GRID = 48   # 672 / 14 = 48 patches per side (依赖行优先排列)


# ----------------------------- 数据加载 -----------------------------

def load_split_features(obj, split_name):
    d = os.path.join(CACHE_DIR, obj, split_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    feats = [np.load(os.path.join(d, f)).astype(np.float32) for f in files]
    return feats


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
    counts = torch.zeros(k, device=device)

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


# ----------------------------- 空间金字塔词频 -----------------------------

def cell_ids_for_level(level, device):
    """返回长度 GRID*GRID 的张量,每个 patch 属于该 level 下的第几个 cell (行优先)。
    level=1 -> 全部同一个 cell; level=2 -> 2x2=4 cells; level=4 -> 4x4=16 cells。"""
    rows = torch.arange(GRID, device=device).view(GRID, 1).expand(GRID, GRID)
    cols = torch.arange(GRID, device=device).view(1, GRID).expand(GRID, GRID)
    cell_r = rows * level // GRID
    cell_c = cols * level // GRID
    cell = (cell_r * level + cell_c).reshape(-1)  # [GRID*GRID]
    return cell  # 值域 [0, level*level)


def pyramid_histogram(labels_per_img, k, levels, device):
    """labels_per_img: list,每个元素是一张图的 patch 词 id (长度应为 GRID*GRID)。
    返回 [num_img, k * sum(l*l for l in levels)] 的金字塔词频向量。"""
    level_cell_ids = {lv: cell_ids_for_level(lv, device) for lv in levels}
    total_dim = k * sum(lv * lv for lv in levels)
    out = torch.zeros((len(labels_per_img), total_dim), dtype=torch.float64, device=device)

    for img_i, labels in enumerate(labels_per_img):
        col = 0
        for lv in levels:
            n_cells = lv * lv
            cells = level_cell_ids[lv]  # [GRID*GRID]
            # 联合索引 = cell * k + word,一次 bincount 出 n_cells*k 维
            joint = cells * k + labels   # [GRID*GRID]
            hist = torch.bincount(joint, minlength=n_cells * k).to(torch.float64)
            out[img_i, col:col + n_cells * k] = hist
            col += n_cells * k
    return out.cpu().numpy()


def build_pyramid_hist(feats_list, centroids, k, levels, device):
    """对一个 split: 拼所有 patch 一次性分配,再按图切回、按金字塔统计。"""
    counts_per_img = [f.shape[0] for f in feats_list]
    # 正确性检查: 每张图 patch 数必须等于 GRID*GRID,否则金字塔网格无意义
    for i, c in enumerate(counts_per_img):
        if c != GRID * GRID:
            raise ValueError(
                f"image {i} has {c} patches, expected {GRID*GRID}. "
                f"金字塔要求所有图统一 {GRID}x{GRID} patch(各向异性 resize 保证了这点)")
    big = torch.from_numpy(np.concatenate(feats_list, axis=0)).to(device)
    labels = assign_gpu(big, centroids)

    labels_per_img = []
    off = 0
    for c in counts_per_img:
        labels_per_img.append(labels[off:off + c])
        off += c
    return pyramid_histogram(labels_per_img, k, levels, device)


# ----------------------------- 单类别流程 -----------------------------

def run_one_category(obj, k, seed, levels, device_str):
    device = torch.device(device_str)
    train_feats = load_split_features(obj, "train_good")
    good_feats = load_split_features(obj, "test_good")
    logical_feats = load_split_features(obj, "test_logical")

    train_tensor = torch.from_numpy(np.concatenate(train_feats, axis=0)).to(device)
    centroids = kmeans_gpu(train_tensor, k, seed, device)

    train_hist = build_pyramid_hist(train_feats, centroids, k, levels, device)
    good_hist = build_pyramid_hist(good_feats, centroids, k, levels, device)
    logical_hist = build_pyramid_hist(logical_feats, centroids, k, levels, device)

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
    obj, k, seed, levels, gpu_id = task
    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        device_str = "cpu"
    auc = run_one_category(obj, k, seed, levels, device_str)
    return obj, seed, auc


def run_all_jobs(objs, k, seeds, levels, device_pref, workers):
    jobs = [(obj, seed) for seed in seeds for obj in objs]
    results = {obj: {} for obj in objs}
    n_gpus = 0 if device_pref == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)

    if n_gpus >= 1:
        tagged = [(obj, k, seed, levels, i % n_gpus) for i, (obj, seed) in enumerate(jobs)]
        max_workers = workers or n_gpus
        ctx = mp.get_context("spawn")   # 关键: 避免 fork + CUDA 挂死
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, t) for t in tagged]
            for fut in as_completed(futs):
                obj, seed, auc = fut.result()
                results[obj][seed] = auc
                print(f"[seed={seed}] {obj:22s}: {auc:.4f}", flush=True)
    else:
        for (obj, seed) in jobs:
            _, _, auc = _worker((obj, k, seed, levels, -1))
            results[obj][seed] = auc
            print(f"[seed={seed}] {obj:22s}: {auc:.4f}", flush=True)

    ordered = {obj: [results[obj][s] for s in seeds] for obj in objs}
    return ordered


# ----------------------------- 主入口 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", choices=OBJS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pyramid", type=int, nargs="+", default=[1],
                    help="金字塔级别,如 1 (全局) / 1 2 / 1 2 4。1=全局对照锚点")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    objs = OBJS if args.all else ([args.obj] if args.obj else None)
    if objs is None:
        print("请指定 --obj <category> 或 --all")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但无可用 GPU")

    levels = args.pyramid
    for lv in levels:
        if GRID % lv != 0:
            print(f"[warn] GRID={GRID} 不能被 level={lv} 整除,cell 大小会不均匀(仍可跑但边缘 cell 偏大)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_all_jobs(objs, args.k, args.seeds, levels, args.device, args.workers)

    n_gpus = 0 if args.device == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    dim = args.k * sum(lv * lv for lv in levels)
    lines = [
        f"# GPU k-means bag-of-words + spatial pyramid + Mahalanobis (full-data logical AD)",
        f"# DINOv2-with-registers-giant, anisotropic resize 672, mean layers [-18,-12]",
        f"# k={args.k}, pyramid={levels} (feature dim={dim}), seeds={args.seeds}, "
        f"device={'%d GPU'%n_gpus if n_gpus else 'CPU'}, "
        f"time={datetime.now().isoformat(timespec='seconds')}",
        "",
        f"{'category':22s}  {'mean':>7s}  {'std':>6s}  per-seed",
    ]
    macro_ps = np.zeros(len(args.seeds))
    for obj in objs:
        arr = np.array(results[obj])
        lines.append(f"{obj:22s}  {arr.mean():.4f}  {arr.std():.4f}  "
                     f"[{', '.join(f'{a:.4f}' for a in arr)}]")
    if args.all:
        for si in range(len(args.seeds)):
            macro_ps[si] = np.mean([results[obj][si] for obj in objs])
        lines.append(f"{'MACRO':22s}  {macro_ps.mean():.4f}  {macro_ps.std():.4f}  "
                     f"[{', '.join(f'{m:.4f}' for m in macro_ps)}]")

    report = "\n".join(lines)
    print("\n" + report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lv_tag = "".join(str(x) for x in levels)
    out_path = os.path.join(RESULTS_DIR, f"kmeans_gpu_pyr{lv_tag}_k{args.k}_{stamp}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
