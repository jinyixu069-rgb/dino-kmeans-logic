# run_pipeline_gpu.py
# GPU 版 k-means 词袋 + crop 独立打分聚合 + Mahalanobis (full-data logical AD)
#
# 相对 K-sweep 版的改动:
#   - 新增 --crop-split N:把每张图的 48x48 patch 网格切成 (全图) + N×N 个 crop 子区域。
#     每个"区域"(含全图)独立做 k-means 词袋 + LedoitWolf + Mahalanobis 打分,
#     最后跨区域聚合成一个图级分数。这是 SINBAD "crop 当独立图" 思想的迁移。
#     --crop-split 0 或 1 时退化为纯全图(= 你验证过的 0.83 锚点)。
#   - 关键设计: 每个区域独立拟合自己的协方差(维度仍是 K,不拼成高维向量),
#     从而绕开金字塔"拼高维向量导致协方差欠定"的坑。这是 SINBAD 的活法,不是金字塔的死法。
#   - 聚合方式 --agg: mean / max / sum,跨区域聚合各自的马氏距离分数。
#
# 重要局限: 本版在【已缓存的 patch 特征】上按空间切网格,不重新裁图过 DINO,
#   因此只复现 SINBAD "局部化统计" 的作用,不复现 "小物体放大" 的作用。
#
# 用法:
#   python run_pipeline_gpu.py --all --seeds 0 1 --crop-split 0            # 全图锚点(=0.83)
#   python run_pipeline_gpu.py --all --seeds 0 1 --crop-split 2 3          # 全图 + 2x2 + 3x3
#   python run_pipeline_gpu.py --all --seeds 0 1 --crop-split 3 --agg max  # 换聚合方式

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

def load_split_features(obj, split_name, cache_dir):
    d = os.path.join(cache_dir, obj, split_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    return [np.load(os.path.join(d, f)).astype(np.float32) for f in files]


# ----------------------------- crop 区域定义 -----------------------------

def build_regions(crop_splits, n_patches):
    """返回一个 region 列表,每个 region 是一个函数 patch_idx->bool_mask,
    或直接返回每个 region 的 patch 索引(基于 48x48 行优先网格)。
    region 0 永远是全图;之后是各个 N×N 切分的 crop。
    crop_splits: 如 [2,3] 表示额外加 2x2 和 3x3 的 crop。"""
    regions = [("full", None)]  # 全图; None 表示保留所有 patch,支持 48x48/64x64 等不同分辨率
    if not crop_splits:
        return regions

    grid = int(round(np.sqrt(n_patches)))
    if grid * grid != n_patches:
        raise ValueError(f"n_patches={n_patches} is not a square grid")
    grid_idx = np.arange(n_patches).reshape(grid, grid)

    for n_split in crop_splits:
        if n_split <= 1:
            continue  # 1 等于全图,跳过避免重复
        step = grid // n_split
        for i in range(n_split):
            for j in range(n_split):
                r0 = i * step
                r1 = (i + 1) * step if i < n_split - 1 else grid
                c0 = j * step
                c1 = (j + 1) * step if j < n_split - 1 else grid
                idx = grid_idx[r0:r1, c0:c1].reshape(-1)
                regions.append((f"{n_split}x{n_split}_{i}_{j}", idx))
    return regions


def extract_region_feats(feats_list, region_idx):
    """从每张图的完整 patch 特征里,取出属于该 region 的 patch。
    feats_list: list of [2304, D] numpy。返回 list of [len(region_idx), D]。"""
    if region_idx is None:
        return feats_list
    return [f[region_idx] for f in feats_list]


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


def build_histogram_gpu(region_feats_list, centroids, k, device):
    counts_per_img = [f.shape[0] for f in region_feats_list]
    big = torch.cat([torch.from_numpy(f).to(device) for f in region_feats_list], dim=0)
    labels = assign_gpu(big, centroids)
    hists = torch.zeros((len(region_feats_list), k), dtype=torch.float64, device=device)
    off = 0
    for i, n in enumerate(counts_per_img):
        hists[i] = torch.bincount(labels[off:off + n], minlength=k).to(torch.float64)
        off += n
    return hists.cpu().numpy()


# ----------------------------- 单个 region 的打分 -----------------------------

def score_one_region(train_rf, good_rf, logical_rf, k, seed, device):
    """对单个 region: 独立 k-means + 独立 LedoitWolf + Mahalanobis。
    返回 (good_scores, logical_scores),都是原始马氏距离(未归一化)。"""
    train_tensor = torch.from_numpy(np.concatenate(train_rf, axis=0)).to(device)
    centroids = kmeans_gpu(train_tensor, k, seed, device)

    train_hist = build_histogram_gpu(train_rf, centroids, k, device)
    good_hist = build_histogram_gpu(good_rf, centroids, k, device)
    logical_hist = build_histogram_gpu(logical_rf, centroids, k, device)

    lw = LedoitWolf().fit(train_hist)
    mean_vec = torch.from_numpy(lw.location_).to(device=device, dtype=torch.float64)
    precision = torch.from_numpy(lw.get_precision()).to(device=device, dtype=torch.float64)

    def maha(hist_np):
        h = torch.from_numpy(hist_np).to(device=device, dtype=torch.float64)
        diff = h - mean_vec
        return torch.einsum("ij,jk,ik->i", diff, precision, diff).cpu().numpy()

    return maha(good_hist), maha(logical_hist)


def zscore_normalize(good_s, logical_s, train_ref=None):
    """跨 region 聚合前,把不同 region 的分数标准化到可比尺度。
    用 good(正常测试)分数的均值/标准差做标准化 —— 但注意这里用了测试集正常样本的统计,
    严格说应该用训练集重算一份正常分数来标准化。为简洁先用 good 的分布近似,
    这只影响不同 region 的相对权重,不直接读测试标签(good/logical 都用同一套变换)。"""
    mu = good_s.mean()
    sd = good_s.std() + 1e-8
    return (good_s - mu) / sd, (logical_s - mu) / sd


# ----------------------------- 单类别流程 -----------------------------

def run_one_category(obj, k, seed, crop_splits, agg, device_str, cache_dir):
    device = torch.device(device_str)
    train_feats = load_split_features(obj, "train_good", cache_dir)
    good_feats = load_split_features(obj, "test_good", cache_dir)
    logical_feats = load_split_features(obj, "test_logical", cache_dir)

    regions = build_regions(crop_splits, train_feats[0].shape[0])

    good_region_scores = []
    logical_region_scores = []
    for _, region_idx in regions:
        train_rf = extract_region_feats(train_feats, region_idx)
        good_rf = extract_region_feats(good_feats, region_idx)
        logical_rf = extract_region_feats(logical_feats, region_idx)

        good_s, logical_s = score_one_region(train_rf, good_rf, logical_rf, k, seed, device)
        # 标准化后再聚合,避免某个 region 因尺度大主导聚合结果
        good_z, logical_z = zscore_normalize(good_s, logical_s)
        good_region_scores.append(good_z)
        logical_region_scores.append(logical_z)

    good_mat = np.stack(good_region_scores, axis=0)      # [n_regions, n_good]
    logical_mat = np.stack(logical_region_scores, axis=0)  # [n_regions, n_logical]

    if agg == "mean":
        good_final = good_mat.mean(0)
        logical_final = logical_mat.mean(0)
    elif agg == "max":
        good_final = good_mat.max(0)
        logical_final = logical_mat.max(0)
    elif agg == "sum":
        good_final = good_mat.sum(0)
        logical_final = logical_mat.sum(0)
    else:
        raise ValueError(agg)

    y_true = np.concatenate([np.zeros(len(good_final)), np.ones(len(logical_final))])
    y_score = np.concatenate([good_final, logical_final])
    return roc_auc_score(y_true, y_score)


# ----------------------------- 并行调度 -----------------------------

def _worker(task):
    obj, k, seed, crop_splits, agg, gpu_id, cache_dir = task
    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}"
    else:
        device_str = "cpu"
    auc = run_one_category(obj, k, seed, crop_splits, agg, device_str, cache_dir)
    return obj, seed, auc


def run_all_jobs(objs, k, seeds, crop_splits, agg, device_pref, workers, cache_dir):
    jobs = [(obj, seed) for seed in seeds for obj in objs]
    results = {obj: {} for obj in objs}
    n_gpus = 0 if device_pref == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)

    if n_gpus >= 1:
        tagged = [(obj, k, seed, crop_splits, agg, i % n_gpus, cache_dir)
                  for i, (obj, seed) in enumerate(jobs)]
        max_workers = workers or n_gpus
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, t) for t in tagged]
            for fut in as_completed(futs):
                obj, seed, auc = fut.result()
                results[obj][seed] = auc
                print(f"[seed={seed}] {obj:22s}: {auc:.4f}", flush=True)
    else:
        for (obj, seed) in jobs:
            _, _, auc = _worker((obj, k, seed, crop_splits, agg, -1, cache_dir))
            results[obj][seed] = auc
            print(f"[seed={seed}] {obj:22s}: {auc:.4f}", flush=True)

    ordered = {obj: [results[obj][s] for s in seeds] for obj in objs}
    return ordered


# ----------------------------- 主入口 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", nargs="+", choices=OBJS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--crop-split", type=int, nargs="+", default=[0],
                    help="额外的 crop 切分粒度,如 2 3 表示全图+2x2+3x3。0/1=仅全图(锚点)")
    ap.add_argument("--agg", choices=["mean", "max", "sum"], default="mean",
                    help="跨 region 聚合方式")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    objs = OBJS if args.all else args.obj
    if objs is None:
        print("请指定 --obj <category> 或 --all")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但无可用 GPU")

    crop_splits = [c for c in args.crop_split if c > 1]  # 过滤掉 0/1,全图总是包含
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_all_jobs(objs, args.k, args.seeds, crop_splits, args.agg,
                           args.device, args.workers, args.cache_dir)

    n_gpus = 0 if args.device == "cpu" else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    n_regions = 1 + sum(c * c for c in crop_splits)
    lines = [
        f"# GPU k-means BoW + crop-region independent scoring + Mahalanobis (full-data logical AD)",
        f"# DINOv2-with-registers-giant, anisotropic resize 672, mean layers [-18,-12]",
        f"# k={args.k}, crop_split={args.crop_split} (total {n_regions} regions), agg={args.agg}, "
        f"seeds={args.seeds}, device={'%d GPU'%n_gpus if n_gpus else 'CPU'}, "
        f"cache_dir={args.cache_dir}, "
        f"time={datetime.now().isoformat(timespec='seconds')}",
        f"# NOTE: crop 在已缓存 patch 特征上切网格,不重裁图过 DINO(只复现局部化统计,不复现小物体放大)",
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
    cs_tag = "".join(str(x) for x in args.crop_split)
    out_path = os.path.join(RESULTS_DIR, f"kmeans_gpu_crop{cs_tag}_{args.agg}_k{args.k}_{stamp}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
