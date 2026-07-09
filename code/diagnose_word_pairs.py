# diagnose_word_pairs.py
# 诊断视觉词对(word-pair)空间共现频次分布的集中程度。
#
# 只使用 train_good,不读取测试集。对每个类别:
#   1. 用 train_good patch 特征拟合 K=64 MiniBatchKMeans 视觉词表
#   2. 将每张图的词 id reshape 成 48x48 网格
#   3. 只统计右/下四邻居无向边,词对按 (min, max) 合并方向
#   4. 报告 top-k 高频词对占比、非零词对数、归一化熵

import os
import argparse
import numpy as np
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans

PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "word_pair_diagnostic")

DEFAULT_OBJS = ["pushpins", "breakfast_box", "screw_bag"]
DEFAULT_K = 64
GRID = 48
KMEANS_FIT_SUBSAMPLE = 200_000
TOP_CUTS = [10, 50, 100, 200, 500]


def load_train_features(obj):
    d = os.path.join(CACHE_DIR, obj, "train_good")
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    feats = [np.load(os.path.join(d, f)).astype(np.float32) for f in files]
    return feats


def fit_kmeans(train_feats, k, seed):
    pooled = np.concatenate(train_feats, axis=0)
    n = pooled.shape[0]
    if n > KMEANS_FIT_SUBSAMPLE:
        idx = np.random.RandomState(seed).choice(n, KMEANS_FIT_SUBSAMPLE, replace=False)
        pooled = pooled[idx]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
    km.fit(pooled)
    return km


def pair_index(a, b, k):
    """Map unordered pair (a,b), a<=b, to [0, k*(k+1)//2)."""
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    return lo * k - (lo * (lo - 1)) // 2 + (hi - lo)


def index_pair(idx, k):
    """Inverse of pair_index for display."""
    start = 0
    for a in range(k):
        n = k - a
        if idx < start + n:
            return a, a + (idx - start)
        start += n
    raise ValueError(idx)


def count_word_pairs(km, train_feats, k):
    n_pairs = k * (k + 1) // 2
    counts = np.zeros(n_pairs, dtype=np.int64)

    expected_patches = GRID * GRID
    for feat in train_feats:
        if feat.shape[0] != expected_patches:
            raise ValueError(f"expected {expected_patches} patches, got {feat.shape[0]}")
        labels = km.predict(feat).reshape(GRID, GRID)

        right_a = labels[:, :-1].reshape(-1)
        right_b = labels[:, 1:].reshape(-1)
        down_a = labels[:-1, :].reshape(-1)
        down_b = labels[1:, :].reshape(-1)

        pair_ids = np.concatenate([
            pair_index(right_a, right_b, k),
            pair_index(down_a, down_b, k),
        ])
        counts += np.bincount(pair_ids, minlength=n_pairs)

    return counts


def summarize_counts(obj, counts, n_images, k):
    total_edges = int(counts.sum())
    nonzero = int((counts > 0).sum())
    n_pairs = k * (k + 1) // 2
    sorted_counts = np.sort(counts)[::-1]

    top_fracs = {}
    for top in TOP_CUTS:
        top_fracs[top] = float(sorted_counts[:top].sum() / total_edges)

    p = counts[counts > 0].astype(np.float64) / total_edges
    entropy = float(-(p * np.log(p)).sum())
    norm_entropy = float(entropy / np.log(n_pairs))

    top_ids = np.argsort(counts)[::-1][:10]
    top_pairs = []
    for idx in top_ids:
        a, b = index_pair(int(idx), k)
        c = int(counts[idx])
        top_pairs.append((a, b, c, c / total_edges))

    return {
        "category": obj,
        "n_images": n_images,
        "total_edges": total_edges,
        "nonzero_pairs": nonzero,
        "nonzero_ratio": nonzero / n_pairs,
        "top_fracs": top_fracs,
        "norm_entropy": norm_entropy,
        "top_pairs": top_pairs,
    }


def process_object(obj, k, seed):
    print(f"\n=== {obj} ===", flush=True)
    train_feats = load_train_features(obj)
    print(f"  loaded {len(train_feats)} train_good images", flush=True)
    km = fit_kmeans(train_feats, k, seed)
    print(f"  fitted k-means k={k}", flush=True)
    counts = count_word_pairs(km, train_feats, k)
    summary = summarize_counts(obj, counts, len(train_feats), k)
    print(
        f"  nonzero={summary['nonzero_pairs']}/{k*(k+1)//2}, "
        f"top100={summary['top_fracs'][100]*100:.2f}%, "
        f"norm_entropy={summary['norm_entropy']:.4f}",
        flush=True,
    )
    return summary


def format_summary(summaries, k, seed):
    lines = [
        f"# word-pair frequency concentration diagnostic",
        f"# k={k}, seed={seed}, grid={GRID}x{GRID}, split=train_good, time={datetime.now().isoformat(timespec='seconds')}",
        f"# unordered right/down 4-neighbor word pairs; max_pairs={k*(k+1)//2}",
        "",
        "category               n_images  total_edges  nonzero  nonzero%  "
        "top10%  top50%  top100%  top200%  top500%  norm_entropy",
    ]

    for s in summaries:
        tf = s["top_fracs"]
        lines.append(
            f"{s['category']:22s}  {s['n_images']:8d}  {s['total_edges']:11d}  "
            f"{s['nonzero_pairs']:7d}  {s['nonzero_ratio']*100:8.2f}  "
            f"{tf[10]*100:6.2f}  {tf[50]*100:6.2f}  {tf[100]*100:7.2f}  "
            f"{tf[200]*100:7.2f}  {tf[500]*100:7.2f}  {s['norm_entropy']:12.4f}"
        )

    for s in summaries:
        lines.append("")
        lines.append(f"=== top 10 pairs: {s['category']} ===")
        lines.append("rank  pair     count    percent")
        for rank, (a, b, c, frac) in enumerate(s["top_pairs"], start=1):
            lines.append(f"{rank:4d}  ({a:2d},{b:2d})  {c:8d}  {frac*100:8.4f}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", nargs="+", default=DEFAULT_OBJS)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = [process_object(obj, args.k, args.seed) for obj in args.obj]

    report = format_summary(summaries, args.k, args.seed)
    print("\n" + report, flush=True)
    out_path = os.path.join(OUT_DIR, "word_pair_summary.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
