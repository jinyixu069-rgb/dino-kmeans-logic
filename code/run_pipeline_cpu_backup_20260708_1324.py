# run_pipeline.py  (加入 RMD 背景修正开关)
# --score full   : 只用完整协方差马氏距离(= 你已验证的 0.8342 版本)
# --score bg     : 只用对角协方差背景马氏距离
# --score rmd     : RMD = MD_full - MD_background(对角协方差背景)
# --score both    : 两个都算,并排输出,方便直接对照
# --score all     : full/bg/rmd 都算,用于诊断 RMD 机制
#
# 用法:
#   python run_pipeline.py --all --seeds 0 1 2 3 4 5 --score all

import os
import argparse
import numpy as np
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OBJS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

DEFAULT_K = 64
KMEANS_FIT_SUBSAMPLE = 200_000


def load_split_features(obj, split_name):
    d = os.path.join(CACHE_DIR, obj, split_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    return [np.load(os.path.join(d, f)).astype(np.float32) for f in files]


def fit_kmeans(train_feats, k, subsample, seed):
    pooled = np.concatenate(train_feats, axis=0)
    n = pooled.shape[0]
    if n > subsample:
        idx = np.random.RandomState(seed).choice(n, subsample, replace=False)
        pooled = pooled[idx]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
    km.fit(pooled)
    return km


def build_histogram(km, feats_list, k):
    hists = np.zeros((len(feats_list), k), dtype=np.float64)
    for i, feat in enumerate(feats_list):
        hists[i] = np.bincount(km.predict(feat), minlength=k)
    return hists


def print_score_debug(name, good_s, logical_s):
    y_true = np.concatenate([np.zeros(len(good_s)), np.ones(len(logical_s))])
    y_score = np.concatenate([good_s, logical_s])
    auc = roc_auc_score(y_true, y_score)
    print(f"  {name}: auc={auc:.4f}", flush=True)
    for split, arr in (("good", good_s), ("logical", logical_s)):
        print(
            f"    {split}: min={arr.min():.6e}, p50={np.percentile(arr, 50):.6e}, "
            f"p95={np.percentile(arr, 95):.6e}, max={arr.max():.6e}",
            flush=True,
        )


def print_cov_debug(obj, seed, raw_var_diag, clipped_var_diag,
                    good_full, logical_full, good_bg, logical_bg, good_rmd, logical_rmd):
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    quantiles = np.percentile(raw_var_diag, qs)
    print(f"[debug-cov][seed={seed}] {obj}", flush=True)
    print(
        "  raw diag quantiles: "
        + ", ".join(f"p{q}={v:.6e}" for q, v in zip(qs, quantiles)),
        flush=True,
    )
    print(
        "  raw diag small counts: "
        f"<1e-8={(raw_var_diag < 1e-8).sum()}, "
        f"<1e-6={(raw_var_diag < 1e-6).sum()}, "
        f"<1e-4={(raw_var_diag < 1e-4).sum()}, "
        f"min/clipped_min={raw_var_diag.min():.6e}/{clipped_var_diag.min():.6e}",
        flush=True,
    )
    tiny_idx = np.argsort(raw_var_diag)[:10]
    print(
        "  10 smallest diag entries: "
        + ", ".join(f"{i}:{raw_var_diag[i]:.6e}" for i in tiny_idx),
        flush=True,
    )
    print_score_debug("full", good_full, logical_full)
    print_score_debug("bg", good_bg, logical_bg)
    print_score_debug("rmd", good_rmd, logical_rmd)


def run_one_category(obj, k, seed, score_mode, debug_cov=False):
    train_feats = load_split_features(obj, "train_good")
    good_feats = load_split_features(obj, "test_good")
    logical_feats = load_split_features(obj, "test_logical")

    km = fit_kmeans(train_feats, k, KMEANS_FIT_SUBSAMPLE, seed)
    train_hist = build_histogram(km, train_feats, k)
    good_hist = build_histogram(km, good_feats, k)
    logical_hist = build_histogram(km, logical_feats, k)

    # ---- 完整协方差(捕捉词与词之间的联合结构)----
    lw = LedoitWolf().fit(train_hist)
    mean_vec = lw.location_
    precision_full = lw.get_precision()

    # ---- 背景:对角协方差(只保留每个词自己的边际方差,丢掉词间相关)----
    raw_var_diag = np.diag(lw.covariance_).copy()
    var_diag = raw_var_diag.copy()
    var_diag[var_diag < 1e-8] = 1e-8            # 数值保护,避免除零
    precision_bg = np.diag(1.0 / var_diag)

    def maha(hists, precision):
        diff = hists - mean_vec
        return np.einsum("ij,jk,ik->i", diff, precision, diff)

    scores = {}
    good_full = maha(good_hist, precision_full)
    logical_full = maha(logical_hist, precision_full)
    good_bg = maha(good_hist, precision_bg)
    logical_bg = maha(logical_hist, precision_bg)
    good_rmd = good_full - good_bg
    logical_rmd = logical_full - logical_bg

    if debug_cov:
        print_cov_debug(obj, seed, raw_var_diag, var_diag,
                        good_full, logical_full, good_bg, logical_bg,
                        good_rmd, logical_rmd)

    if score_mode in ("full", "both", "all"):
        scores["full"] = (good_full, logical_full)
    if score_mode in ("bg", "all"):
        scores["bg"] = (good_bg, logical_bg)
    if score_mode in ("rmd", "both", "all"):
        scores["rmd"] = (good_rmd, logical_rmd)

    aucs = {}
    for name, (good_s, logical_s) in scores.items():
        y_true = np.concatenate([np.zeros(len(good_s)), np.ones(len(logical_s))])
        y_score = np.concatenate([good_s, logical_s])
        aucs[name] = roc_auc_score(y_true, y_score)
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", choices=OBJS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--score", choices=["full", "bg", "rmd", "both", "all"], default="both")
    ap.add_argument("--debug-cov", action="store_true",
                    help="print covariance diagonal diagnostics for RMD background")
    args = ap.parse_args()

    objs = OBJS if args.all else ([args.obj] if args.obj else None)
    if objs is None:
        print("请指定 --obj <category> 或 --all")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # results[score_name][obj] = list over seeds
    if args.score == "both":
        score_names = ["full", "rmd"]
    elif args.score == "all":
        score_names = ["full", "bg", "rmd"]
    else:
        score_names = [args.score]
    results = {sn: {obj: [] for obj in objs} for sn in score_names}

    for seed in args.seeds:
        for obj in objs:
            aucs = run_one_category(obj, args.k, seed, args.score, args.debug_cov)
            for sn in score_names:
                results[sn][obj].append(aucs[sn])
            msg = "  ".join(f"{sn}={aucs[sn]:.4f}" for sn in score_names)
            print(f"[seed={seed}] {obj:22s}: {msg}", flush=True)

    lines = [
        f"# k-means bag-of-words + Mahalanobis, RMD ablation (full-data logical AD)",
        f"# DINOv2-with-registers-giant, anisotropic resize 672, mean layers [-18,-12]",
        f"# k={args.k}, seeds={args.seeds}, score={args.score}, "
        f"time={datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for sn in score_names:
        lines.append(f"=== score = {sn} ===")
        lines.append(f"{'category':22s}  {'mean':>7s}  {'std':>6s}  per-seed")
        macro_ps = np.zeros(len(args.seeds))
        for obj in objs:
            arr = np.array(results[sn][obj])
            lines.append(f"{obj:22s}  {arr.mean():.4f}  {arr.std():.4f}  "
                         f"[{', '.join(f'{a:.4f}' for a in arr)}]")
        if args.all:
            for si in range(len(args.seeds)):
                macro_ps[si] = np.mean([results[sn][obj][si] for obj in objs])
            lines.append(f"{'MACRO':22s}  {macro_ps.mean():.4f}  {macro_ps.std():.4f}  "
                         f"[{', '.join(f'{m:.4f}' for m in macro_ps)}]")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"kmeans_rmd_k{args.k}_{stamp}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
