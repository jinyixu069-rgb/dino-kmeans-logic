# run_pipeline.py
# OT 版本 (路线甲): 直接在原始 patch 特征集合上做图-对-图的 Sinkhorn 最优传输,
# 异常分数 = 测试图对正常参考集的 (最小 k 个 OT 距离的均值)。
# 彻底甩掉 k-means 直方图 + 高斯/Mahalanobis 假设。
#
# 关键近似(都可调,都会打进报告):
#   --patch-sample M   : 每张图随机采 M 个 patch 再算 OT(默认 512;2304=全用但慢 ~20x)
#   --ref-sample R     : 参考集只用 R 张正常图(默认 0=全用)
#   --topk-ref k       : 异常分数取"对参考集最小的 k 个 OT 距离"的均值(默认 5)
#
# 用法(第一版,先看量级):
#   python run_pipeline.py --obj pushpins breakfast_box --seeds 0 --patch-sample 512 --topk-ref 5

import os
import argparse
import numpy as np
from datetime import datetime

import torch
from geomloss import SamplesLoss
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OBJS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

# Sinkhorn 参数
SINKHORN_P = 2         # cost = 欧氏距离的 p 次方
SINKHORN_BLUR = 0.05   # 正则强度(越小越接近精确 OT 但越慢/越不稳)
SINKHORN_SCALING = 0.5 # 多尺度退火比例,0.5 是速度/精度折中


def load_split_features(obj, split_name):
    d = os.path.join(CACHE_DIR, obj, split_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    return [np.load(os.path.join(d, f)).astype(np.float32) for f in files]


def subsample_patches(feat, m, rng):
    """feat: [N, D] numpy. 随机采 m 个 patch(m<=0 或 m>=N 时全用)。"""
    n = feat.shape[0]
    if m <= 0 or m >= n:
        return feat
    idx = rng.choice(n, m, replace=False)
    return feat[idx]


def prepare_set(feats_list, m, rng, device):
    """把一个 split 的每张图处理成 [M, D] 的 GPU tensor 列表。"""
    out = []
    for f in feats_list:
        sub = subsample_patches(f, m, rng)
        out.append(torch.from_numpy(sub).to(device))
    return out


def ot_scores_for_split(test_sets, ref_sets, loss_fn, topk):
    """对每张测试图,算它对所有参考图的 OT 距离,取最小 topk 个的均值当分数。"""
    scores = np.zeros(len(test_sets))
    for i, t in enumerate(test_sets):
        dists = torch.empty(len(ref_sets), device=t.device)
        for j, r in enumerate(ref_sets):
            dists[j] = loss_fn(t, r)
        k = min(topk, len(ref_sets))
        scores[i] = torch.topk(dists, k, largest=False).values.mean().item()
    return scores


def run_one_category(obj, seed, patch_sample, ref_sample, topk, device_str):
    device = torch.device(device_str)
    rng = np.random.RandomState(seed)

    train_feats = load_split_features(obj, "train_good")
    good_feats = load_split_features(obj, "test_good")
    logical_feats = load_split_features(obj, "test_logical")

    if ref_sample > 0 and ref_sample < len(train_feats):
        ref_idx = rng.choice(len(train_feats), ref_sample, replace=False)
        train_feats = [train_feats[i] for i in ref_idx]

    ref_sets = prepare_set(train_feats, patch_sample, rng, device)
    good_sets = prepare_set(good_feats, patch_sample, rng, device)
    logical_sets = prepare_set(logical_feats, patch_sample, rng, device)

    loss_fn = SamplesLoss("sinkhorn", p=SINKHORN_P, blur=SINKHORN_BLUR, scaling=SINKHORN_SCALING)

    good_s = ot_scores_for_split(good_sets, ref_sets, loss_fn, topk)
    logical_s = ot_scores_for_split(logical_sets, ref_sets, loss_fn, topk)

    y_true = np.concatenate([np.zeros(len(good_s)), np.ones(len(logical_s))])
    y_score = np.concatenate([good_s, logical_s])
    auc = roc_auc_score(y_true, y_score)
    return auc, len(ref_sets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", nargs="+", default=["pushpins", "breakfast_box"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--patch-sample", type=int, default=512,
                    help="每张图采样多少 patch 算 OT(2304=全用但慢)")
    ap.add_argument("--ref-sample", type=int, default=0,
                    help="参考集用多少张正常图(0=全用)")
    ap.add_argument("--topk-ref", type=int, default=5,
                    help="异常分数取对参考集最小的 k 个 OT 距离的均值")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    device_str = "cuda:0" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {obj: [] for obj in args.obj}
    ref_used = {}
    import time
    for seed in args.seeds:
        for obj in args.obj:
            t0 = time.time()
            auc, n_ref = run_one_category(obj, seed, args.patch_sample,
                                          args.ref_sample, args.topk_ref, device_str)
            dt = time.time() - t0
            results[obj].append(auc)
            ref_used[obj] = n_ref
            print(f"[seed={seed}] {obj:22s}: AUROC={auc:.4f}  (n_ref={n_ref}, {dt:.1f}s)", flush=True)

    lines = [
        f"# OT (Sinkhorn) image-to-image on raw patch features (full-data logical AD)",
        f"# DINOv2-with-registers-giant, patch_sample={args.patch_sample}, "
        f"ref_sample={args.ref_sample or 'all'}, topk_ref={args.topk_ref}",
        f"# sinkhorn p={SINKHORN_P}, blur={SINKHORN_BLUR}, scaling={SINKHORN_SCALING}, "
        f"seeds={args.seeds}, device={device_str}, time={datetime.now().isoformat(timespec='seconds')}",
        "",
        f"{'category':22s}  {'mean':>7s}  {'std':>6s}  per-seed",
    ]
    for obj in args.obj:
        arr = np.array(results[obj])
        lines.append(f"{obj:22s}  {arr.mean():.4f}  {arr.std():.4f}  "
                     f"[{', '.join(f'{a:.4f}' for a in arr)}]")

    report = "\n".join(lines)
    print("\n" + report)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"ot_ps{args.patch_sample}_top{args.topk_ref}_{stamp}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
