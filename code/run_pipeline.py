# dual_branch_bow_pipeline.py
# 真正的双分支设计,替代 mask_weighted_bow_pipeline.py 里"同一词袋加权"的思路。
#
# 读了 SALAD (ICCV 2025, arXiv 2509.02101) 之后发现:它用 mask 的方式不是"给一个统计量去噪",
# 而是彻底的多分支架构 + 分数级融合(appearance / composition / global 三个分支各自独立打分,
# 用各分支在验证集上的 z-score 标准化后直接相加)。这和你原本设想的"双分支"是一回事,
# 和我上一版"同一词袋、只改计数权重"的思路不是一回事——这份脚本按你的原意重做。
#
# 两个分支完全独立,不共享词表也不共享统计量:
#   分支 A(不变,必须先复现锚点): 全图 patch 的 k-means 词袋 + LedoitWolf + Mahalanobis,
#                                和你现有 baseline 完全一致。
#   分支 B(新增,独立前景词袋): 只用 SAM3 前景 patch(权重 > 阈值)独立拟合一套 k-means 词表,
#                                独立统计直方图(仍用软权重计数,兼顾"贴边"的前景 patch),
#                                独立拟合 LedoitWolf + Mahalanobis。
#   融合: 用 train_good 各分支自己的 Mahalanobis 分数算 mean/std 做 z-score 标准化,
#         两分支标准化后相加,得到融合分数。只用 train 统计量做标准化,不碰测试标签,
#         对应 SALAD 论文 Eq.5 的融合方式(他们用验证集统计量,这里用 train_good 自身,
#         和你现有 pipeline 里 LedoitWolf 只在 train_good 上拟合的惯例一致)。
#
# 输出 A / B / Fused 三个 AUROC,而不是只看融合结果:
#   - A 必须先对上你已知的 pushpins 单类锚点(约 0.70-0.73),对不上说明代码有 bug。
#   - B 单独多高,直接反映"前景词袋本身有没有信号"。
#   - Fused 相对 max(A,B) 有没有提升,反映两个分支是否提供了互补信息
#     (如果 Fused 提升不明显,说明 A、B 抓到的可能是同一种信号,双分支意义不大)。
#
# 用法(先跑 cache_sam3_foreground_masks.py 缓存前景权重,再跑这个):
#   python dual_branch_bow_pipeline.py --obj pushpins --seeds 0 1

import os
import argparse
import numpy as np
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")
MASK_CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache_fgmask")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "dual_branch_bow")

DEFAULT_K = 64
KMEANS_FIT_SUBSAMPLE = 200_000
FG_FIT_THRESHOLD = 0.5   # 分支 B:只用前景权重 > 此阈值的 patch 拟合词表(词表纯度)


# ----------------------------- 数据加载 -----------------------------

def load_split(obj, split_name, cache_dir, mask_cache_dir):
    feat_dir = os.path.join(cache_dir, obj, split_name)
    mask_dir = os.path.join(mask_cache_dir, obj, split_name)
    files = sorted(f for f in os.listdir(feat_dir) if f.endswith(".npy"))

    feats, weights = [], []
    missing_mask = 0
    for f in files:
        basename = f.rsplit(".", 1)[0]
        feat = np.load(os.path.join(feat_dir, f)).astype(np.float32)
        mask_path = os.path.join(mask_dir, basename + ".npy")
        if os.path.exists(mask_path):
            w = np.load(mask_path).astype(np.float32)
        else:
            w = np.ones(feat.shape[0], dtype=np.float32)
            missing_mask += 1
        feats.append(feat)
        weights.append(w)

    if missing_mask:
        print(f"  [WARN] {obj}/{split_name}: {missing_mask}/{len(files)} 张图缺少缓存 mask,"
              f"已退化为全前景权重。分支 B 的独立性会被这些图拉低,先补跑 "
              f"cache_sam3_foreground_masks.py", flush=True)
    return feats, weights


# ----------------------------- k-means 拟合 -----------------------------

def fit_kmeans(pooled, k, seed, subsample=KMEANS_FIT_SUBSAMPLE):
    n = pooled.shape[0]
    if n > subsample:
        idx = np.random.RandomState(seed).choice(n, subsample, replace=False)
        pooled = pooled[idx]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
    km.fit(pooled)
    return km


def fit_branch_A(train_feats, k, seed):
    """全图词表,不看 mask。必须先复现已知锚点。"""
    pooled = np.concatenate(train_feats, axis=0)
    return fit_kmeans(pooled, k, seed)


def detect_coverage_outliers(train_weights, z_thresh=2.5):
    """基于 train_good 自身前景覆盖率分布做统计离群检测,两端都标:
    覆盖率异常高 -> 大概率是背景/前景检测失败、退化成整图前景(比如这次遇到的
    screw_bag/pushpins 没检出 tray/bag 的情况);覆盖率异常低 -> 前景基本没被
    检出。只用 train_good 自己的统计量,不碰任何测试信息,不需要任何模型判断——
    纯统计、确定性,可以在论文方法部分精确描述这个剔除规则。
    返回 bool 数组,True=离群(不参与词表/协方差拟合,但仍然正常参与打分,不丢数据)。"""
    cov = np.array([w.mean() for w in train_weights])
    mu, sd = cov.mean(), cov.std()
    if sd < 1e-8:
        return np.zeros(len(cov), dtype=bool)
    z = (cov - mu) / sd
    return np.abs(z) > z_thresh


def fit_branch_B(train_feats, train_weights, k, seed, threshold, outlier_mask=None):
    """只用前景 patch 独立拟合词表。outlier_mask 为 True 的图(覆盖率统计离群,
    大概率是分割失败、整图变前景)不参与拟合——不这样做的话,一张失败图能贡献出
    接近全图 2304 个 patch,而正常图可能只贡献两三百个,词表会被这一张图主导。"""
    if outlier_mask is None:
        outlier_mask = np.zeros(len(train_feats), dtype=bool)
    fg_chunks = []
    for feat, w, is_outlier in zip(train_feats, train_weights, outlier_mask):
        if is_outlier:
            continue
        keep = w > threshold
        if keep.any():
            fg_chunks.append(feat[keep])
    if not fg_chunks:
        raise RuntimeError(
            "剔除离群图后,没有 train_good 图在当前阈值下还有前景 patch,"
            "检查 mask 缓存是否正常,或调低 FG_FIT_THRESHOLD / z_thresh")
    pooled = np.concatenate(fg_chunks, axis=0)
    n_total = sum(f.shape[0] for f in train_feats)
    n_excluded = int(outlier_mask.sum())
    print(f"    分支B独立拟合: 前景 patch {pooled.shape[0]}/{n_total} "
          f"({pooled.shape[0]/n_total*100:.1f}%) 参与词表拟合"
          f"{f',剔除覆盖率离群图 {n_excluded}/{len(train_feats)} 张' if n_excluded else ''}",
          flush=True)
    return fit_kmeans(pooled, k, seed)


# ----------------------------- 直方图统计 -----------------------------

def build_hard_histogram(km, feats, k):
    hists = np.zeros((len(feats), k), dtype=np.float64)
    for i, feat in enumerate(feats):
        labels = km.predict(feat)
        hists[i] = np.bincount(labels, minlength=k)
    return hists


def build_weighted_histogram(km, feats, weights, k):
    """用前景软权重加权计数,即使不在阈值内的 patch 也按权重部分计入
    (阈值只用来决定谁能参与拟合词表,不用来决定谁能计入直方图,避免二次硬截断放大噪声)。"""
    hists = np.zeros((len(feats), k), dtype=np.float64)
    for i, (feat, w) in enumerate(zip(feats, weights)):
        labels = km.predict(feat)
        hists[i] = np.bincount(labels, weights=w, minlength=k)
    return hists


# ----------------------------- LedoitWolf + Mahalanobis(带 train 自身分数,便于做标准化) -----------------------------

def fit_ledoitwolf_and_score(train_hist, good_hist, logical_hist, train_fit_mask=None):
    """train_fit_mask 为 True 的图(覆盖率离群)不参与 LedoitWolf 拟合,避免污染协方差
    估计;但这些图仍然正常参与打分(train_s 对全部图计算),不丢数据,只是不让它们
    的坏统计进入模型本身。"""
    if train_fit_mask is not None and train_fit_mask.any():
        fit_hist = train_hist[~train_fit_mask]
        if fit_hist.shape[0] < 2:
            print(f"    [WARN] 剔除离群图后 train_good 只剩 {fit_hist.shape[0]} 张,"
                  f"不够拟合 LedoitWolf,回退为不剔除", flush=True)
            fit_hist = train_hist
    else:
        fit_hist = train_hist

    lw = LedoitWolf().fit(fit_hist)
    mean_vec = lw.location_
    precision = lw.get_precision()

    def maha(hist):
        diff = hist - mean_vec
        return np.einsum("ij,jk,ik->i", diff, precision, diff)

    train_s = maha(train_hist)     # 全部图都要打分(离群图也要),用于后续 z-score 标准化
    good_s = maha(good_hist)
    logical_s = maha(logical_hist)
    return train_s, good_s, logical_s


def auroc(good_s, logical_s):
    y_true = np.concatenate([np.zeros(len(good_s)), np.ones(len(logical_s))])
    y_score = np.concatenate([good_s, logical_s])
    return roc_auc_score(y_true, y_score)


def zscore(train_s, good_s, logical_s):
    mu, sd = train_s.mean(), train_s.std() + 1e-8
    return (good_s - mu) / sd, (logical_s - mu) / sd


def branch_correlation(good_za, logical_za, good_zb, logical_zb):
    """诊断用:A、B 两个分支的 z-score 在全部测试图(good+logical)上的相关系数。
    不参与任何打分或权重决策,不涉及测试标签的"设计选择",单纯解释
    "为什么融合没有比单分支好"这个机制性问题——相关性高说明 B 只是"去噪版的 A",
    两者没有互补信息;相关性低说明 A 里其实有 B 抓不到的独立信号,只是加法这种
    融合方式没用好。"""
    za = np.concatenate([good_za, logical_za])
    zb = np.concatenate([good_zb, logical_zb])
    return float(np.corrcoef(za, zb)[0, 1])


# ----------------------------- 单个 seed 的完整流程 -----------------------------

def run_one_seed(obj, k, seed, threshold, cache_dir, mask_cache_dir, outlier_z_thresh=2.5):
    train_feats, train_w = load_split(obj, "train_good", cache_dir, mask_cache_dir)
    good_feats, good_w = load_split(obj, "test_good", cache_dir, mask_cache_dir)
    logical_feats, logical_w = load_split(obj, "test_logical", cache_dir, mask_cache_dir)

    # ---- 分支 A:全图词袋(和现有 baseline 一致) ----
    km_a = fit_branch_A(train_feats, k, seed)
    train_hist_a = build_hard_histogram(km_a, train_feats, k)
    good_hist_a = build_hard_histogram(km_a, good_feats, k)
    logical_hist_a = build_hard_histogram(km_a, logical_feats, k)
    train_sa, good_sa, logical_sa = fit_ledoitwolf_and_score(
        train_hist_a, good_hist_a, logical_hist_a)
    auc_a = auroc(good_sa, logical_sa)

    # ---- 分支 B:独立前景词袋(剔除覆盖率离群图,防止个别 SAM3 分割失败图
    #      污染词表拟合和协方差估计。只用 train_good 自己的覆盖率分布判断,
    #      不碰测试标签,不需要任何模型判断,纯统计、确定性) ----
    outlier_mask = detect_coverage_outliers(train_w, z_thresh=outlier_z_thresh)
    if outlier_mask.any():
        print(f"    [离群检测] train_good 中 {int(outlier_mask.sum())}/{len(train_w)} "
              f"张图前景覆盖率统计离群(z>{outlier_z_thresh}),已从分支B的词表/协方差"
              f"拟合中剔除,但仍正常参与打分", flush=True)
    km_b = fit_branch_B(train_feats, train_w, k, seed, threshold, outlier_mask=outlier_mask)
    train_hist_b = build_weighted_histogram(km_b, train_feats, train_w, k)
    good_hist_b = build_weighted_histogram(km_b, good_feats, good_w, k)
    logical_hist_b = build_weighted_histogram(km_b, logical_feats, logical_w, k)
    train_sb, good_sb, logical_sb = fit_ledoitwolf_and_score(
        train_hist_b, good_hist_b, logical_hist_b, train_fit_mask=outlier_mask)
    auc_b = auroc(good_sb, logical_sb)

    # ---- 融合:各分支用 train_good 自己的分数做 z-score ----
    good_za, logical_za = zscore(train_sa, good_sa, logical_sa)
    good_zb, logical_zb = zscore(train_sb, good_sb, logical_sb)

    # 融合方式一:相加(SALAD 用的方式)
    good_sum = good_za + good_zb
    logical_sum = logical_za + logical_zb
    auc_sum = auroc(good_sum, logical_sum)

    # 融合方式二:取较大值。对"一个分支弱/带噪声"这种情况通常比相加更稳健——
    # 噪声分支只有在它比另一个分支更"确信异常"时才会被采纳,不会把强分支的
    # 高分拉低。用和你 crop 聚合代码里 --agg max 同一个思路,零额外算力
    # (复用同一批已经算好的 z-score,不用重新拟合)。
    good_max = np.maximum(good_za, good_zb)
    logical_max = np.maximum(logical_za, logical_zb)
    auc_max = auroc(good_max, logical_max)

    # 诊断:A、B 两分支的分数相关性,解释融合为什么有效/无效,不参与打分决策
    corr = branch_correlation(good_za, logical_za, good_zb, logical_zb)

    return auc_a, auc_b, auc_sum, auc_max, corr


# ----------------------------- 主入口 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pushpins",
                    choices=["breakfast_box", "juice_bottle", "pushpins",
                             "screw_bag", "splicing_connectors"])
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--fg-threshold", type=float, default=FG_FIT_THRESHOLD)
    ap.add_argument("--outlier-z-thresh", type=float, default=2.5,
                    help="分支B词表/协方差拟合时,train_good 前景覆盖率 z-score 超过"
                         "此阈值的图会被剔除出拟合(仍正常参与打分),用来防止个别 SAM3 "
                         "分割失败图污染词表和协方差估计")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--mask-cache-dir", default=MASK_CACHE_DIR)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {"A": [], "B": [], "Fused": [], "FusedMax": [], "corr": []}
    for seed in args.seeds:
        print(f"\n=== {args.obj} seed={seed} ===", flush=True)
        auc_a, auc_b, auc_sum, auc_max, corr = run_one_seed(
            args.obj, args.k, seed, args.fg_threshold, args.cache_dir, args.mask_cache_dir,
            outlier_z_thresh=args.outlier_z_thresh)
        print(f"  A (全图词袋, 现有 baseline)     : {auc_a:.4f}", flush=True)
        print(f"  B (独立前景词袋)                : {auc_b:.4f}", flush=True)
        print(f"  Fused-sum (A、B z-score 相加)   : {auc_sum:.4f}", flush=True)
        print(f"  Fused-max (A、B z-score 取大)   : {auc_max:.4f}", flush=True)
        print(f"  corr(A, B) z-score 相关系数     : {corr:.4f}  "
              f"(诊断用,不参与打分决策)", flush=True)
        results["A"].append(auc_a)
        results["B"].append(auc_b)
        results["Fused"].append(auc_sum)
        results["FusedMax"].append(auc_max)
        results["corr"].append(corr)

    lines = [
        f"# dual-branch BoW (global + independent foreground), obj={args.obj}, k={args.k}, "
        f"fg_threshold={args.fg_threshold}, seeds={args.seeds}, "
        f"time={datetime.now().isoformat(timespec='seconds')}",
        f"# 分支设计参考 SALAD (ICCV 2025, arXiv 2509.02101) 的多分支 + z-score 融合机制",
        "",
        "variant                    mean     std      per-seed",
    ]
    for name, label in [("A", "A (全图词袋)"), ("B", "B (独立前景词袋)"),
                        ("Fused", "Fused-sum (A+B)"), ("FusedMax", "Fused-max (A,B 取大)")]:
        arr = np.array(results[name])
        lines.append(f"{label:24s}  {arr.mean():.4f}  {arr.std():.4f}  "
                     f"[{', '.join(f'{a:.4f}' for a in arr)}]")
    corr_arr = np.array(results["corr"])
    lines.append(f"{'corr(A,B) 诊断':24s}  {corr_arr.mean():.4f}  {corr_arr.std():.4f}  "
                 f"[{', '.join(f'{c:.4f}' for c in corr_arr)}]  (不参与打分决策)")

    lines.append("")
    lines.append("解读提示:")
    lines.append("  - A 必须先对上你已知的 pushpins 单类锚点(约0.70-0.73);对不上说明这份代码有 bug,")
    lines.append("    B/Fused 的数字先不要采信。")
    lines.append("  - B 单独的数字反映'前景词袋'本身有没有信号,和 A 无关。")
    lines.append("  - corr(A,B) 低说明 A 里有 B 抓不到的独立信息,只是融合方式没用好;")
    lines.append("    corr(A,B) 高说明 B 基本就是'去噪版的 A',两分支本来就没有互补信息,")
    lines.append("    融合难有起色,应该考虑直接弃用 A、只用 B。")
    lines.append("  - Fused-max 若比 Fused-sum 好,说明问题出在'相加'这个融合方式本身")
    lines.append("    (被弱分支的噪声拖累),不是双分支思路错了。")
    lines.append("  - 若两种融合方式都不如 B,且 corr(A,B) 偏高,说明对 pushpins 这一类,")
    lines.append("    A 分支没有增量价值,建议在 breakfast_box 上再跑一遍同样的对比——")
    lines.append("    该类背景 AUROC 单独就有 0.95+,若那里 Fused 明显超过 B,说明")
    lines.append("    '融合是否有效'取决于该类别背景是否携带真实信号,而不是融合方式本身不行。")
    lines.append("    若 Fused 接近 max(A,B),说明两分支冗余,加分支的复杂度不划算。")

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{args.obj}_k{args.k}_{stamp}.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
