# visualize_clusters.py
# 诊断工具: 看 k-means 视觉词在 pushpins / splicing_connectors 这类卡住的类别上,
# 聚类质量到底纯不纯 —— 而不是继续猜"缺什么信息"。
#
# 输出两样东西:
#   1. results/cluster_viz/silhouette_summary.txt   —— 每个类别的聚类质量数值(silhouette score)
#   2. results/cluster_viz/{obj}/{split}_{filename}.png —— 每张采样图的三联图:
#      [原图] | [叠加色块的原图] | [纯色块图]
#      色块按 patch 归属的视觉词上色,64 个词用 64 种最大区分度的颜色。
#
# 用法:
#   python visualize_clusters.py                                   # 默认对比 pushpins/splicing/breakfast_box
#   python visualize_clusters.py --obj pushpins screw_bag           # 自定义类别
#   python visualize_clusters.py --obj pushpins --n-per-split 5     # 每个 split 多采样几张
#   python visualize_clusters.py --obj pushpins --k 128             # 换 K 看聚类是否变纯

import os
import argparse
import colorsys
import random

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")
DATASET_ROOT = "/mnt/nfs/xujy/logicdataset/datasets/mvtec_loco_anomaly_detection"
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "cluster_viz")

DEFAULT_OBJS = ["pushpins", "splicing_connectors", "breakfast_box"]
DEFAULT_K = 64
KMEANS_FIT_SUBSAMPLE = 200_000
SILHOUETTE_SAMPLE = 5000   # silhouette 计算开销大,用子采样估计,不需要全量

IMAGE_SIZE = 672
GRID = 48  # 672 / 14

SPLIT_SUBDIRS = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_logical": "test/logical_anomalies",
}


# ----------------------------- 基础工具 -----------------------------

def load_split_features_with_names(obj, split_name):
    d = os.path.join(CACHE_DIR, obj, split_name)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    feats = [np.load(os.path.join(d, f)).astype(np.float32) for f in files]
    basenames = [f.rsplit(".", 1)[0] for f in files]
    return feats, basenames


def find_original_image(obj, split_subdir, basename):
    src_dir = os.path.join(DATASET_ROOT, obj, split_subdir)
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(src_dir, basename + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到原图: {src_dir}/{basename}.*")


def generate_palette(k):
    """生成 k 个尽量区分度大的颜色,用 HSV 均匀分布再转 RGB。"""
    colors = []
    for i in range(k):
        hue = i / k
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colors, dtype=np.uint8)  # [k, 3]


def labels_to_color_image(labels, grid, palette, out_size):
    """labels: [grid*grid] -> 上色后放大到 out_size x out_size 的 RGB 图 (nearest, 保持色块边界清晰)。"""
    color_grid = palette[labels].reshape(grid, grid, 3)  # [grid, grid, 3]
    img = Image.fromarray(color_grid, mode="RGB")
    return img.resize((out_size, out_size), Image.NEAREST)


def blend(orig_img, color_img, alpha):
    return Image.blend(orig_img.convert("RGB"), color_img.convert("RGB"), alpha)


def make_triplet(orig_img, overlay_img, color_img, pad=6):
    w, h = orig_img.size
    canvas = Image.new("RGB", (w * 3 + pad * 2, h), (255, 255, 255))
    canvas.paste(orig_img, (0, 0))
    canvas.paste(overlay_img, (w + pad, 0))
    canvas.paste(color_img, (2 * (w + pad), 0))
    return canvas


# ----------------------------- 主流程 -----------------------------

def process_object(obj, k, n_per_split, alpha, seed, rng):
    print(f"\n=== {obj} ===")
    train_feats, _ = load_split_features_with_names(obj, "train_good")

    # 拟合 k-means: 和主 pipeline 的 CPU 版逻辑保持一致(同样的 subsample/seed 策略),
    # 这样看到的聚类质量对应的就是你已验证的那个方法本身,而不是另一套实现。
    pooled = np.concatenate(train_feats, axis=0)
    n = pooled.shape[0]
    if n > KMEANS_FIT_SUBSAMPLE:
        idx = np.random.RandomState(seed).choice(n, KMEANS_FIT_SUBSAMPLE, replace=False)
        fit_pool = pooled[idx]
    else:
        fit_pool = pooled
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init="auto")
    km.fit(fit_pool)

    # ---- 数值诊断: silhouette score(子采样估计,避免全量 O(n^2) 太贵) ----
    if pooled.shape[0] > SILHOUETTE_SAMPLE:
        sil_idx = np.random.RandomState(seed).choice(pooled.shape[0], SILHOUETTE_SAMPLE, replace=False)
        sil_X = pooled[sil_idx]
    else:
        sil_X = pooled
    sil_labels = km.predict(sil_X)
    # 极端情况下某次子采样可能只采到 1 个簇,silhouette_score 会报错,做个保护
    if len(set(sil_labels.tolist())) < 2:
        sil = float("nan")
    else:
        sil = silhouette_score(sil_X, sil_labels)
    print(f"  silhouette score (n={len(sil_X)} samples, k={k}): {sil:.4f}")

    # ---- 可视化: 每个 split 采样几张图,画三联图 ----
    palette = generate_palette(k)
    obj_out_dir = os.path.join(OUT_DIR, obj)
    os.makedirs(obj_out_dir, exist_ok=True)

    for split_name, split_subdir in SPLIT_SUBDIRS.items():
        feats, names = load_split_features_with_names(obj, split_name)
        if not feats:
            continue
        idxs = list(range(len(feats)))
        rng.shuffle(idxs)
        idxs = idxs[:n_per_split]

        for i in idxs:
            feat = feats[i]
            basename = names[i]
            labels = km.predict(feat)  # [grid*grid]

            img_path = find_original_image(obj, split_subdir, basename)
            orig_img = Image.open(img_path).convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)  # 和特征提取时的预处理保持一致,保证 patch 对齐

            color_img = labels_to_color_image(labels, GRID, palette, IMAGE_SIZE)
            overlay_img = blend(orig_img, color_img, alpha)
            triplet = make_triplet(orig_img, overlay_img, color_img)

            out_path = os.path.join(obj_out_dir, f"{split_name}_{basename}.png")
            triplet.save(out_path)
            print(f"  saved {out_path}")

    return sil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", nargs="+", default=DEFAULT_OBJS)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--n-per-split", type=int, default=3,
                    help="每个 split(train_good/test_good/test_logical)采样几张图做可视化")
    ap.add_argument("--alpha", type=float, default=0.5, help="叠加色块的透明度,0=纯原图,1=纯色块")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    summary_lines = [f"# cluster quality diagnostic, k={args.k}, seed={args.seed}", ""]
    for obj in args.obj:
        sil = process_object(obj, args.k, args.n_per_split, args.alpha, args.seed, rng)
        summary_lines.append(f"{obj:22s}  silhouette={sil:.4f}")

    summary_path = os.path.join(OUT_DIR, "silhouette_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n{'='*60}")
    print("\n".join(summary_lines))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved visualizations under {OUT_DIR}/<obj>/")


if __name__ == "__main__":
    main()
