# sam3_instance_features.py
# 把 sam3_segment_check.py 往前推一格:从"肉眼看分割质量"进到"实例级 DINO 表示 + 最便宜的 count 验证"。
#
# 这一步只回答一个最小成本的问题:
#   在 pushpins 上,光靠 SAM3 数出来的【实例数量】,能不能把 good / logical 分开?
# 如果 count-only 就有强信号 -> 论文骨架("实例级统计")立住了,再上类型分布/位置统计。
# 如果 count-only 分不开 -> 说明 pushpins 的逻辑异常不在"数量",别在 count 上继续投入。
#
# 三件事:
#   1) 对指定类别的 train_good / test_good / test_logical 跑 SAM3 文本提示分割,逐图得到实例 mask + 数量。
#   2) 把每个 mask 各向异性缩到 48x48(和 DINO 的 anisotropic-672 预处理对齐,672/14=48),
#      在【已缓存的 DINO 特征】上做 soft-weight masked average pooling,得到每实例 [1536] 向量并存盘。
#      —— 不重新过 DINO,直接复用你已验证的 features_cache,喂给下一步(实例集合打分)用。
#   3) 只用 train_good 拟合 count 中心值,对 test 打"偏离量"分数,算 logical AUROC。
#      严格无泄漏:参考统计只来自 train_good;good/logical 用同一套变换。
#
# SAM3 加载 / 推理接口完全照抄你已验证的 sam3_segment_check.py,没有自己发明 API。
#
# 用法:
#   python sam3_instance_features.py --checkpoint /path/to/sam3_model/sam3.pt --obj pushpins
#   python sam3_instance_features.py --checkpoint ... --obj pushpins --confidence 0.5 --no-save-feats

import os
import sys
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

DATASET_ROOT = "/mnt/nfs/xujy/logicdataset/datasets/mvtec_loco_anomaly_detection"
PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")
DEFAULT_SAM3_ROOT = os.path.join(PROJECT_ROOT, "third_party", "sam3")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "sam3_instances")

GRID = 48          # 672 / 14,和特征提取时行优先 patch 排列一致
FEAT_DIM = 1536    # DINOv2-with-registers-giant

SPLITS = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_logical": "test/logical_anomalies",
}

# 只有 pushpins 的提示词是你已验证过的。其它类别的提示词你还没验证,别用默认值直接信。
VALIDATED_PROMPTS = {
    "pushpins": "pushpin",
}


# ----------------------------- SAM3 加载(照抄 sam3_segment_check.py) -----------------------------

def resolve_checkpoint(path):
    if path is None:
        path = os.environ.get("SAM3_CHECKPOINT")
    if path is None:
        raise ValueError("请通过 --checkpoint 或环境变量 SAM3_CHECKPOINT 指定 sam3.pt")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def import_sam3(sam3_root):
    sam3_root = os.path.abspath(sam3_root)
    if not os.path.isdir(sam3_root):
        raise FileNotFoundError(f"SAM3 repo not found: {sam3_root}")
    sys.path.insert(0, sam3_root)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    return build_sam3_image_model, Sam3Processor, bpe_path


def inference_context(device):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def segment_instances(processor, image, prompt, device):
    """返回 (masks_np, scores_np)。masks_np: [N, H, W] bool(原图分辨率); scores_np: [N]。"""
    with inference_context(device):
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None:
        return np.zeros((0, image.height, image.width), dtype=bool), np.zeros((0,), dtype=np.float32)

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4:        # [N,1,H,W] -> [N,H,W]
        masks_np = masks_np[:, 0]
    masks_np = masks_np > 0.5
    scores_np = (scores.detach().float().cpu().numpy()
                 if scores is not None else np.ones(masks_np.shape[0], dtype=np.float32))
    return masks_np, scores_np


# ----------------------------- mask -> DINO 网格 -> 实例特征 -----------------------------

def mask_to_grid_weights(mask_2d, grid=GRID):
    """把原图分辨率的 bool mask 各向异性缩到 grid x grid 的 soft 覆盖率权重 [grid*grid]。
    各向异性缩放(忽略长宽比)刻意与 DINO 预处理的 anisotropic-672 对齐,保证 patch 对齐。
    BILINEAR 给出的是每个 patch 被 mask 覆盖的近似比例,做 soft-weight pooling 比硬阈值更稳
    (pushpins 很小,常常只占某个 patch 的一部分)。"""
    m = Image.fromarray((mask_2d.astype(np.uint8) * 255))
    m = m.resize((grid, grid), Image.BILINEAR)
    w = np.asarray(m, dtype=np.float32) / 255.0   # [grid, grid], 行优先
    return w.reshape(-1)                          # [grid*grid]


def load_dino_grid_feat(obj, split_name, basename):
    """读缓存的 [2304, 1536] float16 -> float32 [grid*grid, 1536]。"""
    p = os.path.join(CACHE_DIR, obj, split_name, basename + ".npy")
    feat = np.load(p).astype(np.float32)
    if feat.shape[0] != GRID * GRID:
        raise ValueError(f"{p}: expected {GRID*GRID} patches, got {feat.shape[0]}")
    return feat


def pool_instance_feats(grid_feat, masks_np, min_weight=1e-3):
    """对每个实例做 soft-weight masked average pooling。
    返回 (feats [M,1536], kept_idx) —— 过滤掉在 48x48 网格上几乎没落到任何 patch 的实例
    (mask 太小,缩到 48x48 后权重和≈0,pool 无意义)。"""
    inst_feats = []
    kept = []
    for i, mask in enumerate(masks_np):
        w = mask_to_grid_weights(mask)          # [grid*grid]
        wsum = w.sum()
        if wsum < min_weight:
            continue
        vec = (grid_feat * w[:, None]).sum(axis=0) / wsum   # [1536]
        inst_feats.append(vec)
        kept.append(i)
    if inst_feats:
        return np.stack(inst_feats, axis=0), np.array(kept, dtype=int)
    return np.zeros((0, FEAT_DIM), dtype=np.float32), np.array([], dtype=int)


# ----------------------------- 每类流程 -----------------------------

def list_images(obj, split_subdir):
    d = os.path.join(DATASET_ROOT, obj, split_subdir)
    if not os.path.isdir(d):
        return []
    fs = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    fs.sort()
    return fs


def process_split(processor, obj, split_name, split_subdir, prompt, device,
                  out_dir, save_feats):
    files = list_images(obj, split_subdir)
    counts = []
    names = []
    feat_dir = os.path.join(out_dir, obj, f"instances_{split_name}")
    if save_feats:
        os.makedirs(feat_dir, exist_ok=True)

    for f in files:
        basename = f.rsplit(".", 1)[0]
        img_path = os.path.join(DATASET_ROOT, obj, split_subdir, f)
        image = Image.open(img_path).convert("RGB")

        masks_np, scores_np = segment_instances(processor, image, prompt, device)
        grid_feat = load_dino_grid_feat(obj, split_name, basename)
        inst_feats, kept = pool_instance_feats(grid_feat, masks_np)

        count = int(inst_feats.shape[0])   # 用"落到网格上的有效实例数"作为 count
        counts.append(count)
        names.append(basename)

        if save_feats:
            np.savez(
                os.path.join(feat_dir, basename + ".npz"),
                feats=inst_feats.astype(np.float16),
                scores=scores_np[kept].astype(np.float16) if len(kept) else np.zeros(0, np.float16),
                count=count,
            )
        print(f"  [{obj}/{split_name}] {basename}: raw={masks_np.shape[0]} "
              f"grid_valid={count}", flush=True)

    return np.array(counts, dtype=float), names


def count_based_auroc(train_counts, good_counts, logical_counts):
    """零泄漏 count-only 打分: 参考中心值只来自 train_good,分数 = |count - 中心|。
    good=0 / logical=1,越偏离越异常。"""
    from sklearn.metrics import roc_auc_score
    center = np.median(train_counts)            # 只用 train_good
    good_score = np.abs(good_counts - center)
    logical_score = np.abs(logical_counts - center)
    y_true = np.concatenate([np.zeros(len(good_score)), np.ones(len(logical_score))])
    y_score = np.concatenate([good_score, logical_score])
    auroc = roc_auc_score(y_true, y_score)
    return center, auroc


def summarize(obj, train_c, good_c, logical_c):
    def stat(c):
        return (f"n={len(c)} mean={c.mean():.2f} median={np.median(c):.1f} "
                f"min={int(c.min())} max={int(c.max())}") if len(c) else "n=0"
    lines = [
        f"=== {obj} ===",
        f"  train_good    : {stat(train_c)}",
        f"  test_good     : {stat(good_c)}",
        f"  test_logical  : {stat(logical_c)}",
    ]
    if len(train_c) and len(good_c) and len(logical_c):
        center, auroc = count_based_auroc(train_c, good_c, logical_c)
        lines.append(f"  count-only reference center (train median) = {center:.1f}")
        lines.append(f"  count-only logical AUROC = {auroc:.4f}")
    else:
        lines.append("  [skip AUROC] 某个 split 为空")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pushpins",
                    choices=["breakfast_box", "juice_bottle", "pushpins",
                             "screw_bag", "splicing_connectors"])
    ap.add_argument("--prompt", default=None,
                    help="文本提示词。不给则用已验证映射(目前只有 pushpins->'pushpin')")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sam3-root", default=DEFAULT_SAM3_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-save-feats", action="store_true",
                    help="只做 count 验证,不落盘实例特征(更快)")
    args = ap.parse_args()

    prompt = args.prompt or VALIDATED_PROMPTS.get(args.obj)
    if prompt is None:
        raise SystemExit(
            f"[{args.obj}] 没有已验证的提示词。请用 --prompt 显式指定,"
            f"并先跑 sam3_segment_check.py 肉眼确认分割质量再上这一步。")

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    build_sam3_image_model, Sam3Processor, bpe_path = import_sam3(args.sam3_root)
    print(f"Loading SAM3 from {checkpoint} on {device}, prompt='{prompt}' ...", flush=True)
    model = build_sam3_image_model(
        bpe_path=bpe_path, checkpoint_path=checkpoint,
        load_from_HF=False, device=device,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence)

    save_feats = not args.no_save_feats
    per_split = {}
    for split_name, split_subdir in SPLITS.items():
        print(f"\n[{args.obj}/{split_name}]", flush=True)
        counts, names = process_split(processor, args.obj, split_name, split_subdir,
                                      prompt, device, args.out_dir, save_feats)
        per_split[split_name] = counts

    report = summarize(
        args.obj,
        per_split.get("train_good", np.array([])),
        per_split.get("test_good", np.array([])),
        per_split.get("test_logical", np.array([])),
    )
    print("\n" + report, flush=True)

    os.makedirs(os.path.join(args.out_dir, args.obj), exist_ok=True)
    out_txt = os.path.join(args.out_dir, args.obj, "count_summary.txt")
    with open(out_txt, "w") as f:
        f.write(f"# SAM3 instance count-only diagnostic, prompt='{prompt}', "
                f"confidence={args.confidence}\n")
        f.write(report + "\n")
    print(f"\nSaved summary to {out_txt}", flush=True)
    if save_feats:
        print(f"Instance feats saved under {args.out_dir}/{args.obj}/instances_<split>/", flush=True)


if __name__ == "__main__":
    main()
