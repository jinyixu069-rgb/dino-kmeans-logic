# sam3_segment_check.py
# 用 SAM 3 文本概念提示验证各类别实例/前景分割质量。
# 只做肉眼诊断,不接入现有 anomaly pipeline。
#
# 权重建议从 ModelScope 下载到本地,然后用 --checkpoint 指向 sam3.pt:
#   modelscope download --model facebook/sam3 --local_dir /path/to/sam3_model
#   python sam3_segment_check.py --checkpoint /path/to/sam3_model/sam3.pt
#
# SAM 3 源码来自:
#   https://github.com/facebookresearch/sam3

import os
import sys
import json
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

DATASET_ROOT = "/mnt/nfs/xujy/logicdataset/datasets/mvtec_loco_anomaly_detection"
PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
DEFAULT_SAM3_ROOT = os.path.join(PROJECT_ROOT, "third_party", "sam3")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "sam3_check")
GENERATED_PROMPTS_PATH = os.path.join(
    PROJECT_ROOT, "results", "qwen_sam3_prompts", "generated_prompts.json")

SPLITS = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_logical": "test/logical_anomalies",
}

PROMPTS = {
    "breakfast_box": ["box"],
    "juice_bottle": ["bottle"],
    "pushpins": ["pushpin"],
    "screw_bag": ["screw"],
    "splicing_connectors": ["connector"],
}


def resolve_checkpoint(path):
    if path is None:
        path = os.environ.get("SAM3_CHECKPOINT")
    if path is None:
        raise ValueError(
            "请通过 --checkpoint 或环境变量 SAM3_CHECKPOINT 指定 ModelScope 下载的 sam3.pt"
        )
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


def overlay_masks_np(image, masks_np):
    overlay = np.array(image).astype(np.float32)
    rng = np.random.RandomState(0)

    for i, mask in enumerate(masks_np):
        color = rng.randint(0, 255, 3).astype(np.float32)
        m = mask.astype(bool)
        overlay[m] = overlay[m] * 0.4 + color * 0.6
    return overlay.astype(np.uint8)


def draw_boxes(ax, boxes, scores):
    if boxes is None:
        return
    boxes_np = boxes.detach().float().cpu().numpy()
    scores_np = scores.detach().float().cpu().numpy() if scores is not None else [None] * len(boxes_np)
    for box, score in zip(boxes_np, scores_np):
        x0, y0, x1, y1 = box.tolist()
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.5, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        if score is not None:
            ax.text(x0, max(0, y0 - 4), f"{float(score):.2f}",
                    color="lime", fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})


def inference_context(device):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def list_images(obj, split):
    d = os.path.join(DATASET_ROOT, obj, split)
    if not os.path.isdir(d):
        return []
    files = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    return files


def select_evenly(files, n):
    if n <= 0 or len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n, dtype=int)
    return [files[i] for i in np.unique(idx)]


def run_sam3_prompt(processor, image, prompt, device):
    with inference_context(device):
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")
    if masks is None or masks.shape[0] == 0:
        return np.zeros((0, image.height, image.width), dtype=bool), boxes, scores

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    return masks_np > 0.5, boxes, scores


def run_one(processor, image_path, prompts, out_path, device):
    image = Image.open(image_path).convert("RGB")

    all_masks = []
    all_boxes = []
    all_scores = []
    per_prompt_counts = {}
    for prompt in prompts:
        masks_np, boxes, scores = run_sam3_prompt(processor, image, prompt, device)
        per_prompt_counts[prompt] = int(masks_np.shape[0])
        if masks_np.shape[0]:
            all_masks.append(masks_np)
            if boxes is not None:
                all_boxes.append(boxes.detach().float().cpu().numpy())
            if scores is not None:
                all_scores.append(scores.detach().float().cpu().numpy())

    if all_masks:
        masks_np = np.concatenate(all_masks, axis=0)
    else:
        masks_np = np.zeros((0, image.height, image.width), dtype=bool)
    boxes_np = np.concatenate(all_boxes, axis=0) if all_boxes else None
    scores_np = np.concatenate(all_scores, axis=0) if all_scores else None
    n_instances = int(masks_np.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title(os.path.basename(image_path))
    axes[0].axis("off")

    if n_instances > 0:
        overlay = overlay_masks_np(image, masks_np)
    else:
        overlay = np.array(image)
    axes[1].imshow(overlay)
    if boxes_np is not None:
        for box, score in zip(boxes_np, scores_np if scores_np is not None else [None] * len(boxes_np)):
            x0, y0, x1, y1 = box.tolist()
            rect = plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.5, edgecolor="lime", facecolor="none"
            )
            axes[1].add_patch(rect)
            if score is not None:
                axes[1].text(x0, max(0, y0 - 4), f"{float(score):.2f}",
                             color="lime", fontsize=8,
                             bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})
    axes[1].set_title(f"{n_instances} instances for {prompts}")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n_instances, per_prompt_counts


def load_generated_prompts():
    if not os.path.exists(GENERATED_PROMPTS_PATH):
        raise FileNotFoundError(
            f"generated prompt manifest not found: {GENERATED_PROMPTS_PATH}")
    with open(GENERATED_PROMPTS_PATH) as f:
        return json.load(f)


def resolve_prompts(obj, cli_prompts, use_generated):
    if cli_prompts:
        return cli_prompts
    if use_generated:
        manifest = load_generated_prompts()
        if obj not in manifest:
            raise KeyError(f"{obj} not found in {GENERATED_PROMPTS_PATH}")
        return manifest[obj]
    return PROMPTS[obj]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="ModelScope 下载目录中的 sam3.pt 路径; 也可用 SAM3_CHECKPOINT 环境变量")
    ap.add_argument("--sam3-root", default=DEFAULT_SAM3_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--obj", nargs="+", default=["pushpins"],
                    choices=sorted(PROMPTS.keys()))
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="覆盖默认 prompt,支持多个短语并集")
    ap.add_argument("--use-generated-prompts", action="store_true",
                    help="从 results/qwen_sam3_prompts/generated_prompts.json 读取每类 prompt 列表")
    ap.add_argument("--n-per-split", type=int, default=6,
                    help="每个 split 均匀采样多少张图做可视化")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    if args.prompts is not None and len(args.obj) != 1:
        raise SystemExit("--prompts 覆盖只支持单个 --obj; 多类别请使用脚本内 PROMPTS 或 generated manifest")

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    build_sam3_image_model, Sam3Processor, bpe_path = import_sam3(args.sam3_root)
    print(f"Loading SAM3 from {checkpoint} on {device} ...", flush=True)
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint,
        load_from_HF=False,
        device=device,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence,
    )

    for obj in args.obj:
        prompts = resolve_prompts(obj, args.prompts, args.use_generated_prompts)
        obj_out_dir = os.path.join(args.out_dir, obj)
        os.makedirs(obj_out_dir, exist_ok=True)
        print(f"\n=== {obj}, prompts={prompts} ===", flush=True)

        for split_name, split_subdir in SPLITS.items():
            files = select_evenly(list_images(obj, split_subdir), args.n_per_split)
            print(f"[{split_name}] visualizing {len(files)} images", flush=True)
            for fname in files:
                img_path = os.path.join(DATASET_ROOT, obj, split_subdir, fname)
                if not os.path.exists(img_path):
                    print(f"[skip] not found: {img_path}", flush=True)
                    continue
                out_name = f"{split_name}_{fname}"
                out_path = os.path.join(obj_out_dir, out_name)
                n_instances, per_prompt_counts = run_one(processor, img_path, prompts, out_path, device)
                print(
                    f"  {obj}/{split_subdir}/{fname}: detected {n_instances} instances "
                    f"{per_prompt_counts}",
                    flush=True,
                )
                print(f"    saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
