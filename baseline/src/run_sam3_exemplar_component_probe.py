"""Find repeated screw-bag components with a generated visual exemplar.

The exemplar and target are placed on one square canvas because SAM3's public
image processor officially supports same-image visual exemplars.  The target
image is kept at its native resolution; the exemplar occupies added padding.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = Path("/mnt/nfs/xujy/logicdataset/dataset_loco/screw_bag/train/good/011.png")
DEFAULT_EXEMPLARS = ROOT / "assets/component_exemplars/screw_bag"
DEFAULT_OUTPUT = ROOT / "baseline/results/sam3_exemplar_component_probe/screw_bag/011/search"
DEFAULT_SAM3_ROOT = ROOT / "third_party/sam3"
DEFAULT_CHECKPOINT = ROOT / "models/sam3_modelscope/sam3.pt"

# Tight object boxes in the generated exemplar images, as fractions of width/height.
EXEMPLAR_BOXES = {
    "nut": (0.16, 0.17, 0.84, 0.82),
    "washer": (0.16, 0.16, 0.84, 0.84),
    "bolt": (0.12, 0.32, 0.88, 0.69),
    "pushpin": (0.17, 0.28, 0.84, 0.72),
}
COLORS = {
    "nut": (245, 80, 80),
    "washer": (65, 165, 255),
    "bolt": (75, 220, 120),
    "pushpin": (245, 185, 55),
}
TEXT_PROMPTS = {
    "nut": "silver hex nut",
    "washer": "flat silver metal washer",
    "bolt": "silver threaded hex-head bolt",
    "pushpin": "yellow plastic pushpin with a metal needle",
}


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def make_canvas(target: Image.Image, exemplars: dict[str, Image.Image], exemplar_size: int):
    width, height = target.size
    canvas_size = max(width, height + exemplar_size, exemplar_size * len(exemplars))
    canvas = Image.new("RGB", (canvas_size, canvas_size), (18, 18, 18))
    canvas.paste(target, (0, 0))
    top = canvas_size - exemplar_size
    boxes = {}
    boxes_xyxy = {}
    for index, (kind, exemplar) in enumerate(exemplars.items()):
        exemplar = exemplar.resize((exemplar_size, exemplar_size), Image.Resampling.LANCZOS)
        left = index * exemplar_size
        canvas.paste(exemplar, (left, top))
        x0, y0, x1, y1 = EXEMPLAR_BOXES[kind]
        x0 = left + x0 * exemplar_size
        x1 = left + x1 * exemplar_size
        y0 = top + y0 * exemplar_size
        y1 = top + y1 * exemplar_size
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        boxes[kind] = [cx / canvas_size, cy / canvas_size,
                       (x1 - x0) / canvas_size, (y1 - y0) / canvas_size]
        boxes_xyxy[kind] = (x0, y0, x1, y1)
    return canvas, boxes, boxes_xyxy


def import_sam3(args):
    sys.path.insert(0, str(args.sam3_root))
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = args.sam3_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe), checkpoint_path=str(args.checkpoint),
        load_from_HF=False, device=args.device, enable_inst_interactivity=True,
    )
    model.eval()
    return Sam3Processor(
        model, device=args.device, confidence_threshold=args.confidence_threshold
    )


def inference_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def clean_target_masks(masks: np.ndarray, scores: np.ndarray, target_size: tuple[int, int],
                       min_area: int):
    width, height = target_size
    records = []
    for mask, score in zip(masks, scores):
        target_mask = mask[:height, :width].astype(bool)
        area = int(target_mask.sum())
        if area < min_area:
            continue
        records.append((target_mask, float(score), area))
    records.sort(key=lambda item: item[1], reverse=True)
    return records


def render_kind(target: Image.Image, kind: str, records, canvas: Image.Image,
                exemplar_box, output: Path):
    color = np.asarray(COLORS[kind], dtype=np.float32)
    values = np.asarray(target).astype(np.float32)
    for mask, _, _ in records:
        values[mask] = values[mask] * 0.55 + color * 0.45
    overlay = Image.fromarray(values.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for index, (mask, score, _) in enumerate(records):
        yy, xx = np.where(mask)
        if len(yy):
            draw.text((int(xx.mean()), int(yy.mean())), f"{index}:{score:.2f}",
                      fill="white", stroke_width=2, stroke_fill="black", font=font(22))
    overlay.save(output / f"{kind}_target_overlay.png")

    debug = canvas.copy()
    ddraw = ImageDraw.Draw(debug)
    ddraw.rectangle(exemplar_box, outline=(255, 255, 0), width=5)
    debug.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    debug.save(output / f"{kind}_prompt_canvas.png")


def render_combined(target: Image.Image, all_records: dict, output: Path):
    values = np.asarray(target).astype(np.float32)
    for kind, records in all_records.items():
        color = np.asarray(COLORS[kind], dtype=np.float32)
        for mask, _, _ in records:
            values[mask] = values[mask] * 0.55 + color * 0.45
    panel = Image.fromarray(values.astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    y = 12
    for kind in all_records:
        draw.rectangle((12, y, 34, y + 22), fill=COLORS[kind])
        draw.text((42, y), f"{kind}: {len(all_records[kind])}", fill="white",
                  stroke_width=2, stroke_fill="black", font=font(22))
        y += 34
    panel.save(output / "all_types_overlay.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--exemplar-root", type=Path, default=DEFAULT_EXEMPLARS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--exemplar-size", type=int, default=512)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--types", nargs="+", default=["nut", "washer", "bolt"])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    target = Image.open(args.image).convert("RGB")
    processor = import_sam3(args)
    all_records = {}
    metadata = {}
    exemplars = {
        kind: Image.open(args.exemplar_root / f"{kind}.png").convert("RGB")
        for kind in args.types
    }
    canvas, boxes, boxes_xyxy = make_canvas(target, exemplars, args.exemplar_size)
    for kind in args.types:
        with torch.inference_mode(), inference_context(args.device):
            state = processor.set_image(canvas)
            state = processor.set_text_prompt(TEXT_PROMPTS[kind], state=state)
            state = processor.add_geometric_prompt(box=boxes[kind], label=True, state=state)
        masks = state["masks"].squeeze(1).detach().cpu().numpy()
        scores = state["scores"].detach().float().cpu().numpy()
        records = clean_target_masks(masks, scores, target.size, args.min_area)
        all_records[kind] = records
        np.savez_compressed(
            args.output / f"{kind}_masks.npz",
            masks=np.stack([item[0] for item in records]) if records else
                  np.zeros((0, target.height, target.width), dtype=bool),
            scores=np.asarray([item[1] for item in records], dtype=np.float32),
        )
        metadata[kind] = {
            "n_predictions_total_canvas": int(len(masks)),
            "n_target_instances": int(len(records)),
            "areas": [item[2] for item in records],
            "scores": [item[1] for item in records],
            "normalized_exemplar_box_cxcywh": boxes[kind],
            "prompt_mode": "text_visual",
            "text_prompt": TEXT_PROMPTS[kind],
        }
        render_kind(target, kind, records, canvas, boxes_xyxy[kind], args.output)
        print(f"{kind}: total={len(masks)}, target={len(records)}", flush=True)
        del state
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    render_combined(target, all_records, args.output)
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
