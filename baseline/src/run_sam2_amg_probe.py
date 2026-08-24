"""Run official SAM2AutomaticMaskGenerator and render its raw proposals.

This deliberately uses SAM2's packaged AMG rather than a hand-written loop of
interactive point prompts, so its stability filtering, crop handling, and NMS
match the official implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_ROOT = Path("/mnt/nfs/xujy/logicdataset/dataset_loco/juice_bottle/train/good")
DEFAULT_BOTTLE_ROOT = ROOT / "features_cache_fgmask/juice_bottle/train_good"
DEFAULT_CHECKPOINT = ROOT / "models/sam2.1_hiera_large.pt"
DEFAULT_OUTPUT = ROOT / "baseline/results/sam2_amg_probe"


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def bottle_to_image(bottle: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray((bottle.reshape(48, 48) >= 0.30).astype(np.uint8) * 255).resize(
            size, Image.Resampling.NEAREST
        )
    ) > 127


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def render_proposals(output: Path, image: Image.Image, records: list[dict]) -> None:
    palette = [(245, 80, 80), (80, 160, 255), (80, 220, 120), (250, 185, 55),
               (190, 75, 220), (55, 215, 215), (245, 125, 65), (150, 220, 90),
               (245, 100, 165), (150, 150, 255), (230, 210, 80), (80, 200, 165)]
    values = np.asarray(image).astype(np.float32)
    for index, record in enumerate(records):
        color = np.asarray(palette[index % len(palette)], dtype=np.float32)
        values[record["mask"]] = values[record["mask"]] * 0.72 + color * 0.28
    combined = Image.fromarray(values.astype(np.uint8))
    draw = ImageDraw.Draw(combined)
    for index, record in enumerate(records):
        yy, xx = np.where(record["mask"])
        if len(yy):
            draw.text((int(xx.mean()), int(yy.mean())), str(index), fill="white", stroke_width=2,
                      stroke_fill="black", font=font(22))
    combined.save(output / "amg_all_filtered.png")

    width = 320
    height = round(image.height * width / image.width)
    panels = []
    for index, record in enumerate(records):
        color = np.asarray(palette[index % len(palette)], dtype=np.float32)
        values = np.asarray(image).astype(np.float32)
        values[record["mask"]] = values[record["mask"]] * 0.50 + color * 0.50
        panel = Image.fromarray(values.astype(np.uint8))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 12), f"{index}: area={record['area_fraction']:.3f}", fill="white",
                  stroke_width=2, stroke_fill="black", font=font(20))
        panels.append(panel.resize((width, height), Image.Resampling.LANCZOS))
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    grid = Image.new("RGB", (columns * width, rows * height), "black")
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % columns) * width, (index // columns) * height))
    grid.save(output / "amg_filtered_grid.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basename", default="000")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--bottle-root", type=Path, default=DEFAULT_BOTTLE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--max-proposals", type=int, default=24)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.85)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--min-area-fraction", type=float, default=0.0005)
    args = parser.parse_args()

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    output = args.output_root / args.basename
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image_root / f"{args.basename}.png").convert("RGB")
    image_array = np.asarray(image)
    bottle = bottle_to_image(np.load(args.bottle_root / f"{args.basename}.npy"), image.size)
    print("Loading SAM2.1 large", flush=True)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(args.checkpoint), device="cuda")
    generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=args.points_per_side, points_per_batch=64,
        pred_iou_thresh=args.pred_iou_thresh, stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        output_mode="binary_mask",
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        raw = generator.generate(image_array)
    print(f"Official AMG generated {len(raw)} raw masks", flush=True)
    filtered = []
    for ann in raw:
        mask = ann["segmentation"].astype(bool)
        area_fraction = float(mask.mean())
        inside_bottle = float((mask & bottle).sum() / max(int(mask.sum()), 1))
        if not (args.min_area_fraction <= area_fraction <= 0.55 and inside_bottle >= 0.75):
            continue
        filtered.append({
            "mask": mask, "area_fraction": area_fraction, "inside_bottle": inside_bottle,
            "predicted_iou": float(ann["predicted_iou"]),
            "stability_score": float(ann["stability_score"]), "bbox": ann["bbox"],
            "point_coords": ann["point_coords"],
        })
    # NMS retains one candidate per near-duplicate region, but leaves nested masks intact.
    filtered.sort(key=lambda item: (item["predicted_iou"], item["stability_score"]), reverse=True)
    kept = []
    for candidate in filtered:
        if any(mask_iou(candidate["mask"], prior["mask"]) >= 0.93 for prior in kept):
            continue
        kept.append(candidate)
        if len(kept) == args.max_proposals:
            break
    render_proposals(output, image, kept)
    np.savez_compressed(output / "amg_masks.npz", masks=np.stack([item["mask"] for item in kept]))
    serializable = [{key: value for key, value in item.items() if key != "mask"} for item in kept]
    (output / "amg_metadata.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(f"Retained {len(kept)} bottle-interior proposals: {output}", flush=True)


if __name__ == "__main__":
    main()
