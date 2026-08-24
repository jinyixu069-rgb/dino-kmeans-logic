"""Fuse scored SAM3 semantic instances with SAM2 AMG residual regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = Path("/mnt/nfs/xujy/logicdataset/dataset_loco/screw_bag/train/good/011.png")
DEFAULT_SEARCH = ROOT / "baseline/results/sam3_exemplar_component_probe/screw_bag/011/search_text_visual"
DEFAULT_AMG = ROOT / "baseline/results/sam2_component_graph_all/amg/screw_bag/011/amg_masks.npz"
COLORS = {
    "nut": (245, 80, 80), "washer": (65, 165, 255), "bolt": (75, 220, 120),
    "pushpin": (245, 185, 55), "amg_residual": (190, 75, 220),
}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def load_candidates(root: Path, types):
    candidates = []
    for kind in types:
        data = np.load(root / f"{kind}_masks.npz")
        for mask, score in zip(data["masks"].astype(bool), data["scores"]):
            candidates.append({"type": kind, "mask": mask, "score": float(score)})
    return candidates


def resolve_semantic_candidates(candidates, duplicate_iou: float):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        conflicts = [prior for prior in kept if iou(candidate["mask"], prior["mask"]) >= duplicate_iou]
        if conflicts:
            continue
        kept.append(candidate)
    return kept


def snap_to_amg(semantic, amg_masks, snap_iou: float):
    used = set()
    for item in semantic:
        overlaps = [iou(item["mask"], mask) if idx not in used else -1.0
                    for idx, mask in enumerate(amg_masks)]
        best = int(np.argmax(overlaps))
        item["best_amg_iou"] = float(overlaps[best])
        item["best_amg_id"] = best
        item["snapped"] = bool(overlaps[best] >= snap_iou)
        if item["snapped"]:
            item["mask"] = amg_masks[best].copy()
            used.add(best)
    return used


def make_exclusive(semantic):
    occupied = np.zeros_like(semantic[0]["mask"], dtype=bool)
    for item in sorted(semantic, key=lambda value: value["score"], reverse=True):
        item["mask"] = item["mask"] & ~occupied
        occupied |= item["mask"]
    return occupied


def add_amg_residuals(semantic, amg_masks, occupied, used_amg, min_area, min_fraction,
                      max_components):
    residuals = []
    for idx, mask in enumerate(amg_masks):
        if idx in used_amg:
            continue
        residual = mask & ~occupied
        fraction = float(residual.sum() / max(int(mask.sum()), 1))
        if residual.sum() < min_area or fraction < min_fraction:
            continue
        residuals.append({
            "type": "amg_residual", "mask": residual, "score": 0.0,
            "amg_id": idx, "residual_fraction": fraction, "snapped": False,
        })
    residuals.sort(key=lambda item: int(item["mask"].sum()), reverse=True)
    for item in residuals[:max(0, max_components - len(semantic))]:
        item["mask"] &= ~occupied
        occupied |= item["mask"]
        semantic.append(item)
    return occupied


def render(image: Image.Image, components, output: Path):
    values = np.asarray(image).astype(np.float32)
    for item in components:
        color = np.asarray(COLORS[item["type"]], dtype=np.float32)
        values[item["mask"]] = values[item["mask"]] * 0.50 + color * 0.50
    overlay = Image.fromarray(values.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    counts = {}
    for item in components:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
        yy, xx = np.where(item["mask"])
        if len(yy):
            label = f"{item['type']} {counts[item['type']]}"
            draw.text((int(xx.mean()), int(yy.mean())), label, fill="white",
                      stroke_width=2, stroke_fill="black", font=font(20))
    overlay.save(output / "fused_partition_overlay.png")

    width = 720
    height = round(image.height * width / image.width)
    left = image.resize((width, height), Image.Resampling.LANCZOS)
    right = overlay.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width * 2, height + 42), "white")
    canvas.paste(left, (0, 42)); canvas.paste(right, (width, 42))
    header = ImageDraw.Draw(canvas)
    header.text((10, 9), "original", fill="black", font=font(20))
    header.text((width + 10, 9), "SAM3 semantic + compatible AMG edges", fill="black", font=font(20))
    canvas.save(output / "fusion_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--amg", type=Path, default=DEFAULT_AMG)
    parser.add_argument("--output", type=Path, default=DEFAULT_SEARCH / "fusion")
    parser.add_argument("--duplicate-iou", type=float, default=0.80)
    parser.add_argument("--snap-iou", type=float, default=0.80)
    parser.add_argument("--residual-min-area", type=int, default=1000)
    parser.add_argument("--residual-min-fraction", type=float, default=0.50)
    parser.add_argument("--max-components", type=int, default=30)
    parser.add_argument("--types", nargs="+", default=["nut", "washer", "bolt"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    candidates = load_candidates(args.search_root, args.types)
    semantic = resolve_semantic_candidates(candidates, args.duplicate_iou)
    amg_masks = np.load(args.amg)["masks"].astype(bool)
    used_amg = snap_to_amg(semantic, amg_masks, args.snap_iou)
    occupied = make_exclusive(semantic)
    occupied = add_amg_residuals(
        semantic, amg_masks, occupied, used_amg, args.residual_min_area,
        args.residual_min_fraction, args.max_components,
    )
    background = ~occupied

    masks = [item["mask"] for item in semantic] + [background]
    types = [item["type"] for item in semantic] + ["background"]
    np.savez_compressed(
        args.output / "exclusive_component_masks.npz",
        masks=np.stack(masks), types=np.asarray(types), background_index=len(semantic),
    )
    serializable = []
    for index, item in enumerate(semantic):
        serializable.append({
            key: value for key, value in {
                **item, "component_id": index, "area": int(item["mask"].sum())
            }.items() if key != "mask"
        })
    summary = {
        "n_raw_sam3_candidates": len(candidates),
        "n_semantic_instances": sum(item["type"] != "amg_residual" for item in semantic),
        "n_amg_residual_instances": sum(item["type"] == "amg_residual" for item in semantic),
        "n_components_including_background": len(masks),
        "foreground_coverage": float(occupied.mean()),
        "components": serializable,
    }
    (args.output / "fusion_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render(image, semantic, args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
