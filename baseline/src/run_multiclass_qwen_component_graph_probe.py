"""Extend the SAM2 mask-ID + Qwen node graph probe beyond juice_bottle.

This is deliberately a validation runner, not a tuned benchmark implementation:
each object uses the full-image-Qwen medoid as its single automatic anchor, the
30 normal images nearest the normal centroid, and all official test images.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from code.logic_prism.qwen3_vl_embedding_adapter import Qwen3VLEmbedder
from .common import metric_set, upper_tail_evidence
from .run_qwen_node_semantic_graph_probe import INSTRUCTION, item_canvas_set, semantic_scores
from .run_sam2_component_graph_probe import graph_features, track


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/nfs/xujy/logicdataset/dataset_loco")
FULL_QWEN = ROOT / "baseline/results/qwen_embeddings"
OUTPUT = ROOT / "baseline/results/sam2_component_graph_all"
AMG = OUTPUT / "amg"
TRACKED = OUTPUT / "tracked"
CHECKPOINT = ROOT / "models/sam2.1_hiera_large.pt"
SAM2_ROOT = ROOT / "third_party/sam2"
MODEL_PATH = Path(os.environ.get(
    "QWEN3_VL_EMBEDDING_PATH", ROOT / "models/Qwen3-VL-Embedding-8B"
))
OBJECTS = ["breakfast_box", "pushpins", "screw_bag", "splicing_connectors"]
ANCHOR_OVERRIDES = {
    "pushpins": "037",
    "screw_bag": "011",
}
FUSED_MASKS = {
    "pushpins": ROOT / "baseline/results/sam3_exemplar_component_probe/pushpins/037/search_text_visual/fusion/exclusive_component_masks.npz",
    "screw_bag": ROOT / "baseline/results/sam3_exemplar_component_probe/screw_bag/011/search_text_visual/fusion/exclusive_component_masks.npz",
}
SPLITS = {
    "test_good": "test/good",
    "test_logical": "test/logical_anomalies",
    "test_structural": "test/structural_anomalies",
}


def norm(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def select_train_and_anchor(obj: str, n_train: int = 30) -> tuple[str, list[str]]:
    pairs = []
    for path in sorted((FULL_QWEN / obj / "train_good").glob("*.npy")):
        pairs.append((path.stem, norm(np.load(path))))
    if len(pairs) < n_train:
        raise RuntimeError(f"{obj}: only {len(pairs)} cached normal embeddings")
    centroid = norm(np.mean([vector for _, vector in pairs], axis=0))
    pairs.sort(key=lambda pair: 1.0 - float(pair[1] @ centroid))
    anchor = ANCHOR_OVERRIDES.get(obj, pairs[0][0])
    names = [name for name, _ in pairs[:n_train]]
    if anchor not in names:
        names[-1] = anchor
    return anchor, names


def load_source_masks(obj: str, anchor: str) -> tuple[np.ndarray, str]:
    fused_path = FUSED_MASKS.get(obj)
    if fused_path is not None and fused_path.exists():
        cached = np.load(fused_path)
        masks = cached["masks"].astype(bool)
        types = cached["types"].astype(str)
        masks = masks[types != "background"]
        return masks, "sam3_text_visual_plus_amg"
    mask_path = AMG / obj / anchor / "amg_masks.npz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing AMG masks for automatic anchor {obj}/{anchor}: {mask_path}")
    return np.load(mask_path)["masks"].astype(bool), "sam2_amg"


def all_jobs(obj: str, train_names: list[str]) -> list[dict]:
    data = DATA_ROOT / obj
    jobs = [
        {"object": obj, "split": "train_good", "name": name,
         "path": data / "train/good" / f"{name}.png"}
        for name in train_names
    ]
    for split, subdir in SPLITS.items():
        for path in sorted((data / subdir).glob("*.png")):
            jobs.append({"object": obj, "split": split, "name": path.stem, "path": path})
    return jobs


def fit_geometry(items: list[dict]):
    features = np.stack([item["graph_feature"] for item in items])
    scaler = StandardScaler().fit(features)
    covariance = LedoitWolf().fit(scaler.transform(features))

    def score(item: dict) -> float:
        query = scaler.transform(item["graph_feature"][None])[0]
        delta = query - covariance.location_
        return float(delta @ covariance.get_precision() @ delta)

    return score, float(covariance.shrinkage_)


def load_or_track(predictor, obj: str, anchor: str, source_masks: np.ndarray,
                  jobs: list[dict]) -> list[dict]:
    source = Image.open(DATA_ROOT / obj / "train/good" / f"{anchor}.png").convert("RGB")
    items = []
    for index, item in enumerate(jobs):
        output = TRACKED / obj / item["split"] / f"{item['name']}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            cached = np.load(output)
            masks = cached["masks"].astype(bool)
            confidence = cached["confidence"].astype(np.float32)
        else:
            target = Image.open(item["path"]).convert("RGB")
            masks, confidence = track(predictor, source, target, source_masks)
            np.savez_compressed(output, masks=masks, confidence=confidence)
        item["masks"] = masks
        item["confidence"] = confidence
        item["graph_feature"] = graph_features(masks, confidence)
        items.append(item)
        if (index + 1) % 20 == 0 or index + 1 == len(jobs):
            print(f"{obj}: tracked {index + 1}/{len(jobs)}", flush=True)
    return items


def cache_node_embeddings(model, items: list[dict], batch_size: int = 8) -> None:
    pending = [
        item for item in items
        if not (OUTPUT / "node_embeddings" / item["object"] / item["split"] /
                f"{item['name']}.npy").exists()
    ]
    for index, item in enumerate(pending):
        canvases = item_canvas_set(item["path"], item["masks"])
        vectors = []
        for start in range(0, len(canvases), batch_size):
            batch = [{"image": image, "instruction": INSTRUCTION}
                     for image in canvases[start:start + batch_size]]
            vectors.append(model.process(batch, normalize=True).float().cpu().numpy())
        output = (OUTPUT / "node_embeddings" / item["object"] / item["split"] /
                  f"{item['name']}.npy")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.concatenate(vectors).astype(np.float32))
        if (index + 1) % 20 == 0 or index + 1 == len(pending):
            print(f"node embeddings: {index + 1}/{len(pending)}", flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selections = {}
    jobs_by_object = {}
    for obj in OBJECTS:
        anchor, train_names = select_train_and_anchor(obj)
        source_masks, source_kind = load_source_masks(obj, anchor)
        selections[obj] = {"anchor": anchor, "train_names": train_names,
                           "n_nodes_without_background": int(len(source_masks)),
                           "source_mask_kind": source_kind}
        jobs_by_object[obj] = (all_jobs(obj, train_names), source_masks)
    (OUTPUT / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")

    os.chdir(SAM2_ROOT)
    from sam2.build_sam import build_sam2_video_predictor
    print("Loading SAM2 video predictor", flush=True)
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", str(CHECKPOINT), device="cuda"
    )
    all_items = []
    for obj in OBJECTS:
        jobs, source_masks = jobs_by_object[obj]
        all_items.extend(load_or_track(
            predictor, obj, selections[obj]["anchor"], source_masks, jobs
        ))
    del predictor
    torch.cuda.empty_cache()

    pending = any(
        not (OUTPUT / "node_embeddings" / item["object"] / item["split"] /
             f"{item['name']}.npy").exists()
        for item in all_items
    )
    if pending:
        print("Loading Qwen node embedder", flush=True)
        embedder = Qwen3VLEmbedder(
            model_name_or_path=str(MODEL_PATH), output_dim=512,
            min_pixels=4096, max_pixels=262144, device="cuda", torch_dtype=torch.bfloat16,
        )
        cache_node_embeddings(embedder, all_items)
        del embedder
        torch.cuda.empty_cache()

    for item in all_items:
        item["embedding"] = np.load(
            OUTPUT / "node_embeddings" / item["object"] / item["split"] /
            f"{item['name']}.npy"
        ).astype(np.float32)

    diagnostics = {}
    for obj in OBJECTS:
        train = [item for item in all_items if item["object"] == obj and item["split"] == "train_good"]
        fit, calibration = train[:20], train[20:]
        bank = np.stack([item["embedding"] for item in fit])
        geometry, shrinkage = fit_geometry(fit)
        semantic_calibration = []
        geometry_calibration = []
        for item in calibration:
            _, semantic = semantic_scores(item["embedding"], bank)
            semantic_calibration.append(semantic)
            geometry_calibration.append(geometry(item))
        tests = [item for item in all_items if item["object"] == obj and item["split"] != "train_good"]
        for item in tests:
            node_distances, semantic = semantic_scores(item["embedding"], bank)
            geometric = geometry(item)
            item["semantic_joint"] = semantic
            item["semantic_max_node"] = float(node_distances.max())
            item["geometry"] = geometric
            item["semantic_evidence"] = float(upper_tail_evidence(
                np.asarray(semantic_calibration), np.asarray([semantic]))[0])
            item["geometry_evidence"] = float(upper_tail_evidence(
                np.asarray(geometry_calibration), np.asarray([geometric]))[0])
            item["fused"] = item["semantic_evidence"] + item["geometry_evidence"]
        diagnostics[obj] = {"geometry_shrinkage": shrinkage}

    tests = [item for item in all_items if item["split"] != "train_good"]
    rows = []
    metrics = {}
    for obj in OBJECTS:
        subset = [item for item in tests if item["object"] == obj]
        subtypes = np.asarray([{
            "test_good": "good", "test_logical": "logical_anomalies",
            "test_structural": "structural_anomalies",
        }[item["split"]] for item in subset])
        metrics[obj] = {}
        for scorer in ("semantic_joint", "semantic_max_node", "geometry", "fused"):
            metrics[obj][scorer] = metric_set(subtypes, np.asarray([item[scorer] for item in subset]))
        for item in subset:
            rows.append({key: item[key] for key in (
                "object", "split", "name", "semantic_joint", "semantic_max_node",
                "geometry", "semantic_evidence", "geometry_evidence", "fused",
            )})
    macro = {}
    for scorer in ("semantic_joint", "semantic_max_node", "geometry", "fused"):
        macro[scorer] = {
            metric: float(np.mean([metrics[obj][scorer][metric] for obj in OBJECTS]))
            for metric in ("logical", "structural", "overall")
        }
    fields = list(rows[0])
    with (OUTPUT / "scores_other_four.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "single automatic Qwen-medoid anchor; 20 normal fit + 10 normal calibration; all tests",
        "selection": selections,
        "metrics": metrics,
        "macro_other_four": macro,
        "diagnostics": diagnostics,
    }
    (OUTPUT / "summary_other_four.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"metrics": metrics, "macro_other_four": macro}, indent=2), flush=True)


if __name__ == "__main__":
    main()
