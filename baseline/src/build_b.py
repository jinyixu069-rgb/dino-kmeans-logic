"""Build B0 and composition scores from cached DINO/SAM patch features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import SPLITS, atomic_json, load_config
from .statistics import (
    composition_scorer,
    coverage_outliers,
    fit_mahalanobis,
    fit_vocabulary,
    load_split,
    weighted_histograms,
)


def load_object(config: dict, obj: str):
    return {
        split: load_split(config["feature_cache"], config["mask_cache"], obj, split)
        for split in SPLITS
    }


def build_object_seed(
    config: dict,
    obj: str,
    seed: int,
    overwrite: bool = False,
    data=None,
) -> Path:
    output = Path(config["output_root"]) / "b_scores" / obj / f"seed_{seed}.npz"
    if output.exists() and not overwrite:
        print(f"[skip] {output}", flush=True)
        return output

    if data is None:
        data = load_object(config, obj)
    train_features, train_weights, _ = data["train_good"]
    outliers = coverage_outliers(train_weights, float(config["coverage_outlier_z"]))
    fit_mask = ~outliers
    if fit_mask.sum() < 2:
        raise RuntimeError(f"Fewer than two usable train/good images: {obj}")
    vocabulary = fit_vocabulary(train_features, train_weights, outliers, config, seed)
    histograms = {
        split: weighted_histograms(vocabulary, features, weights, int(config["k"]))
        for split, (features, weights, _) in data.items()
    }
    b0_score = fit_mahalanobis(histograms["train_good"][fit_mask])
    composition_score = composition_scorer(
        histograms["train_good"][fit_mask], float(config["dirichlet_alpha"])
    )
    arrays = {
        "fit_mask": fit_mask,
        "cluster_centers": vocabulary.cluster_centers_.astype(np.float32),
    }
    for split in SPLITS:
        arrays[f"{split}_names"] = data[split][2]
        arrays[f"{split}_B0"] = b0_score(histograms[split])
        arrays[f"{split}_composition"] = composition_score(histograms[split])
    if any(not np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)):
        raise FloatingPointError(f"Non-finite B score for {obj}, seed={seed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    atomic_json(output.with_suffix(".json"), {
        "object": obj,
        "seed": seed,
        "normal_only_fit": True,
        "k": int(config["k"]),
        "coverage_outliers": int(outliers.sum()),
        "split_counts": {split: len(data[split][2]) for split in SPLITS},
    })
    print(f"[done] {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    objects = args.objects or config["objects"]
    seeds = args.seeds or config["seeds"]
    for obj in objects:
        pending = [
            int(seed) for seed in seeds
            if args.overwrite or not (
                Path(config["output_root"]) / "b_scores" / obj / f"seed_{seed}.npz"
            ).exists()
        ]
        if not pending:
            print(f"[skip] all requested seeds exist for {obj}", flush=True)
            continue
        data = load_object(config, obj)
        for seed in pending:
            build_object_seed(config, obj, seed, args.overwrite, data=data)


if __name__ == "__main__":
    main()
