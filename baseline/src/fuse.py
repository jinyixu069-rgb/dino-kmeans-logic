"""Frozen hierarchical B-composition plus S0/S1 fusion and evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import (
    SPLITS,
    TEST_SPLITS,
    atomic_json,
    load_config,
    metric_set,
    test_layout,
    upper_tail_evidence,
)


def load_s_scores(path: Path, aggregation: str):
    payload = np.load(path)
    names = {split: payload[f"{split}_names"].astype(str) for split in SPLITS}
    scores = {
        split: payload[f"{split}_{aggregation}"].astype(np.float64)
        for split in SPLITS
    }
    return names, scores


def concatenate(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([values[split] for split in TEST_SPLITS])


def score_without_labels(
    b_payload,
    s0_names: dict[str, np.ndarray],
    s0_scores: dict[str, np.ndarray],
    s1_names: dict[str, np.ndarray],
    s1_scores: dict[str, np.ndarray],
):
    """Construct final scores using only normal reference and raw query scores."""
    b_names = {split: b_payload[f"{split}_names"].astype(str) for split in SPLITS}
    for split in SPLITS:
        if not np.array_equal(b_names[split], s0_names[split]):
            raise ValueError(f"B/S0 filename mismatch in {split}")
        if not np.array_equal(b_names[split], s1_names[split]):
            raise ValueError(f"B/S1 filename mismatch in {split}")

    b0_train = b_payload["train_good_B0"].astype(np.float64)
    composition_train = b_payload["train_good_composition"].astype(np.float64)
    b0_test = concatenate({split: b_payload[f"{split}_B0"] for split in TEST_SPLITS})
    composition_test = concatenate({split: b_payload[f"{split}_composition"] for split in TEST_SPLITS})

    e_b0_train = upper_tail_evidence(b0_train, b0_train)
    e_composition_train = upper_tail_evidence(composition_train, composition_train)
    e_b0_test = upper_tail_evidence(b0_train, b0_test)
    e_composition_test = upper_tail_evidence(composition_train, composition_test)

    # The inner maximum changes the null distribution, hence the second
    # train/good calibration before adding independent S evidence.
    bmax_train = np.maximum(e_b0_train, e_composition_train)
    bmax_test = np.maximum(e_b0_test, e_composition_test)
    e_b_hierarchical = upper_tail_evidence(bmax_train, bmax_test)

    s0_test, s1_test = concatenate(s0_scores), concatenate(s1_scores)
    e_s0 = upper_tail_evidence(s0_scores["train_good"], s0_test)
    e_s1 = upper_tail_evidence(s1_scores["train_good"], s1_test)
    final = e_b_hierarchical + e_s0 + e_s1
    components = {
        "e_B0": e_b0_test,
        "e_composition": e_composition_test,
        "e_B_hierarchical": e_b_hierarchical,
        "e_S0": e_s0,
        "e_S1": e_s1,
        "final": final,
    }
    if any(not np.isfinite(value).all() for value in components.values()):
        raise FloatingPointError("Non-finite fused score")
    return b_names, components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    config = load_config(args.config)
    objects = args.objects or config["objects"]
    seeds = args.seeds or config["seeds"]
    aggregation = config["s_aggregation"]
    output_root = Path(config["output_root"])
    object_seed_metrics, score_rows = [], []

    for obj in objects:
        s0_path = Path(config["s0_results"]) / obj / "patchcore_standard" / "s_scores.npz"
        s1_path = Path(config["s1_results"]) / obj / "s1_wrn_highres" / "s_scores.npz"
        s0_names, s0_scores = load_s_scores(s0_path, aggregation)
        s1_names, s1_scores = load_s_scores(s1_path, aggregation)
        for seed in seeds:
            b_path = output_root / "b_scores" / obj / f"seed_{seed}.npz"
            if not b_path.exists():
                raise FileNotFoundError(f"Run build_b first: {b_path}")
            with np.load(b_path) as b_payload:
                names, components = score_without_labels(
                    b_payload, s0_names, s0_scores, s1_names, s1_scores
                )
            basenames, subtypes = test_layout(names)
            metrics = metric_set(subtypes, components["final"])
            object_seed_metrics.append({"object": obj, "seed": int(seed), **metrics})
            for index, basename in enumerate(basenames):
                score_rows.append({
                    "object": obj, "seed": int(seed), "basename": basename,
                    "subtype": subtypes[index],
                    **{key: float(value[index]) for key, value in components.items()},
                })

    per_object = []
    for obj in objects:
        rows = [row for row in object_seed_metrics if row["object"] == obj]
        per_object.append({
            "object": obj,
            **{name: float(np.mean([row[name] for row in rows]))
               for name in ("logical", "structural", "overall")},
        })
    macro = {
        name: float(np.mean([row[name] for row in per_object]))
        for name in ("logical", "structural", "overall")
    }
    result_dir = output_root / "fused"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader(); writer.writerows(score_rows)
    report = {
        "method": "hierarchical_max_B0_composition_plus_S0_plus_S1",
        "normal_only_calibration": True,
        "s_aggregation": aggregation,
        "macro": macro,
        "per_object": per_object,
        "per_object_seed": object_seed_metrics,
    }
    atomic_json(result_dir / "summary.json", report)
    lines = ["# Current training-free baseline", "",
             "`hierarchical max(B0, composition) + S0 + S1`", "",
             "| Object | Logical | Structural | Overall |", "|---|---:|---:|---:|"]
    for row in per_object:
        lines.append(f"| {row['object']} | {row['logical']:.4f} | {row['structural']:.4f} | {row['overall']:.4f} |")
    lines += [f"| **Macro** | **{macro['logical']:.4f}** | **{macro['structural']:.4f}** | **{macro['overall']:.4f}** |", ""]
    (result_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
