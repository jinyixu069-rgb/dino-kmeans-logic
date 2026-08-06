"""Shared configuration, calibration, and evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score


SPLITS = ("train_good", "test_good", "test_logical", "test_structural")
TEST_SPLITS = ("test_good", "test_logical", "test_structural")


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    required = ("feature_cache", "mask_cache", "s0_results", "s1_results", "output_root")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing configuration keys: {missing}")
    for key in required:
        config[key] = str(Path(config[key]).expanduser().resolve())
    return config


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def upper_tail_evidence(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Finite-sample upper-tail evidence fitted on normal reference scores."""
    reference = np.sort(np.asarray(train, dtype=np.float64))
    query = np.asarray(query, dtype=np.float64)
    pvalue = (1.0 + len(reference) - np.searchsorted(reference, query, side="left")) / (
        len(reference) + 1.0
    )
    return -np.log(pvalue)


def test_layout(names: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    basenames = np.r_[
        np.char.add("good/", names["test_good"].astype(str)),
        np.char.add("logical_anomalies/", names["test_logical"].astype(str)),
        np.char.add("structural_anomalies/", names["test_structural"].astype(str)),
    ]
    subtypes = np.r_[
        np.repeat("good", len(names["test_good"])),
        np.repeat("logical_anomalies", len(names["test_logical"])),
        np.repeat("structural_anomalies", len(names["test_structural"])),
    ]
    return basenames, subtypes


def metric_set(subtypes: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Evaluation-only helper; scoring functions never receive these labels."""
    labels = (subtypes != "good").astype(np.int64)
    result = {"overall": float(roc_auc_score(labels, scores))}
    for name, subtype in (("logical", "logical_anomalies"), ("structural", "structural_anomalies")):
        keep = (subtypes == "good") | (subtypes == subtype)
        result[name] = float(roc_auc_score(labels[keep], scores[keep]))
    return result
