"""Normal-only visual-word statistics for B0 and composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf


def load_split(feature_root: str, mask_root: str, obj: str, split: str):
    feature_dir = Path(feature_root) / obj / split
    mask_dir = Path(mask_root) / obj / split
    files = sorted(feature_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No feature files in {feature_dir}")
    features, weights, names = [], [], []
    for feature_path in files:
        feature = np.load(feature_path).astype(np.float32)
        mask_path = mask_dir / feature_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing foreground-weight cache: {mask_path}")
        weight = np.load(mask_path).astype(np.float32)
        if feature.ndim != 2 or weight.shape != (len(feature),):
            raise ValueError(f"Feature/mask mismatch: {feature_path}, {mask_path}")
        features.append(feature)
        weights.append(weight)
        names.append(feature_path.stem)
    return features, weights, np.asarray(names)


def coverage_outliers(weights: list[np.ndarray], z_threshold: float) -> np.ndarray:
    coverage = np.asarray([weight.mean() for weight in weights])
    scale = coverage.std()
    if scale < 1e-8:
        return np.zeros(len(coverage), dtype=bool)
    return np.abs((coverage - coverage.mean()) / scale) > z_threshold


def fit_vocabulary(
    features: list[np.ndarray],
    weights: list[np.ndarray],
    outliers: np.ndarray,
    config: dict,
    seed: int,
) -> MiniBatchKMeans:
    threshold = float(config["foreground_fit_threshold"])
    chunks = [
        feature[weight > threshold]
        for feature, weight, outlier in zip(features, weights, outliers)
        if not outlier and np.any(weight > threshold)
    ]
    if not chunks:
        raise RuntimeError("No foreground train patches remain after filtering")
    pooled = np.concatenate(chunks)
    limit = int(config["kmeans_fit_subsample"])
    if len(pooled) > limit:
        indices = np.random.RandomState(seed).choice(len(pooled), limit, replace=False)
        pooled = pooled[indices]
    model = MiniBatchKMeans(
        n_clusters=int(config["k"]), random_state=seed,
        batch_size=int(config["kmeans_batch_size"]), n_init="auto",
    )
    return model.fit(pooled)


def weighted_histograms(
    model: MiniBatchKMeans,
    features: list[np.ndarray],
    weights: list[np.ndarray],
    k: int,
) -> np.ndarray:
    output = np.zeros((len(features), k), dtype=np.float64)
    for index, (feature, weight) in enumerate(zip(features, weights)):
        labels = model.predict(feature)
        output[index] = np.bincount(labels, weights=weight, minlength=k)
    return output


def fit_mahalanobis(train: np.ndarray):
    estimator = LedoitWolf().fit(np.asarray(train, dtype=np.float64))
    def score(query: np.ndarray) -> np.ndarray:
        difference = np.asarray(query, dtype=np.float64) - estimator.location_
        return np.einsum("ij,jk,ik->i", difference, estimator.get_precision(), difference)
    return score


def normalize_histograms(histograms: np.ndarray, alpha: float) -> np.ndarray:
    histograms = np.asarray(histograms, dtype=np.float64)
    return (histograms + alpha) / (
        histograms.sum(axis=1, keepdims=True) + alpha * histograms.shape[1]
    )


def composition_scorer(train_histograms: np.ndarray, alpha: float):
    prototype = train_histograms.sum(axis=0) + alpha
    prototype = prototype / prototype.sum()
    def score(query_histograms: np.ndarray) -> np.ndarray:
        probabilities = normalize_histograms(query_histograms, alpha)
        return np.asarray([
            float(jensenshannon(prototype, row, base=2.0) ** 2)
            for row in probabilities
        ])
    return score
