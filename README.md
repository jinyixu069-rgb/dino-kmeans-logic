# Training-free logical and structural anomaly detection

Research code for MVTec LOCO anomaly detection using frozen visual encoders and
normal-only statistical modeling. No gradient optimization is used by the
current method.

## Current baseline

The frozen baseline is documented and implemented in [`baseline/`](baseline/):

```text
hierarchical-max(B0, composition) + S0 + S1
```

- B0 models SAM-weighted DINO visual-word totals.
- Composition independently models normalized visual-word proportions.
- S0/S1 are frozen WRN PatchCore memories at two input resolutions.
- Every calibration statistic is estimated from `train/good` only.

Five-category MVTec LOCO image-level AUROC:

| Logical | Structural | Overall |
|---:|---:|---:|
| 0.9405 | 0.9007 | 0.9207 |

This is a full-data, training-free protocol: all available normal training
images are used to form non-parametric/statistical reference models.

## Repository layout

- `baseline/`: frozen current method, portable configuration, tests, and result summary.
- `code/`: DINO/SAM preprocessing and earlier branch-B implementation.
- `structural_patch_memory/`: S0/S1 extraction and memory-bank experiments.
- `relational_word_statistics/`: relation and composition ablations; not part of the current baseline.

Datasets, pretrained weights, feature caches, runtime caches, and per-image
result archives are intentionally excluded from Git.

## Status

This repository is currently intended for private research collaboration. The
reported LOCO configuration was selected during exploratory development; use a
frozen protocol on additional data before making a strict SOTA claim.
