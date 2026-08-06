# Training-free MVTec LOCO baseline

This directory contains the frozen current baseline:

`hierarchical-max(B0, composition) + S0 + S1`

- **B0**: K=16 SAM-weighted DINOv2 visual-word totals, scored with a
  train/good Ledoit-Wolf Mahalanobis model.
- **Composition**: the same K=16 histogram normalized to proportions, scored
  by Jensen-Shannon divergence from the train/good prototype.
- **S0**: frozen WRN50-2 PatchCore evidence at 224 x 224.
- **S1**: frozen WRN50-2 PatchCore evidence at an aspect-ratio-aware ~448^2
  pixel budget.

Every raw branch score is converted to finite-sample train/good upper-tail
evidence. B0 and Composition are combined by maximum and calibrated once more
on train/good. The final score is

```text
E(max(E(B0), E(Composition))) + E(S0) + E(S1)
```

No gradients, anomaly synthesis, test distribution calibration, or test labels
are used to construct the score. Test labels are read only by the evaluation
step after all scores have been written.

## Run

Copy the example configuration and edit only paths:

```bash
cp configs/baseline.example.yaml configs/baseline.local.yaml
```

Build B scores from the existing DINO/SAM feature caches:

```bash
python -m baseline.src.build_b --config baseline/configs/baseline.local.yaml
```

Fuse B with precomputed S0/S1 scores and evaluate:

```bash
python -m baseline.src.fuse --config baseline/configs/baseline.local.yaml
```

Or run both stages:

```bash
bash baseline/scripts/run_all.sh baseline/configs/baseline.local.yaml
```

The default protocol uses all five MVTec LOCO categories and seeds 0--3. The
reported reference result is logical/structural/overall AUROC
`0.9405 / 0.9007 / 0.9207`.

## Inputs not stored in Git

- MVTec LOCO images;
- DINO patch-feature cache;
- SAM foreground-weight cache;
- S0/S1 score archives;
- pretrained model weights.

The repository contains code and compact summary reports only.
