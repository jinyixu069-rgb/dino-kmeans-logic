#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 baseline/configs/baseline.local.yaml" >&2
  exit 2
fi

python -m baseline.src.build_b --config "$1"
python -m baseline.src.fuse --config "$1"
