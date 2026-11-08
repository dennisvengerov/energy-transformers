#!/usr/bin/env bash
# Train one depth, then evaluate it. Re-running resumes training from runs/L<N>/ckpt.pt.
#
#   bash scripts/run_depth.sh 12
set -euo pipefail
cd "$(dirname "$0")/.."
N_LAYER=${1:?usage: bash scripts/run_depth.sh N_LAYER}
export PYTHONUNBUFFERED=1

echo "$(date)  host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  n_layer=$N_LAYER"
.venv/bin/python train.py --n-layer "$N_LAYER"
.venv/bin/python evaluate.py --ckpt "runs/L$N_LAYER/ckpt.pt"
