#!/usr/bin/env bash
# Local end-to-end smoke run (CPU/MPS, ~10 min): stream ~17k TinyGSM rows, train a tiny model for a
# few hundred steps, evaluate on a few GSM8K problems. Uses data/smoke and runs/smoke only.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY prepare_data.py --out-dir data/smoke --sample-row-groups 1 --val-every 50
$PY train.py --data-dir data/smoke --run-dir runs/smoke --n-layer 2 --d-model 128 --n-head 4 --d-ff 352 \
  --seq-len 512 --batch-size 16 --grad-accum 1 --max-steps 300 --warmup-steps 30 --log-every 50 \
  --eval-every 100 --eval-batches 5 --ckpt-every 100
$PY evaluate.py --ckpt runs/smoke/ckpt.pt --data-dir data/smoke --limit 20 --ppl-limit 300 --max-new-tokens 128
