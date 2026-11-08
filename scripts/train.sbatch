#!/bin/bash
# Train + evaluate one depth on one GPU. Submit from the project directory:
#
#   mkdir -p logs
#   sbatch --job-name=L12 --export=ALL,N_LAYER=12 scripts/train.sbatch
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=CHANGE_ME   # if your lab uses partitions/accounts, set them here
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
bash scripts/run_depth.sh "${N_LAYER:?pass N_LAYER via --export=ALL,N_LAYER=12}"
