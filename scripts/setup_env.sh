#!/usr/bin/env bash
# Create the Python environment in ./.venv with PyTorch built for CUDA 12.6 (driver 560 supports it).
# Nothing is cached outside the project directory.
#
#   bash scripts/setup_env.sh                          # python3.11 from PATH, else conda
#   PYTHON=/path/to/python3.11 bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_VERSION=2.14.1
TORCH_INDEX=https://download.pytorch.org/whl/cu126
PYTHON=${PYTHON:-python3.11}

if [ ! -x .venv/bin/python ]; then
  if command -v "$PYTHON" >/dev/null 2>&1; then
    "$PYTHON" -m venv .venv
  elif command -v conda >/dev/null 2>&1; then
    CONDA_PKGS_DIRS="$PWD/.cache/conda-pkgs" conda create -y -p "$PWD/.venv" -c conda-forge --override-channels python=3.11 pip
  else
    echo "Need python3.11 on PATH (e.g. 'module load python/3.11') or conda." >&2
    exit 1
  fi
fi

.venv/bin/python -m pip install --no-cache-dir --upgrade pip
.venv/bin/python -m pip install --no-cache-dir "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA build', torch.version.cuda, '| GPU visible here:', torch.cuda.is_available())"
