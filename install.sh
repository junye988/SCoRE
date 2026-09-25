#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    python -m pip install torch --index-url "${TORCH_INDEX_URL}"
fi
python -m pip install -r requirements.txt
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'
printf '\nActivate the environment with: source %s/bin/activate\n' "${VENV_DIR}"
