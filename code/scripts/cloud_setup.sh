#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/root/autodl-tmp/AIC_Challenge}"

cd "$PROJECT_DIR"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "transformers>=4.48,<5" accelerate safetensors

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the rented image and PyTorch build.")
print("gpu:", torch.cuda.get_device_name(0))
print("gpu memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

python scripts/audit_data.py --config configs/preliminary.yaml

echo "Cloud environment and formal dataset audit passed."
