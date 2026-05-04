#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"

download_modelscope_repo() {
  local model_id="$1"
  local target_dir="$2"
  mkdir -p "$target_dir"
  echo "[DOWNLOAD][ModelScope] $model_id -> $target_dir"
  python - "$model_id" "$target_dir" <<'PY'
import sys

try:
    from modelscope import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "modelscope is not installed. Please run 'pip install modelscope' in the active environment first."
    ) from exc

model_id = sys.argv[1]
target_dir = sys.argv[2]
snapshot_download(model_id, local_dir=target_dir)
PY
}

echo "Using MODEL_ROOT=$MODEL_ROOT"
echo "Default download source: ModelScope"
echo "This script downloads only the weights with verified ModelScope mirrors in this project."

download_modelscope_repo "xingzi/diffuEraser" "$MODEL_ROOT/diffuEraser"
download_modelscope_repo "PAI/Wan2.1-Fun-1.3B-InP" "$MODEL_ROOT/Wan2.1-Fun-1.3B-InP"

mkdir -p \
  "$MODEL_ROOT/sam3" \
  "$MODEL_ROOT/sam3.1" \
  "$MODEL_ROOT/sd-vae-ft-mse" \
  "$MODEL_ROOT/stable-diffusion-v1-5" \
  "$MODEL_ROOT/ROSE_transformer"

echo
echo "[MANUAL][UPSTREAM] Please place the remaining weights into:"
echo "  $MODEL_ROOT/sam3                  <- SAM 3 gated checkpoint bundle"
echo "  $MODEL_ROOT/sam3.1                <- SAM 3.1 gated checkpoint bundle"
echo "  $MODEL_ROOT/sd-vae-ft-mse         <- sd-vae-ft-mse"
echo "  $MODEL_ROOT/stable-diffusion-v1-5 <- stable-diffusion-v1-5"
echo "  $MODEL_ROOT/ROSE_transformer      <- ROSE transformer weights"
echo
echo "Model directories are ready:"
echo "  $MODEL_ROOT/sam3"
echo "  $MODEL_ROOT/sam3.1"
echo "  $MODEL_ROOT/diffuEraser"
echo "  $MODEL_ROOT/sd-vae-ft-mse"
echo "  $MODEL_ROOT/stable-diffusion-v1-5"
echo "  $MODEL_ROOT/Wan2.1-Fun-1.3B-InP"
echo "  $MODEL_ROOT/ROSE_transformer"
