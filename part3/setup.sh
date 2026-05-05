#!/usr/bin/env bash
# ============================================================
# Part 3 Setup Script
# Clone external repositories and download model weights.
# This script assumes the Part 3 Python environment is already activated.
# Usage:
#   bash setup.sh
#   bash setup.sh --source hf
#   bash setup.sh --source auto
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/external/repository"
MODEL_ROOT="$SCRIPT_DIR/models"
PART2_PROPainter_DIR="$SCRIPT_DIR/../part2/external/ProPainter"
SOURCE="modelscope"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: bash setup.sh [--source modelscope|hf|auto]"
      exit 1
      ;;
  esac
done

if [[ "$SOURCE" != "modelscope" && "$SOURCE" != "hf" && "$SOURCE" != "auto" ]]; then
  echo "[ERROR] Invalid source: $SOURCE"
  echo "Expected one of: modelscope, hf, auto"
  exit 1
fi

mkdir -p "$REPO_DIR" "$MODEL_ROOT" "$(dirname "$PART2_PROPainter_DIR")"

clone_if_missing() {
  local url="$1"
  local target_dir="$2"
  if [[ -d "$target_dir/.git" ]]; then
    echo "[=] Repo already exists: $target_dir"
  else
    echo "[+] Cloning $url -> $target_dir"
    env LD_LIBRARY_PATH="" git clone "$url" "$target_dir"
  fi
}

ensure_python_pkg() {
  local module_name="$1"
  local pip_name="$2"
  if python - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module_name")
PY
  then
    echo "[=] Python package ready: $pip_name"
  else
    echo "[+] Installing Python package: $pip_name"
    env LD_LIBRARY_PATH="" pip install "$pip_name"
  fi
}

hf_download() {
  local repo_id="$1"
  local target_dir="$2"
  mkdir -p "$target_dir"
  echo "[DOWNLOAD][HF] $repo_id -> $target_dir"
  python - "$repo_id" "$target_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target_dir = sys.argv[2]
snapshot_download(repo_id, local_dir=target_dir)
PY
}

modelscope_download() {
  local model_id="$1"
  local target_dir="$2"
  mkdir -p "$target_dir"
  echo "[DOWNLOAD][ModelScope] $model_id -> $target_dir"
  python - "$model_id" "$target_dir" <<'PY'
import sys
from modelscope import snapshot_download

model_id = sys.argv[1]
target_dir = sys.argv[2]
snapshot_download(model_id, local_dir=target_dir)
PY
}

download_with_preference() {
  local name="$1"
  local target_dir="$2"
  local modelscope_id="$3"
  local hf_id="$4"

  if [[ -n "$(find "$target_dir" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
    echo "[=] $name already exists, skipping: $target_dir"
    return
  fi

  case "$SOURCE" in
    modelscope)
      if [[ -n "$modelscope_id" ]]; then
        modelscope_download "$modelscope_id" "$target_dir"
      elif [[ -n "$hf_id" ]]; then
        echo "[!] No confirmed ModelScope mirror for $name, falling back to Hugging Face."
        hf_download "$hf_id" "$target_dir"
      fi
      ;;
    hf)
      if [[ -n "$hf_id" ]]; then
        hf_download "$hf_id" "$target_dir"
      elif [[ -n "$modelscope_id" ]]; then
        echo "[!] No confirmed Hugging Face path for $name, falling back to ModelScope."
        modelscope_download "$modelscope_id" "$target_dir"
      fi
      ;;
    auto)
      if [[ -n "$modelscope_id" ]]; then
        modelscope_download "$modelscope_id" "$target_dir"
      elif [[ -n "$hf_id" ]]; then
        hf_download "$hf_id" "$target_dir"
      fi
      ;;
  esac
}

echo "============================================"
echo " 1. Cloning external repositories"
echo "============================================"

clone_if_missing "https://github.com/facebookresearch/sam3.git" "$REPO_DIR/sam3"
clone_if_missing "https://github.com/lixiaowen-xw/DiffuEraser.git" "$REPO_DIR/DiffuEraser"
clone_if_missing "https://github.com/Kunbyte-AI/ROSE.git" "$REPO_DIR/ROSE"
clone_if_missing "https://github.com/sczhou/ProPainter.git" "$PART2_PROPainter_DIR"

echo
echo "============================================"
echo " 2. Installing helper download packages"
echo "============================================"

ensure_python_pkg "modelscope" "modelscope"
ensure_python_pkg "huggingface_hub" "huggingface_hub"

echo
echo "============================================"
echo " 3. Downloading model weights"
echo "============================================"
echo "Preferred source: $SOURCE"

download_with_preference "SAM 3" "$MODEL_ROOT/sam3" "facebook/sam3" "facebook/sam3"
download_with_preference "SAM 3.1" "$MODEL_ROOT/sam3.1" "facebook/sam3.1" "facebook/sam3.1"
download_with_preference "DiffuEraser" "$MODEL_ROOT/diffuEraser" "xingzi/diffuEraser" "lixiaowen/diffuEraser"
download_with_preference "sd-vae-ft-mse" "$MODEL_ROOT/sd-vae-ft-mse" "" "stabilityai/sd-vae-ft-mse"
download_with_preference "stable-diffusion-v1-5" "$MODEL_ROOT/stable-diffusion-v1-5" "AI-ModelScope/stable-diffusion-v1-5" "stable-diffusion-v1-5/stable-diffusion-v1-5"
download_with_preference "Wan2.1-Fun-1.3B-InP" "$MODEL_ROOT/Wan2.1-Fun-1.3B-InP" "PAI/Wan2.1-Fun-1.3B-InP" "alibaba-pai/Wan2.1-Fun-1.3B-InP"
download_with_preference "ROSE transformer" "$MODEL_ROOT/ROSE_transformer" "" "Kunbyte/ROSE"

echo
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo "External repos:"
echo "  SAM 3       : $REPO_DIR/sam3"
echo "  DiffuEraser : $REPO_DIR/DiffuEraser"
echo "  ROSE        : $REPO_DIR/ROSE"
echo "  ProPainter  : $PART2_PROPainter_DIR"
echo "Model root:"
echo "  $MODEL_ROOT"
echo
echo "Notes:"
echo "  - Default behavior prefers ModelScope, but will fall back to Hugging Face"
echo "    for weights without a confirmed ModelScope mirror in this project."
echo "  - SAM 3 and SAM 3.1 can be downloaded from ModelScope in this setup,"
echo "    so Hugging Face access approval is not required if you stay on the"
echo "    default ModelScope path."
