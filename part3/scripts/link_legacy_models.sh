#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/../local_paths.sh"
LEGACY_ROOT="$LEGACY_MODELS_ROOT"
TARGET_ROOT="$ROOT_DIR/models"

mkdir -p "$TARGET_ROOT"

link_dir() {
  local name="$1"
  local source="$LEGACY_ROOT/$name"
  local target="$TARGET_ROOT/$name"

  if [[ ! -e "$source" ]]; then
    echo "[WARN] Missing legacy model path: $source"
    return
  fi

  if [[ -L "$target" ]]; then
    if [[ "$(readlink -f "$target")" == "$(readlink -f "$source")" ]]; then
      echo "[=] Link already correct: $target"
      return
    fi
    rm -f "$target"
  elif [[ -d "$target" && -z "$(find "$target" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
    rmdir "$target"
  elif [[ -e "$target" ]]; then
    echo "[ERROR] Target exists and is not a replaceable empty dir/symlink: $target"
    exit 1
  fi

  ln -s "$source" "$target"
  echo "[+] Linked $target -> $source"
}

for name in \
  sam3 \
  sam3.1 \
  diffuEraser \
  PCM_Weights \
  sd-vae-ft-mse \
  stable-diffusion-v1-5 \
  Wan2.1-Fun-1.3B-InP \
  ROSE_transformer
do
  link_dir "$name"
done

echo
echo "Legacy Part 3 model links are ready under: $TARGET_ROOT"
