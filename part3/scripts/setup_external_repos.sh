#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/external/repository"
PART2_PROPainter_DIR="$ROOT_DIR/../part2/external/ProPainter"

mkdir -p "$REPO_DIR"

clone_if_missing() {
  local url="$1"
  local target_dir="$2"
  if [[ -d "$target_dir/.git" ]]; then
    echo "[OK] Found existing repo: $target_dir"
  else
    echo "[CLONE] $url -> $target_dir"
    git clone "$url" "$target_dir"
  fi
}

clone_if_missing "https://github.com/facebookresearch/sam3.git" "$REPO_DIR/sam3"
clone_if_missing "https://github.com/lixiaowen-xw/DiffuEraser.git" "$REPO_DIR/DiffuEraser"
clone_if_missing "https://github.com/Kunbyte-AI/ROSE.git" "$REPO_DIR/ROSE"

mkdir -p "$(dirname "$PART2_PROPainter_DIR")"
clone_if_missing "https://github.com/sczhou/ProPainter.git" "$PART2_PROPainter_DIR"

echo
echo "External repositories are ready."
echo "  SAM 3        : $REPO_DIR/sam3"
echo "  DiffuEraser  : $REPO_DIR/DiffuEraser"
echo "  ROSE         : $REPO_DIR/ROSE"
echo "  ProPainter   : $PART2_PROPainter_DIR"
