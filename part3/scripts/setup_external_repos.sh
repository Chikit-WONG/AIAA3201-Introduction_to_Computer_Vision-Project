#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/external/repository"

mkdir -p "$REPO_DIR"
echo "Assuming external repositories are already cloned under: $REPO_DIR"
for name in sam3 DiffuEraser ROSE; do
  if [[ -d "$REPO_DIR/$name" ]]; then
    echo "[OK] Found $name"
  else
    echo "[WARN] Missing $name under $REPO_DIR"
  fi
done

