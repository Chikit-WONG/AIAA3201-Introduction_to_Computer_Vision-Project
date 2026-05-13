#!/bin/bash
# Debug-only wrapper for short sequence-matrix checks.
# Use the non-debug script for longer full-DAVIS runs.
#SBATCH -J p3_seq_matrix_dbg
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o artifacts_debug/slurm_job_logs/p3_seq_matrix_debug_%j.out
#SBATCH -e artifacts_debug/slurm_job_logs/p3_seq_matrix_debug_%j.err

set -euo pipefail

PART3_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
ROOT="$PART3_ROOT"
cd "$ROOT"
mkdir -p artifacts_debug/slurm_job_logs
source "$CONDA_SH"
conda activate "$PART3_CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF_DEFAULT}"

: "${CONFIG:?CONFIG is required}"
: "${SEQUENCE:?SEQUENCE is required}"
: "${INPUT_VIDEO:?INPUT_VIDEO is required}"

slugify() {
  python - "$1" <<'PY'
import sys
from src.io_utils import slugify
print(slugify(sys.argv[1]))
PY
}

method_output_slug() {
  python - "$CONFIG" "$1" <<'PY'
import sys
import yaml
from src.io_utils import part3_method_output_slug

config_path = sys.argv[1]
method = sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(part3_method_output_slug(method, cfg.get("sam3", {}).get("version")))
PY
}

OUTPUT_ROOT="$(python - "$CONFIG" <<'PY'
import os
import sys
import yaml
config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
root = cfg.get("output_root", "outputs")
print(os.path.abspath(root))
PY
)"

SEQ_SLUG="$(slugify "$SEQUENCE")"
BASE_METHOD="sam3_propainter"
BASE_METHOD_SLUG="$(method_output_slug "$BASE_METHOD")"
BASE_MASK_DIR="$OUTPUT_ROOT/masks/$SEQ_SLUG/$BASE_METHOD_SLUG/object_mask"

cleanup_method_dirs() {
  local method_slug="$1"
  rm -rf \
    "$OUTPUT_ROOT/masks/$SEQ_SLUG/$method_slug" \
    "$OUTPUT_ROOT/videos/$SEQ_SLUG/$method_slug" \
    "$OUTPUT_ROOT/logs/$SEQ_SLUG/$method_slug"
}

run_method() {
  local method="$1"
  shift
  local method_slug
  method_slug="$(method_output_slug "$method")"
  cleanup_method_dirs "$method_slug"

  CMD=(
    python run_part3.py
    --config "$CONFIG"
    --method "$method"
    --sequence "$SEQUENCE"
    --input "$INPUT_VIDEO"
  )

  if [[ -n "${PROMPT:-}" ]]; then
    CMD+=(--prompt "$PROMPT")
  fi

  if [[ -n "${INIT_MASK:-}" ]]; then
    CMD+=(--init-mask "$INIT_MASK")
  fi

  if [[ $# -gt 0 ]]; then
    CMD+=("$@")
  fi

  echo "+"
  printf ' %q' "${CMD[@]}"
  echo
  "${CMD[@]}"
}

run_method "$BASE_METHOD"

for method in \
  sam3_diffueraser_object \
  sam3_diffueraser_side_effect \
  sam3_rose_object \
  sam3_rose_side_effect
do
  run_method "$method" --allow-existing-masks --existing-mask-dir "$BASE_MASK_DIR"
done
