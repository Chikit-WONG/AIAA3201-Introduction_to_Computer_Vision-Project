#!/bin/bash
#SBATCH -J part3_method
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o artifacts_debug/slurm_job_logs/part3_method_%j.out
#SBATCH -e artifacts_debug/slurm_job_logs/part3_method_%j.err

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
: "${METHOD:?METHOD is required}"
: "${SEQUENCE:?SEQUENCE is required}"
: "${INPUT_VIDEO:?INPUT_VIDEO is required}"

if [[ "${CLEAN_OUTPUT:-1}" == "1" ]]; then
  python - "$CONFIG" "$SEQUENCE" "$METHOD" <<'PY'
import shutil
import sys
from pathlib import Path
import yaml
from src.io_utils import part3_method_output_slug

def slugify(text: str) -> str:
    lowered = text.strip().lower()
    chars = []
    for char in lowered:
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "default"

config_path = Path(sys.argv[1])
sequence = sys.argv[2]
method = sys.argv[3]
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
output_root = (Path.cwd() / config.get("output_root", "outputs")).resolve()
seq_slug = slugify(sequence)
method_slug = part3_method_output_slug(method, config.get("sam3", {}).get("version"))
for branch in ("masks", "videos", "logs"):
    target = output_root / branch / seq_slug / method_slug
    shutil.rmtree(target, ignore_errors=True)
PY
fi

CMD=(
  python run_part3.py
  --config "$CONFIG"
  --method "$METHOD"
  --sequence "$SEQUENCE"
  --input "$INPUT_VIDEO"
)

if [[ -n "${PROMPT:-}" ]]; then
  CMD+=(--prompt "$PROMPT")
fi

if [[ -n "${INIT_MASK:-}" ]]; then
  CMD+=(--init-mask "$INIT_MASK")
fi

if [[ "${ALLOW_EXISTING_MASKS:-0}" == "1" ]]; then
  CMD+=(--allow-existing-masks)
fi

if [[ -n "${EXISTING_MASK_DIR:-}" ]]; then
  CMD+=(--existing-mask-dir "$EXISTING_MASK_DIR")
fi

"${CMD[@]}"
