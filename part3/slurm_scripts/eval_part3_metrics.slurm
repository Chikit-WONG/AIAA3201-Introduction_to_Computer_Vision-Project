#!/bin/bash
#SBATCH -J part3_eval
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o temp/part3_eval_%j.out
#SBATCH -e temp/part3_eval_%j.err

set -euo pipefail

PART3_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
ROOT="$PART3_ROOT"
cd "$ROOT"
mkdir -p temp
source "$CONDA_SH"
conda activate "$PART3_CONDA_ENV"

: "${CONFIG:?CONFIG is required}"
: "${METHOD:?METHOD is required}"

CMD=(
  python evaluate_part3.py
  --config "$CONFIG"
  --method "$METHOD"
)

if [[ "${EVALUATE_DAVIS:-0}" == "1" ]]; then
  CMD+=(--evaluate-davis)
fi

if [[ "${EVALUATE_WILD:-0}" == "1" ]]; then
  CMD+=(--evaluate-wild)
fi

if [[ "${NO_ALIGN:-0}" == "1" ]]; then
  CMD+=(--no-align)
fi

if [[ -n "${METRICS_TAG:-}" ]]; then
  CMD+=(--metrics-tag "$METRICS_TAG")
fi

if [[ -n "${SEQUENCES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SEQS=(${SEQUENCES})
  CMD+=(--sequence "${EXTRA_SEQS[@]}")
fi

"${CMD[@]}"
