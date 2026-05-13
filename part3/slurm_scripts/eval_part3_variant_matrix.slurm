#!/bin/bash
#SBATCH -J p3_eval_matrix
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:25:00
#SBATCH -o artifacts_debug/slurm_job_logs/p3_eval_matrix_%j.out
#SBATCH -e artifacts_debug/slurm_job_logs/p3_eval_matrix_%j.err

set -euo pipefail

PART3_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
ROOT="$PART3_ROOT"
cd "$ROOT"
mkdir -p artifacts_debug/slurm_job_logs
source "$CONDA_SH"
conda activate "$PART3_CONDA_ENV"

: "${CONFIG:?CONFIG is required}"

METHODS=(
  sam3_propainter
  sam3_diffueraser_object
  sam3_diffueraser_side_effect
  sam3_rose_object
  sam3_rose_side_effect
)

for method in "${METHODS[@]}"; do
  python evaluate_part3.py \
    --config "$CONFIG" \
    --method "$method" \
    --evaluate-davis \
    --metrics-tag "$method" \
    --sequence bmx-trees tennis

  python evaluate_part3.py \
    --config "$CONFIG" \
    --method "$method" \
    --evaluate-wild \
    --metrics-tag "$method" \
    --sequence Wild_Video1 Wild_Video2
done
