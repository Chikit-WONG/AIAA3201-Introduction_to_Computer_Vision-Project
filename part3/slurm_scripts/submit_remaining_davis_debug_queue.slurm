#!/bin/bash
# Lightweight CPU-side submitter that keeps feeding remaining DAVIS debug jobs.
#SBATCH -J p3_dbg_submitter
#SBATCH -p i64m512u
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 2-00:00:00
#SBATCH -o artifacts_debug/slurm_job_logs/p3_dbg_submitter_%j.out
#SBATCH -e artifacts_debug/slurm_job_logs/p3_dbg_submitter_%j.err

set -euo pipefail

PART3_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
cd "$PART3_ROOT"
mkdir -p artifacts_debug/slurm_job_logs

source "$CONDA_SH"
conda activate "$PART3_CONDA_ENV"

: "${SEQUENCE_LIST:?SEQUENCE_LIST is required}"
: "${CONFIG:?CONFIG is required}"

bash scripts/submit_remaining_davis_debug_queue.sh "$SEQUENCE_LIST" "$CONFIG"
