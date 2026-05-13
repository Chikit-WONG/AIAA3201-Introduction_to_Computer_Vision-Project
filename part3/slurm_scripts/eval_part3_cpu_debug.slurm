#!/bin/bash
#SBATCH -J part3_eval
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 00:15:00
#SBATCH -o temp/part3_eval_%j.out
#SBATCH -e temp/part3_eval_%j.err

set -euo pipefail

PART3_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
ROOT="$PART3_ROOT"
cd "$ROOT"
mkdir -p temp

python evaluate_part3.py \
  --config configs/default.yaml \
  --method sam3_diffueraser_object \
  --evaluate-wild
