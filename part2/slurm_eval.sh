#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/eval_output.txt
#SBATCH -e temp/eval_err.txt
#SBATCH -n 1
#SBATCH --time=00:10:00

PART2_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$PART2_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
cd "$PART2_ROOT"

source "$CONDA_SH"
conda activate "$PART2_CONDA_ENV"

echo "Job started at $(date)"

# Compare Pipeline A (VGGT4D) vs Pipeline B (SAM2)
python evaluate.py \
    --pred results/vggt4d \
    --pred2 results/sam2 \
    --davis-root ../data/DAVIS \
    --name1 "VGGT4D+ProPainter" \
    --name2 "SAM2+ProPainter" \
    --save-json results/comparison_metrics.json

echo "Job ended at $(date)"
conda deactivate
