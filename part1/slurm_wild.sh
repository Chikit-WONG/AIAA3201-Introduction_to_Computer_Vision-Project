#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -o temp/wild_output.txt
#SBATCH -e temp/wild_err.txt
#SBATCH -n 1
#SBATCH --time=02:00:00
#SBATCH -J p1_wild

PART1_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$PART1_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
cd "$PART1_ROOT"

source "$CONDA_SH"
conda activate "$PART1_CONDA_ENV"
# CPU-only for Part 1 (traditional CV methods)

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python run.py --davis --config configs/wild.yaml --output results/wild_video

echo ""
echo "Job ended at $(date)"
conda deactivate
