#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/vggt4d_all_output.txt
#SBATCH -e temp/vggt4d_all_err.txt
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

PART2_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$PART2_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
cd "$PART2_ROOT"

source "$CONDA_SH"
conda activate "$PART2_CONDA_ENV"
module load "$CUDA_MODULE_NAME"

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Process all 30 DAVIS 2017 validation sequences
# Note: if this exceeds 30-min debug limit, split into multiple jobs
python run.py --method vggt4d --gpu 0

echo "Job ended at $(date)"
conda deactivate
