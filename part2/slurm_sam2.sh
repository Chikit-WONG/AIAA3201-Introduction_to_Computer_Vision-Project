#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/sam2_output.txt
#SBATCH -e temp/sam2_err.txt
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

python run.py --method sam2 --gpu 0 --sequences bmx-trees tennis

echo "Job ended at $(date)"
conda deactivate
