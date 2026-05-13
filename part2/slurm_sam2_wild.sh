#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o temp/sam2_wild_output.txt
#SBATCH -e temp/sam2_wild_err.txt
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -J p2_sam2_wild

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

python run.py \
    --method sam2 \
    --gpu 0 \
    --davis-root ../data/Wild_Video_DAVIS \
    --sequences ride1 ride2 ride3 run1 run2 run3 \
    --output results/wild_sam2

echo "Job ended at $(date)"
conda deactivate
