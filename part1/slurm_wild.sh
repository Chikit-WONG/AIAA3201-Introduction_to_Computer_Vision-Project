#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -o temp/wild_output.txt
#SBATCH -e temp/wild_err.txt
#SBATCH -n 1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part1
#SBATCH --time=02:00:00
#SBATCH -J p1_wild

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/ckwong627/miniconda3/envs/cv
# CPU-only for Part 1 (traditional CV methods)

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python run.py --davis --config configs/wild.yaml --output results/wild_video

echo ""
echo "Job ended at $(date)"
conda deactivate
