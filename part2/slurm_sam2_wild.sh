#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o temp/sam2_wild_output.txt
#SBATCH -e temp/sam2_wild_err.txt
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part2
#SBATCH --time=01:00:00
#SBATCH -J p2_sam2_wild

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/ckwong627/miniconda3/envs/cv2
module load cuda/12.6

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
