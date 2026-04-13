#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/vggt4d_output.txt
#SBATCH -e temp/vggt4d_err.txt
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part2
#SBATCH --time=00:30:00

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/ckwong627/miniconda3/envs/cv2
module load cuda/12.6

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python run.py --method vggt4d --gpu 0 --sequences bmx-trees tennis

echo "Job ended at $(date)"
conda deactivate
