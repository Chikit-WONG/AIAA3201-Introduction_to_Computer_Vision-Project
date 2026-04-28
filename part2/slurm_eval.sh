#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/eval_output.txt
#SBATCH -e temp/eval_err.txt
#SBATCH -n 1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part2
#SBATCH --time=00:10:00

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/ckwong627/miniconda3/envs/cv2

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
