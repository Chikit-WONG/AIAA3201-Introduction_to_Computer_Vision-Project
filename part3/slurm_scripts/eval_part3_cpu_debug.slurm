#!/bin/bash
#SBATCH -J part3_eval
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -t 00:15:00
#SBATCH -o temp/part3_eval_%j.out
#SBATCH -e temp/part3_eval_%j.err

set -euo pipefail

ROOT="/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3"
cd "$ROOT"
mkdir -p temp

python evaluate_part3.py \
  --config configs/default.yaml \
  --method sam3_diffueraser_object \
  --evaluate-wild
