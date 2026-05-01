#!/bin/bash
#SBATCH -J part3_method
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o temp/part3_method_%j.out
#SBATCH -e temp/part3_method_%j.err

set -euo pipefail

ROOT="/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3"
cd "$ROOT"
mkdir -p temp
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate cv2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${CONFIG:?CONFIG is required}"
: "${METHOD:?METHOD is required}"
: "${SEQUENCE:?SEQUENCE is required}"
: "${INPUT_VIDEO:?INPUT_VIDEO is required}"

CMD=(
  python run_part3.py
  --config "$CONFIG"
  --method "$METHOD"
  --sequence "$SEQUENCE"
  --input "$INPUT_VIDEO"
)

if [[ -n "${PROMPT:-}" ]]; then
  CMD+=(--prompt "$PROMPT")
fi

if [[ "${ALLOW_EXISTING_MASKS:-0}" == "1" ]]; then
  CMD+=(--allow-existing-masks)
fi

if [[ -n "${EXISTING_MASK_DIR:-}" ]]; then
  CMD+=(--existing-mask-dir "$EXISTING_MASK_DIR")
fi

"${CMD[@]}"
