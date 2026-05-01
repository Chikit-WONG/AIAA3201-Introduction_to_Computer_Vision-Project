#!/bin/bash
#SBATCH -J part3_eval
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o temp/part3_eval_%j.out
#SBATCH -e temp/part3_eval_%j.err

set -euo pipefail

ROOT="/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3"
cd "$ROOT"
mkdir -p temp
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate cv2

: "${CONFIG:?CONFIG is required}"
: "${METHOD:?METHOD is required}"

CMD=(
  python evaluate_part3.py
  --config "$CONFIG"
  --method "$METHOD"
)

if [[ "${EVALUATE_DAVIS:-0}" == "1" ]]; then
  CMD+=(--evaluate-davis)
fi

if [[ "${EVALUATE_WILD:-0}" == "1" ]]; then
  CMD+=(--evaluate-wild)
fi

if [[ "${NO_ALIGN:-0}" == "1" ]]; then
  CMD+=(--no-align)
fi

if [[ -n "${METRICS_TAG:-}" ]]; then
  CMD+=(--metrics-tag "$METRICS_TAG")
fi

if [[ -n "${SEQUENCES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SEQS=(${SEQUENCES})
  CMD+=(--sequence "${EXTRA_SEQS[@]}")
fi

"${CMD[@]}"
