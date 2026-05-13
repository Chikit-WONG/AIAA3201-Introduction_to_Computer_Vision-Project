#!/bin/bash
#SBATCH -J sam3_smoke
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o temp/sam3_smoke_%j.out
#SBATCH -e temp/sam3_smoke_%j.err

set -euo pipefail

PART3_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
ROOT="$PART3_ROOT"
cd "$ROOT"
mkdir -p temp
source "$CONDA_SH"
conda activate "$PART3_CONDA_ENV"

SHORT_VIDEO="$ROOT/results_debug/smoke_inputs/Wild_Video1_short33.mp4"
if [[ ! -f "$SHORT_VIDEO" ]]; then
  python - <<'PY'
import os
import cv2
video_path = "../data/Wild_Video/input_with_person/Wild_Video1.mp4"
short_video = "results_debug/smoke_inputs/Wild_Video1_short33.mp4"
os.makedirs(os.path.dirname(short_video), exist_ok=True)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None
count = 0
while True:
    ok, frame = cap.read()
    if not ok or count >= 33:
        break
    h, w = frame.shape[:2]
    if writer is None:
        writer = cv2.VideoWriter(short_video, fourcc, fps, (w, h))
    writer.write(frame)
    count += 1
cap.release()
if writer is not None:
    writer.release()
print(short_video)
PY
fi

python run_part3.py \
  --config configs/sam3_ablation.yaml \
  --method sam3_propainter \
  --sequence Wild_Video1 \
  --input "$SHORT_VIDEO" \
  --prompt "person"
