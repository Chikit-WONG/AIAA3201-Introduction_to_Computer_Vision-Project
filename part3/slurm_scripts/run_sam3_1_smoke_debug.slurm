#!/bin/bash
#SBATCH -J sam31_smoke
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o temp/sam31_smoke_%j.out
#SBATCH -e temp/sam31_smoke_%j.err

set -euo pipefail

ROOT="/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3"
cd "$ROOT"
mkdir -p temp
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate cv2

SHORT_VIDEO="$ROOT/outputs/smoke_inputs/Wild_Video1_short33.mp4"
if [[ ! -f "$SHORT_VIDEO" ]]; then
  python - <<'PY'
import os
import cv2
video_path = "../data/Wild_Video/input_with_person/Wild_Video1.mp4"
short_video = "outputs/smoke_inputs/Wild_Video1_short33.mp4"
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
  --config configs/sam3_1_ablation.yaml \
  --method sam3_propainter \
  --sequence Wild_Video1 \
  --input "$SHORT_VIDEO" \
  --prompt "person"
