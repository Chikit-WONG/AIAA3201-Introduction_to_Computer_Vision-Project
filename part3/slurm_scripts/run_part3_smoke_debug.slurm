#!/bin/bash
#SBATCH -J part3_smoke
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH -o temp/part3_smoke_%j.out
#SBATCH -e temp/part3_smoke_%j.err

set -euo pipefail

ROOT="/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3"
cd "$ROOT"
mkdir -p temp
source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate cv2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SHORT_VIDEO="$ROOT/outputs/smoke_inputs/Wild_Video1_short33.mp4"
MASK_DIR="$ROOT/outputs/smoke_masks/Wild_Video1_short33_blank"
mkdir -p "$(dirname "$SHORT_VIDEO")" "$MASK_DIR"

python - <<'PY'
import os
import cv2
import numpy as np

video_path = "../data/Wild_Video/input_with_person/Wild_Video1.mp4"
short_video = "outputs/smoke_inputs/Wild_Video1_short33.mp4"
mask_dir = "outputs/smoke_masks/Wild_Video1_short33_blank"
os.makedirs(mask_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None
count = 0
max_frames = 33
while True:
    ok, frame = cap.read()
    if not ok or count >= max_frames:
        break
    h, w = frame.shape[:2]
    if writer is None:
        writer = cv2.VideoWriter(short_video, fourcc, fps, (w, h))
    writer.write(frame)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.imwrite(os.path.join(mask_dir, f"{count:05d}.png"), mask)
    count += 1
cap.release()
if writer is not None:
    writer.release()
print(f"Prepared short smoke video with {count} frames: {short_video}")
print(f"Prepared {count} blank masks in {mask_dir}")
PY

python run_part3.py \
  --config configs/default.yaml \
  --method part2_sam2_propainter \
  --sequence Wild_Video1 \
  --input "$SHORT_VIDEO" \
  --prompt "person" \
  --allow-existing-masks \
  --existing-mask-dir "$MASK_DIR"
