#!/usr/bin/env python3
"""
Prepare Wild Video for Part 2 Pipelines
========================================
1. Extract frames from mp4 files at 480p into DAVIS-like directory structure.
2. Auto-generate first-frame person annotations using YOLOv8-seg
   (so SAM 2 can use centroid+bbox prompts without manual GT masks).

Output structure:
    data/Wild_Video_DAVIS/
    ├── JPEGImages/480p/<seq>/00000.jpg ...
    └── Annotations/480p/<seq>/00000.png  (YOLOv8 person mask, pixel value = object_id)
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np


TARGET_H = 480  # Target height; width scaled to preserve aspect ratio
YOLO_MODEL = "yolov8n-seg.pt"
PERSON_CLASS = 0  # COCO class 0 = person


def extract_frames(video_path: str, out_dir: str, target_h: int = TARGET_H):
    """Extract frames from mp4, resize to target_h, save as JPEG."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return 0

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = int(w0 * target_h / h0)
    # Ensure target_w is even (required by some codecs)
    target_w = target_w if target_w % 2 == 0 else target_w + 1

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if h0 != target_h:
            frame = cv2.resize(frame, (target_w, target_h),
                               interpolation=cv2.INTER_AREA)
        out_path = os.path.join(out_dir, f"{idx:05d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1

    cap.release()
    print(f"  Extracted {idx} frames ({target_w}x{target_h}) -> {out_dir}")
    return idx


MAX_SCAN_FRAMES = 9999  # Scan all frames to find the first frame with a person


def generate_annotation(frames_dir: str, ann_dir: str, model):
    """Run YOLOv8-seg, scan frames until a person is found, save DAVIS-format PNG.

    The annotation is saved with the filename matching the frame where the person
    was first detected (e.g. 00005.png if found in frame 5). SAM2 reads this
    filename to determine which frame_idx to use as the prompt.
    """
    os.makedirs(ann_dir, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))
                    + glob.glob(os.path.join(frames_dir, "*.png")))
    if not frames:
        print(f"  [WARN] No frames found in {frames_dir}")
        return

    scan_frames = frames[:MAX_SCAN_FRAMES]
    ann_mask = None
    ann_frame_idx = None

    for i, frame_path in enumerate(scan_frames):
        img = cv2.imread(frame_path)
        h, w = img.shape[:2]
        candidate_mask = np.zeros((h, w), dtype=np.uint8)

        results = model(img, verbose=False)
        obj_id = 1
        for result in results:
            if result.masks is None:
                continue
            boxes = result.boxes
            masks = result.masks.data.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for cls, mask in zip(classes, masks):
                if cls != PERSON_CLASS:
                    continue
                mask_resized = cv2.resize(mask, (w, h),
                                          interpolation=cv2.INTER_NEAREST)
                candidate_mask[mask_resized > 0.5] = obj_id
                obj_id += 1

        if obj_id > 1:  # At least one person found
            ann_mask = candidate_mask
            ann_frame_idx = i
            break

    if ann_mask is None:
        # No person found in any scanned frame — save empty mask at frame 0
        sample = cv2.imread(frames[0])
        h, w = sample.shape[:2]
        ann_mask = np.zeros((h, w), dtype=np.uint8)
        ann_frame_idx = 0
        print(f"  [WARN] No persons found in first {len(scan_frames)} frames! SAM2 will skip this sequence.")

    ann_fname = f"{ann_frame_idx:05d}.png"
    ann_path = os.path.join(ann_dir, ann_fname)
    cv2.imwrite(ann_path, ann_mask)

    n_persons = int(ann_mask.max())
    print(f"  Annotation: {n_persons} person(s) detected at frame {ann_frame_idx} -> {ann_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Wild Video for Part 2")
    parser.add_argument("--video-dir", default=None,
                        help="Directory containing Wild Video mp4 files")
    parser.add_argument("--output-root", default=None,
                        help="Output root for DAVIS-like structure")
    args = parser.parse_args()

    # Resolve paths relative to this script's location (part2/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    video_dir = args.video_dir or os.path.join(
        project_root, "data", "Wild_Video")
    output_root = args.output_root or os.path.join(
        project_root, "data", "Wild_Video_DAVIS")

    jpeg_root = os.path.join(output_root, "JPEGImages", "480p")
    ann_root = os.path.join(output_root, "Annotations", "480p")

    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        print(f"[ERROR] No mp4 files found in {video_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s): {[os.path.basename(v) for v in videos]}")

    # Load YOLOv8-seg model
    print("\nLoading YOLOv8-seg model for auto-annotation...")
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)

    for video_path in videos:
        # Derive sequence name: "Wild_Video-ride1.mp4" -> "ride1"
        basename = os.path.splitext(os.path.basename(video_path))[0]
        seq_name = basename.replace("Wild_Video-", "")

        print(f"\n[{seq_name}] {os.path.basename(video_path)}")

        frames_dir = os.path.join(jpeg_root, seq_name)
        ann_dir = os.path.join(ann_root, seq_name)

        # Step 1: Extract frames
        if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
            n = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
            print(f"  Frames already extracted ({n} frames), skipping.")
        else:
            extract_frames(video_path, frames_dir)

        # Step 2: Generate annotation
        # Check if a non-empty annotation already exists
        existing_anns = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
        has_valid_ann = any(
            cv2.imread(p, cv2.IMREAD_GRAYSCALE).max() > 0
            for p in existing_anns
            if cv2.imread(p, cv2.IMREAD_GRAYSCALE) is not None
        )
        if has_valid_ann:
            ann_fname = os.path.basename(existing_anns[0])
            print(f"  Annotation already exists ({ann_fname}), skipping.")
        else:
            # Remove any empty/stale annotation files before regenerating
            for p in existing_anns:
                os.remove(p)
            generate_annotation(frames_dir, ann_dir, model)

    print(f"\nDone! Wild Video DAVIS structure ready at: {output_root}")
    print(f"Sequences: {[os.path.splitext(os.path.basename(v))[0].replace('Wild_Video-', '') for v in videos]}")


if __name__ == "__main__":
    main()
