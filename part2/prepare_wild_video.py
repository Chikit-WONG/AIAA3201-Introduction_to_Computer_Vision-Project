#!/usr/bin/env python3
"""Prepare Wild Video for Part 2 pipelines."""

import os
import sys
import glob
import argparse

import cv2
import numpy as np

TARGET_H = 480
YOLO_MODEL = "yolov8n-seg.pt"
PERSON_CLASS = 0
MAX_SCAN_FRAMES = 9999


def extract_frames(video_path: str, out_dir: str, target_h: int = TARGET_H):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return 0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w = int(w0 * target_h / h0)
    target_w = target_w if target_w % 2 == 0 else target_w + 1
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if h0 != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(out_dir, f"{idx:05d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    print(f"  Extracted {idx} frames ({target_w}x{target_h}) -> {out_dir}")
    return idx


def generate_annotation(frames_dir: str, ann_dir: str, model):
    os.makedirs(ann_dir, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) + glob.glob(os.path.join(frames_dir, "*.png")))
    if not frames:
        print(f"  [WARN] No frames found in {frames_dir}")
        return
    ann_mask = None
    ann_frame_idx = None
    for i, frame_path in enumerate(frames[:MAX_SCAN_FRAMES]):
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
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                candidate_mask[mask_resized > 0.5] = obj_id
                obj_id += 1
        if obj_id > 1:
            ann_mask = candidate_mask
            ann_frame_idx = i
            break
    if ann_mask is None:
        sample = cv2.imread(frames[0])
        h, w = sample.shape[:2]
        ann_mask = np.zeros((h, w), dtype=np.uint8)
        ann_frame_idx = 0
        print(f"  [WARN] No persons found in first {min(len(frames), MAX_SCAN_FRAMES)} frames! SAM2 may skip this sequence.")
    ann_fname = f"{ann_frame_idx:05d}.png"
    ann_path = os.path.join(ann_dir, ann_fname)
    cv2.imwrite(ann_path, ann_mask)
    n_persons = int(ann_mask.max())
    print(f"  Annotation: {n_persons} person(s) detected at frame {ann_frame_idx} -> {ann_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Wild Video for Part 2")
    parser.add_argument("--video-dir", default=None, help="Directory containing Wild Video mp4 files")
    parser.add_argument("--output-root", default=None, help="Output root for DAVIS-like structure")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    video_dir = args.video_dir or os.path.join(project_root, "data", "Wild_Video", "input_with_person")
    output_root = args.output_root or os.path.join(project_root, "data", "Wild_Video_DAVIS")
    jpeg_root = os.path.join(output_root, "JPEGImages", "480p")
    ann_root = os.path.join(output_root, "Annotations", "480p")
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        print(f"[ERROR] No mp4 files found in {video_dir}")
        sys.exit(1)
    print(f"Found {len(videos)} video(s): {[os.path.basename(v) for v in videos]}")
    from ultralytics import YOLO
    print("\nLoading YOLOv8-seg model for auto-annotation...")
    model = YOLO(YOLO_MODEL)
    for video_path in videos:
        seq_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{seq_name}] {os.path.basename(video_path)}")
        frames_dir = os.path.join(jpeg_root, seq_name)
        ann_dir = os.path.join(ann_root, seq_name)
        if os.path.isdir(frames_dir) and len(glob.glob(os.path.join(frames_dir, "*.jpg"))) > 0:
            n = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
            print(f"  Frames already extracted ({n} frames), skipping.")
        else:
            extract_frames(video_path, frames_dir)
        existing_anns = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
        has_valid_ann = False
        for p in existing_anns:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.max() > 0:
                has_valid_ann = True
                break
        if has_valid_ann:
            print(f"  Annotation already exists ({os.path.basename(existing_anns[0])}), skipping.")
        else:
            for p in existing_anns:
                os.remove(p)
            generate_annotation(frames_dir, ann_dir, model)
    print(f"\nDone! Wild Video DAVIS structure ready at: {output_root}")


if __name__ == "__main__":
    main()
