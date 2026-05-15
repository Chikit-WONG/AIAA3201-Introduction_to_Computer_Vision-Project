#!/usr/bin/env python3
"""Convert DAVIS frame folders into mp4 files for Part 3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def list_sequences(jpeg_root: Path) -> list[str]:
    return sorted(path.name for path in jpeg_root.iterdir() if path.is_dir())


def write_sequence_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPEG frames found in {frame_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis-root", default="../data/DAVIS")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument("--output-dir", default="../inputs/davis_videos")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sequences", nargs="+", default=None)
    args = parser.parse_args()

    part3_dir = Path(__file__).resolve().parents[1]
    davis_root = (part3_dir / args.davis_root).resolve()
    jpeg_root = davis_root / "JPEGImages" / args.resolution
    output_dir = (part3_dir / args.output_dir).resolve()

    sequences = args.sequences or list_sequences(jpeg_root)
    print(f"Preparing {len(sequences)} DAVIS videos into {output_dir}")
    for index, sequence in enumerate(sequences, start=1):
        frame_dir = jpeg_root / sequence
        output_path = output_dir / f"{sequence}.mp4"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(sequences)}] Skip existing: {output_path.name}")
            continue
        print(f"[{index}/{len(sequences)}] Writing {output_path.name}")
        write_sequence_video(frame_dir, output_path, args.fps)


if __name__ == "__main__":
    main()
