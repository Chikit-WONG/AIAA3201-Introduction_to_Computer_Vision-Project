"""Video and frame helpers."""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from .io_utils import ensure_dir


def list_frames(frames_dir: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(frames_dir, "*.jpg"))
        + glob.glob(os.path.join(frames_dir, "*.png"))
    )


def extract_video_frames(video_path: str, output_dir: str, ext: str = ".jpg") -> list[str]:
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    written = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = os.path.join(output_dir, f"{idx:05d}{ext}")
        cv2.imwrite(out_path, frame)
        written.append(out_path)
        idx += 1
    cap.release()
    return written


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def read_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(frames: list[np.ndarray], output_path: str, fps: float) -> str:
    if not frames:
        raise ValueError("No frames provided for video writing.")
    ensure_dir(Path(output_path).parent)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, max(float(fps), 1.0), (width, height))
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    return output_path


def frames_to_video(frames_dir: str, output_path: str, fps: float) -> str:
    paths = list_frames(frames_dir)
    if not paths:
        raise ValueError(f"No frames found in {frames_dir}")
    frames = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Could not read frame: {path}")
        frames.append(frame)
    return write_video(frames, output_path, fps)


def copy_video(src: str, dst: str) -> str:
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)
    return dst


def pad_video_to_16n_plus_1(video_path: str, output_path: str) -> tuple[str, int]:
    frames, fps = read_video_frames(video_path)
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    n = len(frames)
    target = n if (n - 1) % 16 == 0 else ((n - 1) // 16 + 1) * 16 + 1
    while len(frames) < target:
        frames.append(frames[-1].copy())
    write_video(frames, output_path, fps)
    return output_path, target


def resize_video(video_path: str, output_path: str, size_hw: tuple[int, int]) -> str:
    frames, fps = read_video_frames(video_path)
    height, width = size_hw
    resized = [cv2.resize(frame, (width, height)) for frame in frames]
    return write_video(resized, output_path, fps)

