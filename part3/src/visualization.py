"""Visualization helpers."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .io_utils import ensure_dir
from .mask_utils import load_mask
from .video_utils import list_frames, write_video


def create_mask_debug_video(frames_dir: str, object_dir: str, shadow_dir: str, final_dir: str, output_path: str, fps: float = 10.0) -> str:
    frame_paths = list_frames(frames_dir)
    debug_frames = []
    for frame_path in frame_paths:
        name = os.path.splitext(os.path.basename(frame_path))[0] + ".png"
        frame = cv2.imread(frame_path)
        object_mask = load_mask(os.path.join(object_dir, name))
        shadow_mask = load_mask(os.path.join(shadow_dir, name))
        final_mask = load_mask(os.path.join(final_dir, name))
        object_vis = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
        shadow_vis = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
        overlay = frame.copy()
        overlay[final_mask > 0] = (0.5 * overlay[final_mask > 0] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        grid = np.concatenate([frame, object_vis, shadow_vis, overlay], axis=1)
        debug_frames.append(grid)
    ensure_dir(os.path.dirname(output_path))
    return write_video(debug_frames, output_path, fps)

