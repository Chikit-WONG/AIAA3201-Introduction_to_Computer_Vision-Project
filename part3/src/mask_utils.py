"""Mask helpers."""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from .io_utils import ensure_dir


def load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.uint8) * 255


def save_mask(path: str, mask: np.ndarray) -> str:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, binarize_mask(mask))
    return path


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8) * 255)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binarize_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    return cv2.dilate(binarize_mask(mask), kernel, iterations=1)


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binarize_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    return cv2.morphologyEx(binarize_mask(mask), cv2.MORPH_CLOSE, kernel)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = binarize_mask(mask)
    flood = binary.copy()
    h, w = binary.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, inv)


def shift_mask(mask: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    height, width = mask.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        binarize_mask(mask),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted


def polygon_to_mask(shape_hw: tuple[int, int], polygon: list[list[int]]) -> np.ndarray:
    height, width = shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def flip_mask_across_vertical_axis(mask: np.ndarray, x_center: int) -> np.ndarray:
    flipped = cv2.flip(mask, 1)
    width = mask.shape[1]
    current_center = width // 2
    dx = int(2 * (x_center - current_center))
    return shift_mask(flipped, dx=dx, dy=0)


def mask_dir_to_video(mask_dir: str, output_path: str, fps: float) -> str:
    from .video_utils import write_video

    paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not paths:
        raise ValueError(f"No masks found in {mask_dir}")
    frames = []
    for path in paths:
        mask = load_mask(path)
        frames.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    return write_video(frames, output_path, fps)


def union_masks(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("union_masks requires at least one mask")
    out = np.zeros_like(binarize_mask(masks[0]))
    for mask in masks:
        out = cv2.bitwise_or(out, binarize_mask(mask))
    return out

