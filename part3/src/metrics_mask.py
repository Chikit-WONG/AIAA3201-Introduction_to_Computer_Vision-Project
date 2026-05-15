"""Mask metrics for DAVIS."""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np


def compute_iou(pred_mask, gt_mask) -> float:
    pred_bin = pred_mask > 127
    gt_bin = gt_mask > 127
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def evaluate_mask_sequence(pred_mask_dir, gt_mask_dir, threshold: float = 0.5) -> dict:
    gt_paths = sorted(glob.glob(os.path.join(gt_mask_dir, "*.png")))
    if not gt_paths:
        return {"JM": 0.0, "JR": 0.0, "num_frames": 0, "threshold": threshold}

    ious = []
    for gt_path in gt_paths:
        name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_mask_dir, name)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(pred_path) else None
        if pred_mask is None:
            pred_mask = np.zeros_like(gt_mask)
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(compute_iou(pred_mask, gt_mask))

    if not ious:
        return {"JM": 0.0, "JR": 0.0, "num_frames": 0, "threshold": threshold}
    return {
        "JM": float(np.mean(ious)),
        "JR": float(np.mean(np.array(ious) >= threshold)),
        "num_frames": len(ious),
        "threshold": threshold,
    }


def compute_jm(pred_mask_dir, gt_mask_dir) -> float:
    return evaluate_mask_sequence(pred_mask_dir, gt_mask_dir)["JM"]


def compute_jr(pred_mask_dir, gt_mask_dir, threshold: float = 0.5) -> float:
    return evaluate_mask_sequence(pred_mask_dir, gt_mask_dir, threshold=threshold)["JR"]

