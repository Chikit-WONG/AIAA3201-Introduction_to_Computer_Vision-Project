"""Video metrics for paired wild evaluation."""

from __future__ import annotations

import cv2
import numpy as np

from .alignment import align_frame_ecc
from .video_utils import read_video_frames


def compute_psnr(pred_frame, gt_frame) -> float:
    mse = np.mean((pred_frame.astype(np.float32) - gt_frame.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return 100.0
    return float(min(20.0 * np.log10(255.0 / np.sqrt(mse)), 100.0))


def compute_ssim(pred_frame, gt_frame) -> float:
    pred = pred_frame.astype(np.float64)
    gt = gt_frame.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def _ssim_single_channel(x, y):
        kernel = (11, 11)
        sigma = 1.5
        mu_x = cv2.GaussianBlur(x, kernel, sigma)
        mu_y = cv2.GaussianBlur(y, kernel, sigma)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(x * x, kernel, sigma) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, kernel, sigma) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, kernel, sigma) - mu_xy

        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = numerator / (denominator + 1e-12)
        return float(ssim_map.mean())

    channels = [
        _ssim_single_channel(pred[:, :, idx], gt[:, :, idx])
        for idx in range(pred.shape[2])
    ]
    return float(np.mean(channels))


def evaluate_video_quality(pred_video, gt_video, align: bool = True) -> dict:
    pred_frames, _pred_fps = read_video_frames(pred_video)
    gt_frames, _gt_fps = read_video_frames(gt_video)
    count = min(len(pred_frames), len(gt_frames))
    if count == 0:
        return {"PSNR": 0.0, "SSIM": 0.0, "num_frames": 0, "aligned": align}

    psnrs = []
    ssims = []
    aligned_any = False
    for idx in range(count):
        pred = pred_frames[idx]
        gt = gt_frames[idx]
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        if align:
            pred, ok = align_frame_ecc(pred, gt)
            aligned_any = aligned_any or ok
        psnrs.append(compute_psnr(pred, gt))
        ssims.append(compute_ssim(pred, gt))
    return {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "num_frames": count,
        "aligned": aligned_any if align else False,
    }
