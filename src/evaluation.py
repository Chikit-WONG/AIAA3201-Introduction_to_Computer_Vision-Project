"""
Evaluation Module
=================
Metrics: IoU mean (JM), IoU recall (JR), PSNR, SSIM.
Supports per-sequence and aggregate evaluation on DAVIS dataset.
"""

import os
import glob
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ======================================================================
# Mask Quality Metrics (JM & JR)
# ======================================================================

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks (0/255 uint8)."""
    pred_bin = pred > 127
    gt_bin = gt > 127
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def compute_jm_jr(
    pred_masks_dir: str,
    gt_masks_dir: str,
    iou_threshold: float = 0.5,
) -> dict:
    """Compute IoU mean (JM) and IoU recall (JR) over a sequence.

    Args:
        pred_masks_dir: directory containing predicted mask PNGs.
        gt_masks_dir:   directory containing GT mask PNGs (DAVIS Annotations).
        iou_threshold:  threshold for JR (default 0.5).

    Returns:
        dict with keys: "JM", "JR", "per_frame_iou", "num_frames".
    """
    gt_files = sorted(glob.glob(os.path.join(gt_masks_dir, "*.png")))
    if not gt_files:
        return {"JM": 0.0, "JR": 0.0, "per_frame_iou": [], "num_frames": 0}

    ious = []
    for gt_path in gt_files:
        fname = os.path.basename(gt_path)
        pred_path = os.path.join(pred_masks_dir, fname)

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        # GT mask from DAVIS: multi-object with different IDs.
        # Convert to binary: any object > 0 is foreground.
        gt_binary = (gt_mask > 0).astype(np.uint8) * 255

        if os.path.exists(pred_path):
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred_mask is None:
                pred_mask = np.zeros_like(gt_binary)
            # Ensure same size
            if pred_mask.shape != gt_binary.shape:
                pred_mask = cv2.resize(
                    pred_mask, (gt_binary.shape[1], gt_binary.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            pred_mask = np.zeros_like(gt_binary)

        iou = compute_iou(pred_mask, gt_binary)
        ious.append(iou)

    if not ious:
        return {"JM": 0.0, "JR": 0.0, "per_frame_iou": [], "num_frames": 0}

    jm = float(np.mean(ious))
    jr = float(np.sum(np.array(ious) >= iou_threshold)) / len(ious)
    return {
        "JM": jm,
        "JR": jr,
        "per_frame_iou": ious,
        "num_frames": len(ious),
    }


# ======================================================================
# Video Quality Metrics (PSNR & SSIM)
# ======================================================================

def compute_psnr_frame(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR between two BGR frames. Returns 100.0 cap for identical frames."""
    val = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    return min(val, 100.0) if np.isfinite(val) else 100.0


def compute_ssim_frame(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM between two BGR frames."""
    return float(structural_similarity(
        gt, pred, data_range=255, channel_axis=2,
    ))


def compute_video_quality(
    pred_frames_dir: str,
    gt_frames_dir: str,
) -> dict:
    """Compute average PSNR and SSIM over a sequence.

    Args:
        pred_frames_dir: directory with inpainted frame PNGs/JPGs.
        gt_frames_dir:   directory with original frame JPGs (DAVIS JPEGImages).

    Returns:
        dict with "PSNR", "SSIM", "num_frames".
    """
    gt_files = sorted(
        glob.glob(os.path.join(gt_frames_dir, "*.jpg"))
        + glob.glob(os.path.join(gt_frames_dir, "*.png"))
    )
    if not gt_files:
        return {"PSNR": 0.0, "SSIM": 0.0, "num_frames": 0}

    psnrs, ssims = [], []
    for gt_path in gt_files:
        fname = os.path.splitext(os.path.basename(gt_path))[0]
        # Try multiple extensions for predicted frames
        pred_path = None
        for ext in (".png", ".jpg"):
            candidate = os.path.join(pred_frames_dir, fname + ext)
            if os.path.exists(candidate):
                pred_path = candidate
                break
        if pred_path is None:
            continue

        gt_frame = cv2.imread(gt_path)
        pred_frame = cv2.imread(pred_path)
        if gt_frame is None or pred_frame is None:
            continue
        if pred_frame.shape != gt_frame.shape:
            pred_frame = cv2.resize(
                pred_frame, (gt_frame.shape[1], gt_frame.shape[0]),
            )

        psnrs.append(compute_psnr_frame(pred_frame, gt_frame))
        ssims.append(compute_ssim_frame(pred_frame, gt_frame))

    if not psnrs:
        return {"PSNR": 0.0, "SSIM": 0.0, "num_frames": 0}

    return {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "num_frames": len(psnrs),
    }


# ======================================================================
# Aggregate evaluation across sequences
# ======================================================================

def evaluate_dataset(
    pred_root: str,
    davis_root: str,
    resolution: str = "480p",
    sequences: list = None,
) -> dict:
    """Evaluate all sequences under pred_root against DAVIS ground truth.

    Expected directory layout under pred_root:
        <pred_root>/<seq_name>/masks/      -> predicted masks
        <pred_root>/<seq_name>/frames/     -> inpainted frames

    Args:
        pred_root:  root directory of pipeline outputs.
        davis_root: root directory of DAVIS dataset.
        resolution: "480p" or "Full-Resolution".
        sequences:  list of sequence names to evaluate (None = all found).

    Returns:
        dict mapping sequence names to metric dicts, plus an "average" key.
    """
    gt_images_root = os.path.join(davis_root, "JPEGImages", resolution)
    gt_annot_root = os.path.join(davis_root, "Annotations", resolution)

    if sequences is None:
        # Discover sequences from pred_root
        sequences = sorted([
            d for d in os.listdir(pred_root)
            if os.path.isdir(os.path.join(pred_root, d))
        ])

    results = {}
    all_jm, all_jr, all_psnr, all_ssim = [], [], [], []

    for seq in sequences:
        pred_masks_dir = os.path.join(pred_root, seq, "masks")
        pred_frames_dir = os.path.join(pred_root, seq, "frames")
        gt_masks_dir = os.path.join(gt_annot_root, seq)
        gt_frames_dir = os.path.join(gt_images_root, seq)

        seq_result = {}

        # Mask quality
        if os.path.isdir(pred_masks_dir) and os.path.isdir(gt_masks_dir):
            mask_metrics = compute_jm_jr(pred_masks_dir, gt_masks_dir)
            seq_result["JM"] = mask_metrics["JM"]
            seq_result["JR"] = mask_metrics["JR"]
            all_jm.append(mask_metrics["JM"])
            all_jr.append(mask_metrics["JR"])
        else:
            seq_result["JM"] = None
            seq_result["JR"] = None

        # Video quality
        if os.path.isdir(pred_frames_dir) and os.path.isdir(gt_frames_dir):
            vid_metrics = compute_video_quality(pred_frames_dir, gt_frames_dir)
            seq_result["PSNR"] = vid_metrics["PSNR"]
            seq_result["SSIM"] = vid_metrics["SSIM"]
            all_psnr.append(vid_metrics["PSNR"])
            all_ssim.append(vid_metrics["SSIM"])
        else:
            seq_result["PSNR"] = None
            seq_result["SSIM"] = None

        results[seq] = seq_result

    # Aggregate
    results["average"] = {
        "JM": float(np.mean(all_jm)) if all_jm else None,
        "JR": float(np.mean(all_jr)) if all_jr else None,
        "PSNR": float(np.mean(all_psnr)) if all_psnr else None,
        "SSIM": float(np.mean(all_ssim)) if all_ssim else None,
    }

    return results


def print_results_table(results: dict):
    """Pretty-print evaluation results as a table."""
    header = f"{'Sequence':<20} {'JM':>8} {'JR':>8} {'PSNR':>8} {'SSIM':>8}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for seq, metrics in results.items():
        if seq == "average":
            continue
        row = f"{seq:<20}"
        for key in ("JM", "JR", "PSNR", "SSIM"):
            val = metrics.get(key)
            row += f" {val:8.4f}" if val is not None else f" {'N/A':>8}"
        print(row)

    print("-" * len(header))
    avg = results.get("average", {})
    row = f"{'AVERAGE':<20}"
    for key in ("JM", "JR", "PSNR", "SSIM"):
        val = avg.get(key)
        row += f" {val:8.4f}" if val is not None else f" {'N/A':>8}"
    print(row)
    print("=" * len(header))
