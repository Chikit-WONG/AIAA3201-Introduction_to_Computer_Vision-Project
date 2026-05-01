"""Frame alignment helpers for wild paired evaluation."""

from __future__ import annotations

import cv2
import numpy as np


def align_frame_ecc(pred_frame: np.ndarray, gt_frame: np.ndarray) -> tuple[np.ndarray, bool]:
    if pred_frame.shape != gt_frame.shape:
        pred_frame = cv2.resize(pred_frame, (gt_frame.shape[1], gt_frame.shape[0]))

    pred_gray = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-6,
    )
    try:
        _cc, warp = cv2.findTransformECC(gt_gray, pred_gray, warp, cv2.MOTION_EUCLIDEAN, criteria)
        aligned = cv2.warpAffine(
            pred_frame,
            warp,
            (gt_frame.shape[1], gt_frame.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, True
    except cv2.error:
        return pred_frame, False

