"""
Mask Extraction Module
======================
Stage 1: YOLOv8-Seg semantic extraction + Lucas-Kanade optical flow dynamic
filtering + morphological dilation.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class MaskExtractor:
    """Extract dynamic object masks from video frames."""

    def __init__(self, config: dict):
        mask_cfg = config["mask"]
        self.model = YOLO(mask_cfg["model"])
        self.target_classes = set(mask_cfg["target_classes"])
        self.conf_threshold = mask_cfg["confidence_threshold"]
        self.device = mask_cfg.get("device", "cuda:0")

        df_cfg = config["dynamic_filter"]
        self.dynamic_filter_enabled = df_cfg["enabled"]
        self.motion_threshold = df_cfg["motion_threshold"]
        self.min_features = df_cfg["min_features"]

        dil_cfg = config["dilation"]
        self.dilation_kernel_size = dil_cfg["kernel_size"]
        self.dilation_iterations = dil_cfg["iterations"]

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_masks(self, frame: np.ndarray):
        """Run YOLOv8-Seg on a single frame.

        Returns:
            combined_mask: (H, W) uint8 binary mask (0 or 255) of all targets.
            instance_masks: list of (H, W) uint8 binary masks, one per instance.
        """
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        instance_masks = []
        if result.masks is not None:
            for i, cls_id in enumerate(result.boxes.cls):
                if int(cls_id.item()) in self.target_classes:
                    # Get mask from YOLO result and resize to frame size
                    seg = result.masks.data[i].cpu().numpy()  # (mask_h, mask_w)
                    mask = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    instance_masks.append(mask)

        combined = self._combine_masks(instance_masks, h, w)
        return combined, instance_masks

    def filter_dynamic(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        instance_masks: list,
    ) -> tuple:
        """Filter out static objects using sparse optical flow.

        Returns:
            combined_mask: merged binary mask of dynamic-only instances.
            dynamic_masks: list of instance masks that are dynamic.
        """
        if not self.dynamic_filter_enabled or prev_gray is None:
            h, w = curr_gray.shape[:2]
            return self._combine_masks(instance_masks, h, w), instance_masks

        dynamic_masks = []
        for mask in instance_masks:
            # Detect features inside this instance mask
            pts = cv2.goodFeaturesToTrack(
                prev_gray, mask=mask, **self.feature_params
            )
            if pts is None or len(pts) < self.min_features:
                # Not enough features – keep it (conservative)
                dynamic_masks.append(mask)
                continue

            # Calculate optical flow
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts, None, **self.lk_params
            )
            good_old = pts[status.flatten() == 1]
            good_new = next_pts[status.flatten() == 1]

            if len(good_old) < self.min_features:
                dynamic_masks.append(mask)
                continue

            displacement = np.sqrt(
                np.sum((good_new - good_old) ** 2, axis=1)
            )
            mean_disp = np.mean(displacement)

            if mean_disp >= self.motion_threshold:
                dynamic_masks.append(mask)
            # else: static object – discard

        h, w = curr_gray.shape[:2]
        combined = self._combine_masks(dynamic_masks, h, w)
        return combined, dynamic_masks

    def dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological dilation to the mask."""
        if self.dilation_kernel_size <= 0 or self.dilation_iterations <= 0:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.dilation_kernel_size, self.dilation_kernel_size),
        )
        return cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_masks(masks: list, h: int, w: int) -> np.ndarray:
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in masks:
            combined = cv2.bitwise_or(combined, m)
        return combined
