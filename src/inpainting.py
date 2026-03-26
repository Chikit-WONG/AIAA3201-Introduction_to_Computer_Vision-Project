"""
Inpainting Module
=================
Stage 2: Temporal background propagation (toggleable) + cv2.inpaint fallback.
"""

import cv2
import numpy as np


class TemporalPropagator:
    """Borrow clean pixels from neighbouring frames via a sliding window."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size  # look k frames in each direction

    def propagate(
        self,
        frames_buffer: list,
        masks_buffer: list,
        center_idx: int,
    ) -> tuple:
        """Fill masked pixels in the center frame using temporal neighbours.

        Args:
            frames_buffer: list of BGR frames (np.ndarray) in the window.
            masks_buffer:  list of binary masks (0/255, uint8) for each frame.
            center_idx:    index of the current frame inside the buffer.

        Returns:
            repaired_frame: center frame with temporally-borrowed pixels.
            residual_mask:  remaining mask of pixels that couldn't be filled.
        """
        center_frame = frames_buffer[center_idx].copy()
        center_mask = masks_buffer[center_idx].copy()

        n = len(frames_buffer)
        # Build traversal order: closest neighbours first
        # e.g. center-1, center+1, center-2, center+2, ...
        order = []
        for delta in range(1, n):
            for sign in (-1, 1):
                idx = center_idx + sign * delta
                if 0 <= idx < n:
                    order.append(idx)

        # Vectorised fill per neighbour frame
        for idx in order:
            if np.count_nonzero(center_mask) == 0:
                break  # nothing left to fill
            neighbour_mask = masks_buffer[idx]
            # Pixels that are masked in center but clean in neighbour
            fillable = (center_mask == 255) & (neighbour_mask == 0)
            if not np.any(fillable):
                continue
            # Borrow pixels
            center_frame[fillable] = frames_buffer[idx][fillable]
            # Update residual mask
            center_mask[fillable] = 0

        return center_frame, center_mask


class SpatialInpainter:
    """Fallback: fill remaining holes using cv2.inpaint."""

    METHODS = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS,
    }

    def __init__(self, radius: int = 5, method: str = "telea"):
        self.radius = radius
        self.method = self.METHODS.get(method, cv2.INPAINT_TELEA)

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using spatial neighbourhood.

        Args:
            frame: BGR image (H, W, 3).
            mask:  binary mask (0/255, uint8).

        Returns:
            Repaired frame.
        """
        if np.count_nonzero(mask) == 0:
            return frame
        return cv2.inpaint(frame, mask, self.radius, self.method)
