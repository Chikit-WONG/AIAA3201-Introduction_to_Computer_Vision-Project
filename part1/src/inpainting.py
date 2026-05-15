"""
Inpainting Module
=================
Stage 2: Temporal background propagation (toggleable) + cv2.inpaint fallback.
"""

import cv2
import numpy as np


class TemporalPropagator:
    """Borrow clean pixels from neighbouring frames via a sliding window.

    When align=True (default), each neighbour frame is warped to the current
    frame's coordinate system using dense optical flow (Farneback) before
    pixels are copied.  This corrects for camera motion and prevents the
    blurry-but-wrong-position artefacts that occur with direct pixel copy.
    """

    def __init__(self, window_size: int = 5, align: bool = True):
        self.window_size = window_size
        self.align = align

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_flow(gray_src: np.ndarray,
                      gray_dst: np.ndarray) -> np.ndarray:
        """Compute dense Farneback optical flow from src → dst.

        Returns flow[y, x] = (dx, dy): displacement to apply to src coords
        to obtain the corresponding point in dst.
        """
        return cv2.calcOpticalFlowFarneback(
            gray_src, gray_dst,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    @staticmethod
    def _remap(src: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp src with backward flow so it aligns with the destination frame.

        flow[y, x] = (dx, dy) means pixel (x,y) in dst comes from (x+dx, y+dy)
        in src.  We build the remap maps accordingly.
        """
        h, w = flow.shape[:2]
        base_x = np.arange(w, dtype=np.float32)
        base_y = np.arange(h, dtype=np.float32)
        map_x, map_y = np.meshgrid(base_x, base_y)
        map_x += flow[..., 0]
        map_y += flow[..., 1]
        return cv2.remap(
            src, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _warp_neighbour(self, neighbour_frame: np.ndarray,
                        neighbour_mask: np.ndarray,
                        center_gray: np.ndarray) -> tuple:
        """Align neighbour frame/mask to center frame using optical flow."""
        neighbour_gray = cv2.cvtColor(neighbour_frame, cv2.COLOR_BGR2GRAY)
        # Flow from center → neighbour (backward flow for warping)
        flow = self._compute_flow(center_gray, neighbour_gray)
        warped_frame = self._remap(neighbour_frame, flow)
        # Warp mask: float remap then threshold back to binary
        mask_f = neighbour_mask.astype(np.float32)
        warped_mask_f = self._remap(mask_f, flow)
        warped_mask = (warped_mask_f > 127).astype(np.uint8) * 255
        return warped_frame, warped_mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        if self.align:
            center_gray = cv2.cvtColor(center_frame, cv2.COLOR_BGR2GRAY)

        n = len(frames_buffer)
        # Build traversal order: closest neighbours first
        order = []
        for delta in range(1, n):
            for sign in (-1, 1):
                idx = center_idx + sign * delta
                if 0 <= idx < n:
                    order.append(idx)

        for idx in order:
            if np.count_nonzero(center_mask) == 0:
                break  # nothing left to fill

            neighbour_frame = frames_buffer[idx]
            neighbour_mask = masks_buffer[idx]

            if self.align:
                neighbour_frame, neighbour_mask = self._warp_neighbour(
                    neighbour_frame, neighbour_mask, center_gray
                )

            # Pixels that are masked in center but clean in neighbour
            fillable = (center_mask == 255) & (neighbour_mask == 0)
            if not np.any(fillable):
                continue
            center_frame[fillable] = neighbour_frame[fillable]
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
