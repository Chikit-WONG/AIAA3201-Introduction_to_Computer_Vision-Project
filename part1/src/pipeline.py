"""
Pipeline Module
===============
Main orchestrator: reads frames -> mask extraction -> dynamic filter ->
dilation -> temporal propagation (optional) -> spatial inpaint -> output.

Supports both video file input and DAVIS image-sequence input.
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

from src.mask_extraction import MaskExtractor
from src.inpainting import TemporalPropagator, SpatialInpainter


class VideoPipeline:
    """End-to-end video object removal pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.extractor = MaskExtractor(config)

        tp_cfg = config["temporal_propagation"]
        self.use_temporal = tp_cfg["enabled"]
        self.window_size = tp_cfg["window_size"]
        self.propagator = TemporalPropagator(self.window_size)

        sp_cfg = config["spatial_inpaint"]
        self.inpainter = SpatialInpainter(
            radius=sp_cfg["radius"],
            method=sp_cfg["method"],
        )

        out_cfg = config["output"]
        self.save_masks = out_cfg["save_masks"]
        self.save_vis = out_cfg["save_visualization"]
        self.video_fps = out_cfg["video_fps"]
        self.video_codec = out_cfg["video_codec"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_davis_sequence(self, seq_dir: str, output_dir: str):
        """Process a DAVIS sequence (directory of JPEG frames).

        Args:
            seq_dir:    path to JPEGImages/480p/<seq_name>/
            output_dir: output root for this sequence.
        """
        frame_paths = sorted(
            glob.glob(os.path.join(seq_dir, "*.jpg"))
            + glob.glob(os.path.join(seq_dir, "*.png"))
        )
        if not frame_paths:
            print(f"[WARN] No frames found in {seq_dir}")
            return

        # Prepare output directories
        frames_out = os.path.join(output_dir, "frames")
        masks_out = os.path.join(output_dir, "masks")
        vis_out = os.path.join(output_dir, "visualization")
        os.makedirs(frames_out, exist_ok=True)
        if self.save_masks:
            os.makedirs(masks_out, exist_ok=True)
        if self.save_vis:
            os.makedirs(vis_out, exist_ok=True)

        # ---- Pass 1: extract all masks ----
        print(f"  [1/3] Extracting masks ({len(frame_paths)} frames)...")
        all_frames = []
        all_masks = []
        prev_gray = None

        for path in tqdm(frame_paths, desc="  Mask extraction", leave=False):
            frame = cv2.imread(path)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            combined_mask, instance_masks = self.extractor.extract_masks(frame)

            # Dynamic filtering
            combined_mask, _ = self.extractor.filter_dynamic(
                prev_gray, curr_gray, instance_masks,
            )

            # Dilation
            combined_mask = self.extractor.dilate_mask(combined_mask)

            all_frames.append(frame)
            all_masks.append(combined_mask)
            prev_gray = curr_gray

        # ---- Pass 2: inpainting ----
        print(f"  [2/3] Inpainting ({'temporal+spatial' if self.use_temporal else 'spatial only'})...")
        num_frames = len(all_frames)

        # Temporal propagation statistics
        tp_stats = {
            "frames_with_mask": 0,
            "frames_temporal_helped": 0,
            "total_masked_pixels": 0,
            "temporal_filled_pixels": 0,
            "spatial_filled_pixels": 0,
            "frames_fully_filled_by_temporal": 0,
        }

        for i in tqdm(range(num_frames), desc="  Inpainting", leave=False):
            frame = all_frames[i]
            mask = all_masks[i]

            mask_pixels = np.count_nonzero(mask)

            if mask_pixels == 0:
                # No mask – nothing to repair
                repaired = frame
                residual_mask = mask
            elif self.use_temporal:
                tp_stats["frames_with_mask"] += 1
                tp_stats["total_masked_pixels"] += mask_pixels

                # Build sliding window
                buf_start = max(0, i - self.window_size)
                buf_end = min(num_frames, i + self.window_size + 1)
                center_idx = i - buf_start

                frames_buf = all_frames[buf_start:buf_end]
                masks_buf = all_masks[buf_start:buf_end]

                repaired, residual_mask = self.propagator.propagate(
                    frames_buf, masks_buf, center_idx,
                )

                residual_pixels = np.count_nonzero(residual_mask)
                filled_by_temporal = mask_pixels - residual_pixels
                tp_stats["temporal_filled_pixels"] += filled_by_temporal
                tp_stats["spatial_filled_pixels"] += residual_pixels

                if filled_by_temporal > 0:
                    tp_stats["frames_temporal_helped"] += 1
                if residual_pixels == 0:
                    tp_stats["frames_fully_filled_by_temporal"] += 1

                # Spatial fallback for remaining pixels
                repaired = self.inpainter.inpaint(repaired, residual_mask)
            else:
                # Spatial-only mode
                repaired = self.inpainter.inpaint(frame, mask)
                residual_mask = mask

            # Save outputs
            fname = os.path.basename(frame_paths[i])
            fname_png = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(frames_out, fname_png), repaired)

            if self.save_masks:
                cv2.imwrite(os.path.join(masks_out, fname_png), mask)

            if self.save_vis:
                vis = self._make_visualization(frame, mask, repaired)
                cv2.imwrite(os.path.join(vis_out, fname_png), vis)

        # Print temporal propagation statistics
        if self.use_temporal:
            print("\n  === Temporal Propagation Statistics ===")
            fm = tp_stats["frames_with_mask"]
            fh = tp_stats["frames_temporal_helped"]
            ff = tp_stats["frames_fully_filled_by_temporal"]
            tp_px = tp_stats["temporal_filled_pixels"]
            sp_px = tp_stats["spatial_filled_pixels"]
            total_px = tp_stats["total_masked_pixels"]
            print(f"  Frames with mask:              {fm} / {num_frames}")
            print(f"  Frames temporal helped:        {fh} / {fm} "
                  f"({100*fh/fm:.1f}%)" if fm > 0 else "")
            print(f"  Frames fully filled by temporal:{ff} / {fm} "
                  f"({100*ff/fm:.1f}%)" if fm > 0 else "")
            print(f"  Total masked pixels:           {total_px:,}")
            print(f"  Filled by temporal:            {tp_px:,} "
                  f"({100*tp_px/total_px:.1f}%)" if total_px > 0 else "")
            print(f"  Filled by spatial fallback:    {sp_px:,} "
                  f"({100*sp_px/total_px:.1f}%)" if total_px > 0 else "")
            print("  ======================================\n")

        # ---- Pass 3: assemble video ----
        print("  [3/3] Assembling output video...")
        self._assemble_video(frames_out, output_dir)
        print("  Done.")

    def process_video_file(self, video_path: str, output_dir: str):
        """Process a video file (mp4, avi, etc.).

        Extracts frames, processes, and writes output video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            print(f"[WARN] No frames read from {video_path}")
            return

        # Prepare output directories
        frames_out = os.path.join(output_dir, "frames")
        masks_out = os.path.join(output_dir, "masks")
        vis_out = os.path.join(output_dir, "visualization")
        os.makedirs(frames_out, exist_ok=True)
        if self.save_masks:
            os.makedirs(masks_out, exist_ok=True)
        if self.save_vis:
            os.makedirs(vis_out, exist_ok=True)

        # ---- Pass 1: extract masks ----
        print(f"  [1/3] Extracting masks ({len(frames)} frames)...")
        all_masks = []
        prev_gray = None

        for frame in tqdm(frames, desc="  Mask extraction", leave=False):
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            combined_mask, instance_masks = self.extractor.extract_masks(frame)
            combined_mask, _ = self.extractor.filter_dynamic(
                prev_gray, curr_gray, instance_masks,
            )
            combined_mask = self.extractor.dilate_mask(combined_mask)
            all_masks.append(combined_mask)
            prev_gray = curr_gray

        # ---- Pass 2: inpainting ----
        print(f"  [2/3] Inpainting ({'temporal+spatial' if self.use_temporal else 'spatial only'})...")
        num_frames = len(frames)

        for i in tqdm(range(num_frames), desc="  Inpainting", leave=False):
            frame = frames[i]
            mask = all_masks[i]

            if np.count_nonzero(mask) == 0:
                repaired = frame
            elif self.use_temporal:
                buf_start = max(0, i - self.window_size)
                buf_end = min(num_frames, i + self.window_size + 1)
                center_idx = i - buf_start
                repaired, residual = self.propagator.propagate(
                    frames[buf_start:buf_end],
                    all_masks[buf_start:buf_end],
                    center_idx,
                )
                repaired = self.inpainter.inpaint(repaired, residual)
            else:
                repaired = self.inpainter.inpaint(frame, mask)

            fname = f"{i:05d}.png"
            cv2.imwrite(os.path.join(frames_out, fname), repaired)
            if self.save_masks:
                cv2.imwrite(os.path.join(masks_out, fname), mask)
            if self.save_vis:
                vis = self._make_visualization(frame, mask, repaired)
                cv2.imwrite(os.path.join(vis_out, fname), vis)

        # ---- Pass 3: assemble video ----
        print("  [3/3] Assembling output video...")
        self._assemble_video(frames_out, output_dir)
        print("  Done.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _make_visualization(
        original: np.ndarray,
        mask: np.ndarray,
        repaired: np.ndarray,
    ) -> np.ndarray:
        """Create a side-by-side visualization: original | mask overlay | repaired."""
        h, w = original.shape[:2]

        # Mask overlay (red channel)
        overlay = original.copy()
        overlay[mask > 127, 2] = 255  # Red channel

        vis = np.hstack([original, overlay, repaired])
        return vis

    def _assemble_video(self, frames_dir: str, output_dir: str):
        """Assemble PNG frames into an MP4 video."""
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not frame_files:
            return

        sample = cv2.imread(frame_files[0])
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        video_path = os.path.join(output_dir, "output.mp4")
        writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (w, h))

        for fpath in frame_files:
            frame = cv2.imread(fpath)
            writer.write(frame)
        writer.release()
        print(f"  Video saved: {video_path}")
