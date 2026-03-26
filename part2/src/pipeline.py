"""
Video Removal Pipeline
======================
Orchestrates: mask extraction (VGGT4D or SAM 2) -> inpainting (ProPainter).
"""

import os
import time


class VideoRemovalPipeline:
    """End-to-end video object removal pipeline."""

    def __init__(self, config: dict):
        self.config = config
        method = config.get("method", "sam2")
        self.method = method

        # Set GPU
        gpu_id = config.get("gpu_id", 0)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Lazy-load extractors to avoid importing both models at once
        self._mask_extractor = None
        self._inpainter = None

    @property
    def mask_extractor(self):
        if self._mask_extractor is None:
            if self.method == "vggt4d":
                from src.mask_vggt4d import VGGT4DMaskExtractor
                self._mask_extractor = VGGT4DMaskExtractor(self.config)
            elif self.method == "sam2":
                from src.mask_sam2 import SAM2MaskExtractor
                self._mask_extractor = SAM2MaskExtractor(self.config)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        return self._mask_extractor

    @property
    def inpainter(self):
        if self._inpainter is None:
            from src.inpaint_propainter import ProPainterInpainter
            self._inpainter = ProPainterInpainter(self.config)
        return self._inpainter

    def process_sequence(self, seq_name: str, frames_dir: str,
                         gt_ann_dir: str, output_dir: str):
        """Process a single DAVIS sequence.

        Args:
            seq_name: sequence name (e.g., "bmx-trees").
            frames_dir: path to input JPEG frames.
            gt_ann_dir: path to GT annotation masks.
            output_dir: root output directory for this sequence.
        """
        masks_dir = os.path.join(output_dir, "masks")
        inpainted_dir = os.path.join(output_dir, "frames")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(inpainted_dir, exist_ok=True)

        t0 = time.time()

        # Step 1: Mask extraction
        print(f"\n[Pipeline] Step 1: Extracting masks ({self.method}) ...")
        if self.method == "sam2":
            self.mask_extractor.extract_masks(
                frames_dir, gt_ann_dir, masks_dir
            )
        else:  # vggt4d
            self.mask_extractor.extract_masks(frames_dir, masks_dir)

        t1 = time.time()
        print(f"[Pipeline] Mask extraction: {t1 - t0:.1f}s")

        # Step 2: Inpainting with ProPainter
        print(f"\n[Pipeline] Step 2: Inpainting with ProPainter ...")
        self.inpainter.inpaint(frames_dir, masks_dir, inpainted_dir)

        t2 = time.time()
        print(f"[Pipeline] Inpainting: {t2 - t1:.1f}s")
        print(f"[Pipeline] Total: {t2 - t0:.1f}s for {seq_name}")

    def process_davis(self, davis_root: str, output_root: str,
                      sequences: list, resolution: str = "480p"):
        """Process multiple DAVIS sequences.

        Args:
            davis_root: root of DAVIS dataset.
            output_root: root output directory.
            sequences: list of sequence names.
            resolution: "480p" or "Full-Resolution".
        """
        jpeg_root = os.path.join(davis_root, "JPEGImages", resolution)
        annot_root = os.path.join(davis_root, "Annotations", resolution)

        total_t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Processing {len(sequences)} sequences with {self.method}")
        print(f"{'='*60}")

        for i, seq in enumerate(sequences):
            frames_dir = os.path.join(jpeg_root, seq)
            gt_ann_dir = os.path.join(annot_root, seq)
            output_dir = os.path.join(output_root, seq)

            if not os.path.isdir(frames_dir):
                print(f"[WARN] Frames not found: {frames_dir}, skipping.")
                continue

            print(f"\n[{i+1}/{len(sequences)}] Sequence: {seq}")
            print(f"{'-'*40}")
            self.process_sequence(seq, frames_dir, gt_ann_dir, output_dir)

        total_t = time.time() - total_t0
        print(f"\n{'='*60}")
        print(f"All {len(sequences)} sequences processed in {total_t:.1f}s")
        print(f"Results saved to: {output_root}")
        print(f"{'='*60}")
