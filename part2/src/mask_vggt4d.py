"""
VGGT4D Mask Extractor
=====================
Training-free dynamic object segmentation using Vision Transformer attention maps
and Gram similarity from VGGT4D.
"""

import os
import sys
import glob

import numpy as np
import cv2
import torch
from PIL import Image


class VGGT4DMaskExtractor:
    """Extract dynamic object masks using VGGT4D."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        # Add VGGT4D repo to sys.path
        repo_dir = os.path.abspath(config["vggt4d"]["repo_dir"])
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        checkpoint = config["vggt4d"]["checkpoint"]
        print(f"[VGGT4D] Loading model from {checkpoint} ...")

        from vggt4d.models.vggt4d import VGGTFor4D

        self.model = VGGTFor4D(img_size=518, patch_size=14, embed_dim=1024)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        print("[VGGT4D] Model loaded.")

    @torch.no_grad()
    def extract_masks(self, frames_dir: str, output_masks_dir: str):
        """Extract dynamic masks from video frames.

        Args:
            frames_dir: directory containing input JPEG frames (00000.jpg, ...).
            output_masks_dir: directory to save predicted masks (00000.png, ...).
        """
        os.makedirs(output_masks_dir, exist_ok=True)

        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt4d.utils.model_utils import inference

        frame_paths = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg"))
            + glob.glob(os.path.join(frames_dir, "*.png"))
        )
        if not frame_paths:
            print(f"[VGGT4D] No frames found in {frames_dir}")
            return

        print(f"[VGGT4D] Processing {len(frame_paths)} frames from {frames_dir}")

        # Load and preprocess images
        # Note: VGGTFor4D.forward() internally handles unsqueeze(0) for 4D input [S,3,H,W]
        # The inference() function expects unbatched images [S,3,H,W]
        images = load_and_preprocess_images(frame_paths, mode="crop")
        images = images.to(self.device)

        # Run inference
        predictions, qk_dict, enc_feat, agg_tokens_list = inference(
            self.model, images, dyn_masks=None, query_points=None
        )

        # Extract dynamic masks from attention-based Gram similarity
        # After inference(), all predictions are numpy arrays with batch dim squeezed
        if "dynamic_masks" in predictions:
            dyn_masks = predictions["dynamic_masks"]  # [S, H, W] numpy
        else:
            # Fallback: compute masks from depth confidence
            depth_conf = predictions.get("depth_conf", None)
            if depth_conf is not None:
                # depth_conf is numpy array [S, H, W] or [S, H, W, 1]
                if depth_conf.ndim == 4:
                    depth_conf = depth_conf.squeeze(-1)  # [S, H, W]
                # Use Otsu thresholding per frame to separate dynamic/static
                # High confidence = static, low confidence = dynamic
                dyn_masks_list = []
                for i in range(depth_conf.shape[0]):
                    conf_frame = depth_conf[i]
                    # Normalize to 0-255
                    conf_norm = ((conf_frame - conf_frame.min()) /
                                 (conf_frame.max() - conf_frame.min() + 1e-8) * 255).astype(np.uint8)
                    # Invert: low confidence -> dynamic
                    conf_inv = 255 - conf_norm
                    _, mask = cv2.threshold(conf_inv, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    dyn_masks_list.append(mask)
                dyn_masks = np.stack(dyn_masks_list, axis=0)  # [S, H, W]

        if isinstance(dyn_masks, torch.Tensor):
            dyn_masks = dyn_masks.cpu().numpy()
        # Ensure dyn_masks is [S, H, W]
        if dyn_masks.ndim == 4:
            dyn_masks = dyn_masks[0]

        # Read original frame dimensions for resizing
        sample_frame = cv2.imread(frame_paths[0])
        orig_h, orig_w = sample_frame.shape[:2]

        # Save masks
        for i, frame_path in enumerate(frame_paths):
            fname = os.path.splitext(os.path.basename(frame_path))[0] + ".png"
            if i < dyn_masks.shape[0]:
                mask = dyn_masks[i]
                # Ensure binary (0 or 255)
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
                mask = (mask > 127).astype(np.uint8) * 255
                # Resize to original frame dimensions
                if mask.shape[:2] != (orig_h, orig_w):
                    mask = cv2.resize(mask, (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            cv2.imwrite(os.path.join(output_masks_dir, fname), mask)

        print(f"[VGGT4D] Saved {len(frame_paths)} masks to {output_masks_dir}")
