"""
SAM 2 Mask Extractor
====================
Video object segmentation using Meta's SAM 2 with prompts
derived from first-frame ground truth annotations.
"""

import os
import glob

import numpy as np
import cv2
import torch


class SAM2MaskExtractor:
    """Extract object masks using SAM 2 video predictor."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        sam2_cfg = config["sam2"]
        checkpoint = sam2_cfg["checkpoint"]
        model_cfg = sam2_cfg["model_cfg"]

        print(f"[SAM2] Loading model: {model_cfg} from {checkpoint} ...")

        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=self.device,
        )
        print("[SAM2] Model loaded.")

    def _get_prompts_from_gt(self, gt_mask_path: str) -> list:
        """Extract point and bbox prompts from a GT annotation mask.

        Returns list of dicts, one per object:
            {"object_id": int, "points": [[x,y]], "labels": [1], "box": [x1,y1,x2,y2]}
        """
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            return []

        object_ids = sorted(set(gt_mask.flatten()) - {0})
        prompts = []

        for obj_id in object_ids:
            obj_mask = (gt_mask == obj_id).astype(np.uint8)
            ys, xs = np.where(obj_mask > 0)
            if len(xs) == 0:
                continue

            # Centroid
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            # Bounding box
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())

            prompts.append({
                "object_id": int(obj_id),
                "points": np.array([[cx, cy]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
                "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            })

        return prompts

    @torch.no_grad()
    def extract_masks(self, frames_dir: str, gt_ann_dir: str,
                      output_masks_dir: str):
        """Extract masks using SAM 2 video propagation.

        Args:
            frames_dir: directory with input frames (00000.jpg, ...).
            gt_ann_dir: directory with GT annotation masks (for first-frame prompt).
            output_masks_dir: directory to save predicted masks.
        """
        os.makedirs(output_masks_dir, exist_ok=True)

        frame_paths = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg"))
            + glob.glob(os.path.join(frames_dir, "*.png"))
        )
        if not frame_paths:
            print(f"[SAM2] No frames found in {frames_dir}")
            return

        print(f"[SAM2] Processing {len(frame_paths)} frames from {frames_dir}")

        # Get first-frame GT annotation for prompts
        gt_files = sorted(glob.glob(os.path.join(gt_ann_dir, "*.png")))
        if not gt_files:
            print(f"[SAM2] No GT annotations found in {gt_ann_dir}")
            return

        prompts = self._get_prompts_from_gt(gt_files[0])
        if not prompts:
            print("[SAM2] No objects found in first-frame GT annotation")
            return

        print(f"[SAM2] Found {len(prompts)} objects in first frame")

        # Initialize video state using the frames directory
        # SAM2 expects a directory of JPEG frames
        with torch.inference_mode(), torch.autocast(
            str(self.device), dtype=torch.bfloat16
        ):
            state = self.predictor.init_state(video_path=frames_dir)

            # Add prompts for each object on frame 0
            for prompt in prompts:
                strategy = self.config["sam2"].get("prompt_strategy",
                                                   "centroid_and_bbox")
                if strategy == "centroid_and_bbox":
                    self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=prompt["object_id"],
                        points=prompt["points"],
                        labels=prompt["labels"],
                        box=prompt["box"],
                    )
                elif strategy == "bbox":
                    self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=prompt["object_id"],
                        box=prompt["box"],
                    )
                else:  # centroid only
                    self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=prompt["object_id"],
                        points=prompt["points"],
                        labels=prompt["labels"],
                    )

            # Propagate through video
            frame_masks = {}
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(
                state
            ):
                # masks: (num_objects, 1, H, W) logits -> binary
                combined = torch.zeros_like(masks[0, 0])  # (H, W)
                for obj_mask in masks:
                    combined = torch.logical_or(
                        combined, obj_mask[0] > 0.0
                    )
                frame_masks[frame_idx] = combined.cpu().numpy().astype(np.uint8) * 255

        # Read original frame dims
        sample_frame = cv2.imread(frame_paths[0])
        orig_h, orig_w = sample_frame.shape[:2]

        # Save masks
        for i, frame_path in enumerate(frame_paths):
            fname = os.path.splitext(os.path.basename(frame_path))[0] + ".png"
            if i in frame_masks:
                mask = frame_masks[i]
                if mask.shape[:2] != (orig_h, orig_w):
                    mask = cv2.resize(mask, (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.imwrite(os.path.join(output_masks_dir, fname), mask)

        print(f"[SAM2] Saved {len(frame_paths)} masks to {output_masks_dir}")
