"""Side-effect-aware mask construction."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .mask_utils import (
    close_mask,
    dilate_mask,
    fill_holes,
    flip_mask_across_vertical_axis,
    polygon_to_mask,
    shift_mask,
    union_masks,
)


def build_side_effect_mask(
    frame,
    object_mask,
    config: dict,
    prev_frame=None,
    next_frame=None,
    prev_mask=None,
    next_mask=None,
):
    side_cfg = config.get("side_effect", {})
    object_mask = (object_mask > 127).astype(np.uint8) * 255

    expanded = dilate_mask(object_mask, side_cfg.get("dilate_radius", 21))
    expanded = close_mask(expanded, side_cfg.get("close_radius", 11))
    if side_cfg.get("fill_holes", True):
        expanded = fill_holes(expanded)

    shifted = shift_mask(
        object_mask,
        dx=0,
        dy=side_cfg.get("shadow_down_shift", 35),
    )
    shadow_mask = dilate_mask(shifted, side_cfg.get("shadow_radius", 45))

    reflection_mask = np.zeros_like(object_mask)
    for roi in side_cfg.get("reflection_rois", []):
        polygon = roi.get("polygon", [])
        if not polygon:
            continue
        roi_mask = polygon_to_mask(object_mask.shape[:2], polygon)
        reflection_mask = union_masks(reflection_mask, cv2.bitwise_and(expanded, roi_mask))

    mirror_cfg = side_cfg.get("mirror", {})
    if mirror_cfg.get("enable", False) and mirror_cfg.get("axis") == "vertical":
        flipped = flip_mask_across_vertical_axis(object_mask, int(mirror_cfg.get("x_center", 0)))
        roi_name = mirror_cfg.get("roi_name", "")
        if roi_name:
            roi_match = None
            for roi in side_cfg.get("reflection_rois", []):
                if roi.get("name") == roi_name:
                    roi_match = roi
                    break
            if roi_match:
                roi_mask = polygon_to_mask(object_mask.shape[:2], roi_match.get("polygon", []))
                flipped = cv2.bitwise_and(flipped, roi_mask)
        reflection_mask = union_masks(reflection_mask, flipped)

    final_mask = union_masks(object_mask, expanded, shadow_mask, reflection_mask)
    return {
        "object_mask": object_mask,
        "expanded_mask": expanded,
        "shadow_mask": shadow_mask,
        "reflection_mask": reflection_mask,
        "final_mask": final_mask,
    }


def build_side_effect_masks_for_dir(frames_dir: str, object_mask_dir: str, output_root: str, config: dict) -> dict[str, str]:
    frame_paths = sorted(
        [os.path.join(frames_dir, name) for name in os.listdir(frames_dir)]
    )
    mask_paths = sorted(
        [os.path.join(object_mask_dir, name) for name in os.listdir(object_mask_dir) if name.endswith(".png")]
    )
    outputs = {
        "object_mask": os.path.join(output_root, "object_mask"),
        "expanded_mask": os.path.join(output_root, "expanded_mask"),
        "shadow_mask": os.path.join(output_root, "shadow_mask"),
        "reflection_mask": os.path.join(output_root, "reflection_mask"),
        "final_side_effect_mask": os.path.join(output_root, "final_side_effect_mask"),
    }
    for path in outputs.values():
        os.makedirs(path, exist_ok=True)

    from .mask_utils import load_mask, save_mask

    for frame_path, mask_path in zip(frame_paths, mask_paths):
        frame = cv2.imread(frame_path)
        mask = load_mask(mask_path)
        built = build_side_effect_mask(frame, mask, config)
        basename = os.path.basename(mask_path)
        for key, out_dir in outputs.items():
            source_key = "final_mask" if key == "final_side_effect_mask" else key
            save_mask(os.path.join(out_dir, basename), built[source_key])

    return outputs

