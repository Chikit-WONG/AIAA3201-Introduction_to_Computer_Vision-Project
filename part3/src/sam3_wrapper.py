"""SAM 3 mask generation wrapper."""

from __future__ import annotations

import gc
import getpass
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import ensure_dir, slugify
from .mask_utils import save_mask, union_masks
from .video_utils import extract_video_frames, get_video_info, list_frames, write_video


class SAM3MaskGenerator:
    def __init__(self, sam3_dir: str, checkpoint: str | None = None, device: str = "cuda",
                 version: str = "sam3", frame_index: int = 0, compile: bool = False,
                 async_loading_frames: bool = False, use_fa3: bool | None = None,
                 use_rope_real: bool | None = None):
        self.sam3_dir = os.path.abspath(sam3_dir)
        self.checkpoint = checkpoint or None
        self.device = device
        self.version = version
        self.frame_index = frame_index
        self.compile = compile
        self.async_loading_frames = async_loading_frames
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real
        if not os.path.isdir(self.sam3_dir):
            raise FileNotFoundError(f"SAM 3 repository not found: {self.sam3_dir}")

    def _build_model(self):
        if self.sam3_dir not in sys.path:
            sys.path.insert(0, self.sam3_dir)
        username = getpass.getuser()
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_cache_{username}")
        from sam3 import build_sam3_predictor

        build_kwargs = {
            "version": self.version,
            "compile": self.compile,
            "async_loading_frames": self.async_loading_frames,
        }
        if self.use_fa3 is not None:
            build_kwargs["use_fa3"] = self.use_fa3
        elif self.version == "sam3.1":
            build_kwargs["use_fa3"] = False
        if self.use_rope_real is not None:
            build_kwargs["use_rope_real"] = self.use_rope_real
        if self.checkpoint:
            build_kwargs["checkpoint_path"] = self.checkpoint

        try:
            return build_sam3_predictor(**build_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "SAM 3 checkpoint is unavailable. Please request access and set "
                "SAM3 checkpoint path in the config, or run with --allow-existing-masks."
            ) from exc

    @staticmethod
    def _collect_propagation(model, session_id: str) -> dict[int, dict[int, np.ndarray]]:
        mask_dict: dict[int, dict[int, np.ndarray]] = {}
        for response in model.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_idx = response.get("frame_index")
            if frame_idx is None:
                continue
            outputs = response.get("outputs", {})
            obj_ids = outputs.get("out_obj_ids", [])
            binary_masks = outputs.get("out_binary_masks")
            if binary_masks is None:
                mask_dict[frame_idx] = {}
                continue
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.cpu().numpy()
            if isinstance(binary_masks, torch.Tensor):
                binary_masks = binary_masks.cpu().numpy()
            masks = {}
            for index, object_id in enumerate(obj_ids):
                mask = binary_masks[index]
                if mask.ndim == 3:
                    mask = mask[0]
                masks[int(object_id)] = (mask > 0).astype(np.uint8) * 255
            mask_dict[frame_idx] = masks
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return mask_dict

    @staticmethod
    def _render_overlay(frame_bgr: np.ndarray, mask: np.ndarray, text: str | None = None) -> np.ndarray:
        overlay = frame_bgr.copy()
        mask_bool = mask > 0
        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        if text:
            cv2.putText(overlay, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return overlay

    @staticmethod
    def _build_prompt_request(prompt: str | None, init_mask_path: str | None) -> dict[str, Any]:
        request: dict[str, Any] = {}
        if prompt:
            request["text"] = prompt

        if init_mask_path:
            init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
            if init_mask is None:
                raise FileNotFoundError(f"Initial mask not found or unreadable: {init_mask_path}")
            ys, xs = np.where(init_mask > 0)
            if ys.size == 0 or xs.size == 0:
                raise ValueError(f"Initial mask is empty: {init_mask_path}")

            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())
            width = max(1, x_max - x_min + 1)
            height = max(1, y_max - y_min + 1)
            center_x = float(xs.mean())
            center_y = float(ys.mean())

            request["bounding_boxes"] = [[x_min, y_min, width, height]]
            request["bounding_box_labels"] = [1]
            request["points"] = [[center_x, center_y]]
            request["point_labels"] = [1]
        return request

    def generate_video_masks(
        self,
        input_video: str,
        output_mask_dir: str,
        prompt: str | None,
        init_mask_path: str | None = None,
        output_overlay_video: str | None = None,
        output_mask_video: str | None = None,
    ) -> dict:
        ensure_dir(output_mask_dir)
        sequence = Path(input_video).stem
        prompt_descriptor = prompt or (Path(init_mask_path).stem if init_mask_path else "init_mask")
        prompt_slug = slugify(prompt_descriptor)
        legacy_frames_dir = os.path.join(output_mask_dir, "_sam3_input_frames")
        if os.path.isdir(legacy_frames_dir):
            shutil.rmtree(legacy_frames_dir)
        temp_root = ensure_dir(os.path.join(os.path.dirname(output_mask_dir), "_sam3_temp"))
        frames_dir = os.path.join(temp_root, f"{Path(input_video).stem}_input_frames")
        if os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir)
        frame_paths = extract_video_frames(input_video, frames_dir)
        if not frame_paths:
            raise ValueError(f"No frames extracted from {input_video}")

        model = self._build_model()
        response = model.handle_request(
            {"type": "start_session", "resource_path": frames_dir}
        )
        session_id = response["session_id"]
        prompt_request = self._build_prompt_request(prompt, init_mask_path)
        if not prompt_request:
            raise ValueError("SAM 3 requires either a text prompt or --init-mask.")
        model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": self.frame_index,
                **prompt_request,
            }
        )
        mask_dict = self._collect_propagation(model, session_id)

        info = get_video_info(input_video)
        overlay_frames = []
        mask_video_frames = []
        for idx, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(frame_path)
            masks = mask_dict.get(idx, {})
            combined = union_masks(*masks.values()) if masks else np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            save_mask(os.path.join(output_mask_dir, f"{idx:05d}.png"), combined)
            if output_overlay_video:
                overlay_text = prompt or (Path(init_mask_path).stem if init_mask_path else None)
                overlay_frames.append(self._render_overlay(frame_bgr, combined, text=overlay_text))
            if output_mask_video:
                mask_video_frames.append(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR))

        if output_overlay_video and overlay_frames:
            write_video(overlay_frames, output_overlay_video, info["fps"] or 25.0)
        if output_mask_video and mask_video_frames:
            write_video(mask_video_frames, output_mask_video, info["fps"] or 25.0)

        try:
            model.handle_request({"type": "close_session", "session_id": session_id})
        except Exception:
            pass
        del mask_dict
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "sequence": sequence,
            "prompt": prompt,
            "prompt_slug": prompt_slug,
            "init_mask_path": init_mask_path,
            "num_frames": len(frame_paths),
            "mask_dir": output_mask_dir,
            "mask_video": output_mask_video,
            "overlay_video": output_overlay_video,
        }
