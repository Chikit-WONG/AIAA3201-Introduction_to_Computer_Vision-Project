"""SAM 3 mask generation wrapper."""

from __future__ import annotations

import gc
import getpass
import importlib.util
import inspect
import json
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
        self.requested_sam3_dir = os.path.abspath(sam3_dir)
        self.sam3_dir = self._resolve_sam3_dir(self.requested_sam3_dir)
        self.checkpoint = checkpoint or None
        self.device = device
        self.version = version
        self.frame_index = frame_index
        self.compile = compile
        self.async_loading_frames = async_loading_frames
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real

    def _resolve_checkpoint_path(self) -> str | None:
        if not self.checkpoint:
            return None

        checkpoint_path = os.path.abspath(self.checkpoint)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path

        candidate_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
        if not candidate_dir or not os.path.isdir(candidate_dir):
            return None

        preferred_filenames = []
        if self.version == "sam3.1":
            preferred_filenames.extend(["sam3.1_multiplex.pt", "sam3.1.pt"])
        else:
            preferred_filenames.extend(["sam3.pt", "sam3_multiplex.pt"])

        for filename in preferred_filenames:
            candidate = os.path.join(candidate_dir, filename)
            if os.path.isfile(candidate):
                return candidate

        weight_candidates = sorted(
            path for pattern in ("*.pt", "*.pth") for path in Path(candidate_dir).glob(pattern)
        )
        if len(weight_candidates) == 1:
            return str(weight_candidates[0])
        if weight_candidates:
            return str(weight_candidates[0])
        return None

    @staticmethod
    def _resolve_sam3_dir(requested_dir: str) -> str | None:
        env_repo_dir = os.environ.get("SAM3_REPO_DIR")
        candidate_dirs = []
        if env_repo_dir:
            candidate_dirs.append(os.path.abspath(env_repo_dir))
        candidate_dirs.append(requested_dir)

        for candidate in candidate_dirs:
            if os.path.isdir(candidate):
                return candidate

        if importlib.util.find_spec("sam3") is not None:
            return None

        searched_dirs = ", ".join(candidate_dirs)
        raise FileNotFoundError(
            "SAM 3 repository not found. Searched: "
            f"{searched_dirs}. "
            "Fix by running `bash scripts/setup_external_repos.sh` from the `part3` directory, "
            "or set `SAM3_REPO_DIR` to an existing `sam3` checkout."
        )

    def _build_model(self):
        if self.sam3_dir and self.sam3_dir not in sys.path:
            sys.path.insert(0, self.sam3_dir)
        username = getpass.getuser()
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_cache_{username}")
        from sam3 import build_sam3_predictor

        build_kwargs = {
            "version": self.version,
            "compile": self.compile,
            "async_loading_frames": self.async_loading_frames,
        }
        resolved_checkpoint = None
        if self.use_fa3 is not None:
            build_kwargs["use_fa3"] = self.use_fa3
        elif self.version == "sam3.1":
            build_kwargs["use_fa3"] = False
        if self.use_rope_real is not None:
            build_kwargs["use_rope_real"] = self.use_rope_real
        if self.checkpoint:
            resolved_checkpoint = self._resolve_checkpoint_path()
            if resolved_checkpoint:
                build_kwargs["checkpoint_path"] = resolved_checkpoint
            else:
                print(
                    f"SAM 3 checkpoint bundle not found or empty at {self.checkpoint}; "
                    "falling back to automatic download."
                )

        try:
            return build_sam3_predictor(**build_kwargs)
        except Exception as exc:
            error_message = str(exc)
            if "No CUDA GPUs are available" in error_message:
                raise RuntimeError(
                    "SAM 3 requires a CUDA-capable PyTorch runtime, but no GPU is visible. "
                    "Run this on a GPU node or check your CUDA/PyTorch environment."
                ) from exc
            if self.checkpoint and not resolved_checkpoint:
                raise RuntimeError(
                    "SAM 3 checkpoint is unavailable. Please request access and set "
                    f"a valid checkpoint path in the config. Current setting: {self.checkpoint}"
                ) from exc
            raise RuntimeError(f"Failed to build SAM 3 predictor: {error_message}") from exc

    @staticmethod
    def _collect_propagation(
        model,
        session_id: str,
        start_frame_index: int | None = None,
        propagation_direction: str = "both",
    ) -> dict[int, dict[int, np.ndarray]]:
        mask_dict: dict[int, dict[int, np.ndarray]] = {}
        for response in model.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": start_frame_index,
                "propagation_direction": propagation_direction,
            }
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
    def _extract_per_object_masks(annotation_path: str) -> list[tuple[int, np.ndarray]]:
        """Read a DAVIS-style annotation and return one binary mask per object ID.

        Returns a list of (obj_id, binary_mask) tuples, sorted by obj_id.
        Falls back to single-object (id=1) if the annotation is already binary.
        """
        ann = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        if ann is None:
            raise FileNotFoundError(f"Annotation not found or unreadable: {annotation_path}")
        unique_ids = sorted(int(v) for v in set(ann.flatten()) if v != 0)
        if not unique_ids:
            raise ValueError(f"Annotation mask is empty: {annotation_path}")
        return [(obj_id, (ann == obj_id).astype(np.uint8)) for obj_id in unique_ids]

    @staticmethod
    def _collect_tracker_propagation(
        model,
        frames_dir: str,
        init_mask_path: str,
        frame_index: int,
        async_loading_frames: bool,
    ) -> dict[int, dict[int, np.ndarray]]:
        video_model = model.model
        tracker = video_model.tracker
        from sam3.model.io_utils import load_video_frames

        video_inference_state = video_model.init_state(
            resource_path=frames_dir,
            async_loading_frames=async_loading_frames,
        )
        # The tracker path reads cached backbone features instead of owning a
        # standalone backbone. Cache all frames before handing state to tracker.
        num_frames = int(video_inference_state["num_frames"])
        tracker_feature_cache = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            for idx in range(num_frames):
                video_model._prepare_backbone_feats(
                    video_inference_state,
                    frame_idx=idx,
                    reverse=False,
                )
                if idx in video_inference_state["feature_cache"]:
                    tracker_feature_cache[idx] = video_inference_state["feature_cache"][idx]
        missing_features = [idx for idx in range(num_frames) if idx not in tracker_feature_cache]
        if missing_features:
            raise RuntimeError(f"Missing tracker features for frames: {missing_features[:10]}")
        tracker_init_kwargs = dict(
            cached_features=tracker_feature_cache,
            video_height=video_inference_state["orig_height"],
            video_width=video_inference_state["orig_width"],
            num_frames=video_inference_state["num_frames"],
            offload_state_to_cpu=video_inference_state.get("offload_state_to_cpu", False),
        )
        valid_params = set(inspect.signature(tracker.init_state).parameters.keys())
        inference_state = tracker.init_state(
            **{key: value for key, value in tracker_init_kwargs.items() if key in valid_params}
        )
        if "images" not in inference_state:
            tracker_images, _video_height, _video_width = load_video_frames(
                video_path=frames_dir,
                image_size=tracker.image_size,
                offload_video_to_cpu=inference_state.get("offload_video_to_cpu", False),
                async_loading_frames=async_loading_frames,
            )
            inference_state["images"] = tracker_images

        # --- Multi-object initialization (per-object DAVIS GT masks) ---
        # Extract one binary mask per DAVIS object ID from the annotation.
        # This mirrors how SAM2 initializes tracking (see part2/src/mask_sam2.py)
        # and is the primary reason SAM3 had lower JR: the old code collapsed all
        # objects into a single union mask (obj_id=1), losing per-object identity.
        per_object = SAM3MaskGenerator._extract_per_object_masks(init_mask_path)
        obj_ids = [oid for oid, _ in per_object]
        masks_np = np.stack([m for _, m in per_object], axis=0)  # (N, H, W), uint8 {0,1}
        masks_tensor = torch.from_numpy(masks_np)
        print(f"[SAM3] Multi-object init: {len(obj_ids)} object(s) with IDs {obj_ids}")

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            if hasattr(tracker, "add_new_masks"):
                tracker.add_new_masks(
                    inference_state=inference_state,
                    frame_idx=frame_index,
                    obj_ids=obj_ids,
                    masks=masks_tensor,
                    add_mask_to_memory=True,
                )
            else:
                # Fallback: insert objects one at a time
                for obj_id, mask_np in per_object:
                    tracker.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_index,
                        obj_id=obj_id,
                        mask=torch.from_numpy(mask_np),
                        add_mask_to_memory=True,
                    )

        propagate_kwargs = dict(
            inference_state=inference_state,
            start_frame_idx=frame_index,
            max_frame_num_to_track=None,
            reverse=False,
            propagate_preflight=True,
        )
        valid_params = set(inspect.signature(tracker.propagate_in_video).parameters.keys())
        filtered_propagate_kwargs = {
            key: value for key, value in propagate_kwargs.items() if key in valid_params
        }

        mask_dict: dict[int, dict[int, np.ndarray]] = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            for frame_idx, obj_ids_out, _low_res_masks, video_res_masks, _obj_scores in tracker.propagate_in_video(
                **filtered_propagate_kwargs,
            ):
                if isinstance(obj_ids_out, torch.Tensor):
                    obj_ids_out = obj_ids_out.detach().cpu().numpy()
                if isinstance(video_res_masks, torch.Tensor):
                    video_res_masks = video_res_masks.detach().cpu().numpy()
                frame_masks: dict[int, np.ndarray] = {}
                for idx, object_id in enumerate(obj_ids_out):
                    mask = video_res_masks[idx]
                    if mask.ndim == 3:
                        mask = mask[0]
                    frame_masks[int(object_id)] = (mask > 0).astype(np.uint8) * 255
                mask_dict[int(frame_idx)] = frame_masks

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
    def _resize_mask_to_frame(mask: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        frame_h, frame_w = frame_shape
        if mask.shape[:2] == (frame_h, frame_w):
            return (mask > 0).astype(np.uint8) * 255
        return cv2.resize(
            (mask > 0).astype(np.uint8) * 255,
            (frame_w, frame_h),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def _detect_area_collapse(
        frame_stats: list[dict[str, Any]],
        init_area: int,
        collapse_threshold: float = 0.1,
    ) -> list[int]:
        """Return frame indices where mask area dropped below collapse_threshold * init_area."""
        if init_area == 0:
            return []
        threshold = init_area * collapse_threshold
        return [
            item["frame_index"]
            for item in frame_stats
            if item["nonzero_pixels"] < threshold and item["frame_index"] > 0
        ]

    @staticmethod
    def _write_mask_stats(stats_path: str, frame_stats: list[dict[str, Any]], init_mask_path: str | None) -> dict[str, Any]:
        total_frames = len(frame_stats)
        zero_frames = sum(1 for item in frame_stats if item["nonzero_pixels"] == 0)
        mean_nonzero = (
            float(sum(item["nonzero_pixels"] for item in frame_stats)) / float(total_frames)
            if total_frames
            else 0.0
        )
        summary = {
            "num_frames": total_frames,
            "zero_frames": zero_frames,
            "zero_frame_ratio": float(zero_frames) / float(total_frames) if total_frames else 1.0,
            "mean_nonzero_pixels": mean_nonzero,
            "init_mask_path": init_mask_path,
            "frames": frame_stats,
        }
        ensure_dir(os.path.dirname(stats_path))
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    @staticmethod
    def _validate_mask_stats(summary: dict[str, Any], init_mask_path: str | None, stats_path: str) -> None:
        num_frames = int(summary["num_frames"])
        zero_frames = int(summary["zero_frames"])
        zero_ratio = float(summary["zero_frame_ratio"])
        frames = summary.get("frames", [])
        first_nonzero = int(frames[0]["nonzero_pixels"]) if frames else 0

        if num_frames == 0:
            raise RuntimeError(f"No masks were generated. See {stats_path}")
        if zero_frames == num_frames:
            raise RuntimeError(f"SAM 3 generated all-empty masks. See {stats_path}")
        if init_mask_path and first_nonzero == 0:
            raise RuntimeError(f"SAM 3 generated an empty first-frame mask from --init-mask. See {stats_path}")
        if zero_ratio > 0.5:
            raise RuntimeError(
                f"SAM 3 generated too many empty masks ({zero_frames}/{num_frames}). See {stats_path}"
            )

    @staticmethod
    def _build_prompt_request(
        prompt: str | None,
        init_mask_path: str | None,
        frame_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {}

        if init_mask_path:
            init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
            if init_mask is None:
                raise FileNotFoundError(f"Initial mask not found or unreadable: {init_mask_path}")
            if frame_shape is None:
                raise ValueError("frame_shape is required when building prompts from --init-mask.")
            ys, xs = np.where(init_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                raise ValueError(f"Initial mask is empty: {init_mask_path}")
            x_min = int(xs.min())
            y_min = int(ys.min())
            x_max = int(xs.max())
            y_max = int(ys.max())
            width = max(1, x_max - x_min + 1)
            height = max(1, y_max - y_min + 1)
            frame_h, frame_w = frame_shape
            request["bounding_boxes"] = [[
                float(x_min) / float(frame_w),
                float(y_min) / float(frame_h),
                float(width) / float(frame_w),
                float(height) / float(frame_h),
            ]]
            request["bounding_box_labels"] = [1]
            request["rel_coordinates"] = True
            return request

        if prompt:
            request["text"] = prompt
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
        session_id = None
        frame_shape = cv2.imread(frame_paths[self.frame_index]).shape[:2]
        if init_mask_path:
            # For DAVIS-style evaluation, use the provided first-frame annotation mask
            # with per-object initialization. Each DAVIS object ID gets its own tracked
            # object, matching the SAM2 multi-object prompt strategy and improving JR.
            mask_dict = self._collect_tracker_propagation(
                model,
                frames_dir=frames_dir,
                init_mask_path=init_mask_path,
                frame_index=self.frame_index,
                async_loading_frames=self.async_loading_frames,
            )
            # Seed the init frame with the union of all object masks so frame 0 is always correct
            per_object = self._extract_per_object_masks(init_mask_path)
            init_union = np.zeros(frame_shape, dtype=np.uint8)
            for obj_id, obj_mask in per_object:
                resized = self._resize_mask_to_frame(obj_mask * 255, frame_shape)
                init_union = np.clip(init_union + (resized > 0).astype(np.uint8) * 255, 0, 255).astype(np.uint8)
            init_frame_masks = mask_dict.setdefault(self.frame_index, {})
            # Re-insert each object mask for the init frame so they are available for union
            for obj_id, obj_mask in per_object:
                init_frame_masks[obj_id] = self._resize_mask_to_frame(obj_mask * 255, frame_shape)
        else:
            prompt_request = self._build_prompt_request(
                prompt,
                init_mask_path,
                frame_shape=frame_shape,
            )
            if not prompt_request:
                raise ValueError("SAM 3 requires either a text prompt or --init-mask.")
            response = model.handle_request(
                {"type": "start_session", "resource_path": frames_dir}
            )
            session_id = response["session_id"]
            model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": self.frame_index,
                    **prompt_request,
                }
            )
            mask_dict = self._collect_propagation(
                model,
                session_id,
                start_frame_index=self.frame_index,
                propagation_direction="forward",
            )

        info = get_video_info(input_video)
        overlay_frames = []
        mask_video_frames = []
        frame_stats = []
        for idx, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(frame_path)
            masks = mask_dict.get(idx, {})
            combined = union_masks(*masks.values()) if masks else np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            mask_path = save_mask(os.path.join(output_mask_dir, f"{idx:05d}.png"), combined)
            nonzero_pixels = int(np.count_nonzero(combined))
            frame_stats.append(
                {
                    "frame_index": idx,
                    "mask_path": mask_path,
                    "nonzero_pixels": nonzero_pixels,
                    "coverage": float(nonzero_pixels) / float(combined.shape[0] * combined.shape[1]),
                }
            )
            if output_overlay_video:
                overlay_text = prompt or (Path(init_mask_path).stem if init_mask_path else None)
                overlay_frames.append(self._render_overlay(frame_bgr, combined, text=overlay_text))
            if output_mask_video:
                mask_video_frames.append(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR))

        stats_path = os.path.join(os.path.dirname(output_mask_dir), "mask_stats.json")
        stats_summary = self._write_mask_stats(stats_path, frame_stats, init_mask_path)

        # Detect area-collapse frames and record in stats (for diagnostic purposes)
        if frame_stats and init_mask_path:
            init_area = frame_stats[0]["nonzero_pixels"]
            collapse_frames = self._detect_area_collapse(frame_stats, init_area)
            if collapse_frames:
                print(
                    f"[SAM3] Warning: area collapse detected at {len(collapse_frames)} frame(s): "
                    f"{collapse_frames[:5]}{'...' if len(collapse_frames) > 5 else ''}"
                )
                stats_summary["area_collapse_frames"] = collapse_frames
                # Re-write stats with collapse info
                ensure_dir(os.path.dirname(stats_path))
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats_summary, f, indent=2)

        if output_overlay_video and overlay_frames:
            write_video(overlay_frames, output_overlay_video, info["fps"] or 25.0)
        if output_mask_video and mask_video_frames:
            write_video(mask_video_frames, output_mask_video, info["fps"] or 25.0)

        self._validate_mask_stats(stats_summary, init_mask_path, stats_path)

        if session_id is not None:
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
            "mask_stats": stats_path,
            "mask_video": output_mask_video,
            "overlay_video": output_overlay_video,
        }
