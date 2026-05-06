"""SAM 3 mask generation wrapper."""

from __future__ import annotations

import gc
import getpass
import importlib.util
import inspect
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
        # The tracker path requires cached backbone features for the annotated frame.
        # Build them through the video inference model first, then derive the tracker state.
        video_model._prepare_backbone_feats(
            video_inference_state,
            frame_idx=frame_index,
            reverse=False,
        )
        if hasattr(video_model, "_init_new_tracker_state"):
            inference_state = video_model._init_new_tracker_state(video_inference_state)
        else:
            tracker_init_kwargs = dict(
                cached_features=video_inference_state["feature_cache"],
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

        init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
        if init_mask is None:
            raise FileNotFoundError(f"Initial mask not found or unreadable: {init_mask_path}")
        init_mask_tensor = torch.from_numpy((init_mask > 0).astype(np.uint8))

        if hasattr(tracker, "add_new_masks"):
            tracker.add_new_masks(
                inference_state=inference_state,
                frame_idx=frame_index,
                obj_ids=[1],
                masks=init_mask_tensor.unsqueeze(0),
                add_mask_to_memory=True,
            )
        else:
            tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_index,
                obj_id=1,
                mask=init_mask_tensor,
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
        for frame_idx, obj_ids, _low_res_masks, video_res_masks, _obj_scores in tracker.propagate_in_video(
            **filtered_propagate_kwargs,
        ):
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.detach().cpu().numpy()
            if isinstance(video_res_masks, torch.Tensor):
                video_res_masks = video_res_masks.detach().cpu().numpy()
            frame_masks: dict[int, np.ndarray] = {}
            for idx, object_id in enumerate(obj_ids):
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
    def _build_prompt_request(prompt: str | None, init_mask_path: str | None) -> dict[str, Any]:
        request: dict[str, Any] = {}

        if init_mask_path:
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
        if init_mask_path:
            response = model.handle_request(
                {"type": "start_session", "resource_path": frames_dir}
            )
            session_id = response["session_id"]
            session_state = model._all_inference_states[session_id]["state"]
            init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
            if init_mask is None:
                raise FileNotFoundError(f"Initial mask not found or unreadable: {init_mask_path}")
            init_mask_tensor = torch.from_numpy((init_mask > 0).astype(np.uint8))
            model.model.add_sam2_new_mask(
                session_state,
                frame_idx=self.frame_index,
                obj_id=1,
                mask=init_mask_tensor,
            )
            mask_dict = self._collect_propagation(
                model,
                session_id,
                start_frame_index=self.frame_index,
                propagation_direction="forward",
            )
            initial_mask = (init_mask > 0).astype(np.uint8) * 255
            mask_dict[self.frame_index] = {1: initial_mask}
        else:
            prompt_request = self._build_prompt_request(prompt, init_mask_path)
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
            "mask_video": output_mask_video,
            "overlay_video": output_overlay_video,
        }
