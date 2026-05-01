"""Part 3 execution pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .io_utils import ensure_dir, resolve_path, slugify
from .mask_utils import mask_dir_to_video
from .side_effect_mask import build_side_effect_masks_for_dir
from .video_utils import extract_video_frames, frames_to_video, get_video_info


class Part3Pipeline:
    def __init__(self, config: dict, part3_dir: str):
        self.config = config
        self.part3_dir = os.path.abspath(part3_dir)
        self.output_root = ensure_dir(resolve_path(self.part3_dir, config.get("output_root", "outputs")))

    def _sam3_generator(self) -> SAM3MaskGenerator:
        from .sam3_wrapper import SAM3MaskGenerator

        sam3_cfg = self.config["sam3"]
        return SAM3MaskGenerator(
            sam3_dir=resolve_path(self.part3_dir, sam3_cfg["repo_dir"]),
            checkpoint=resolve_path(self.part3_dir, sam3_cfg["checkpoint"]) if sam3_cfg.get("checkpoint") else None,
            device=self.config.get("device", "cuda"),
            version=sam3_cfg.get("version", "sam3"),
            frame_index=sam3_cfg.get("frame_index", 0),
            compile=sam3_cfg.get("compile", False),
            async_loading_frames=sam3_cfg.get("async_loading_frames", False),
            use_fa3=sam3_cfg.get("use_fa3"),
            use_rope_real=sam3_cfg.get("use_rope_real"),
        )

    def _propainter_runner(self) -> ProPainterRunner:
        from .propainter_wrapper import ProPainterRunner

        cfg = self.config["propainter"]
        return ProPainterRunner(
            repo_dir=resolve_path(self.part3_dir, cfg["repo_dir"]),
            gpu_id=self.config.get("gpu_id", 0),
            fp16=cfg.get("fp16", True),
            neighbor_length=cfg.get("neighbor_length", 10),
            ref_stride=cfg.get("ref_stride", 10),
            subvideo_length=cfg.get("subvideo_length", 80),
        )

    def _diffueraser_runner(self) -> DiffuEraserRunner:
        from .diffueraser_wrapper import DiffuEraserRunner

        cfg = self.config["diffueraser"]
        return DiffuEraserRunner(
            resolve_path(self.part3_dir, cfg["repo_dir"]),
            python_bin=resolve_path(self.part3_dir, cfg["python_bin"]) if cfg.get("python_bin") else None,
        )

    def _rose_runner(self) -> ROSERunner:
        from .rose_wrapper import ROSERunner

        cfg = self.config["rose"]
        return ROSERunner(
            resolve_path(self.part3_dir, cfg["repo_dir"]),
            model_root=resolve_path(self.part3_dir, cfg.get("model_root", "")) if cfg.get("model_root") else None,
            transformer_root=resolve_path(self.part3_dir, cfg.get("transformer_root", "")) if cfg.get("transformer_root") else None,
            python_bin=resolve_path(self.part3_dir, cfg["python_bin"]) if cfg.get("python_bin") else None,
        )

    def run(
        self,
        method: str,
        sequence: str,
        input_video: str,
        prompt: str | None,
        init_mask_path: str | None = None,
        allow_existing_masks: bool = False,
        existing_mask_dir: str | None = None,
    ) -> dict:
        sequence_slug = slugify(sequence)
        method_slug = slugify(method)
        input_stem_slug = slugify(Path(input_video).stem)

        masks_root = ensure_dir(os.path.join(self.output_root, "masks", sequence_slug, method_slug))
        videos_root = ensure_dir(os.path.join(self.output_root, "videos", sequence_slug, method_slug))
        logs_root = ensure_dir(os.path.join(self.output_root, "logs", sequence_slug, method_slug))
        frames_root = ensure_dir(os.path.join(self.output_root, "frames", sequence_slug, input_stem_slug))
        if not os.listdir(frames_root):
            extract_video_frames(input_video, frames_root)

        object_mask_dir = os.path.join(masks_root, "object_mask")
        if existing_mask_dir:
            object_mask_dir = os.path.abspath(existing_mask_dir)
        elif "sam3" in method:
            sam3_output = self._sam3_generator().generate_video_masks(
                input_video=input_video,
                output_mask_dir=ensure_dir(object_mask_dir),
                prompt=prompt,
                init_mask_path=init_mask_path,
                output_overlay_video=os.path.join(videos_root, "sam3_overlay.mp4"),
                output_mask_video=os.path.join(videos_root, "sam3_mask.mp4"),
            )
            object_mask_dir = sam3_output["mask_dir"]
        elif not allow_existing_masks:
            raise RuntimeError(
                "This method requires existing masks or SAM3 masks. Use --allow-existing-masks "
                "with --existing-mask-dir when SAM3 is unavailable."
            )

        final_mask_dir = object_mask_dir
        debug_video = None
        if "side_effect" in method:
            built_dirs = build_side_effect_masks_for_dir(
                frames_dir=frames_root,
                object_mask_dir=object_mask_dir,
                output_root=masks_root,
                config=self.config,
            )
            final_mask_dir = built_dirs["final_side_effect_mask"]
            from .visualization import create_mask_debug_video
            debug_video = create_mask_debug_video(
                frames_root,
                built_dirs["object_mask"],
                built_dirs["shadow_mask"],
                built_dirs["final_side_effect_mask"],
                os.path.join(videos_root, "mask_debug_grid.mp4"),
                fps=max(get_video_info(input_video)["fps"], 1.0),
            )

        result_video = None
        if method.endswith("propainter"):
            output_frames_dir = ensure_dir(os.path.join(videos_root, "frames"))
            self._propainter_runner().run(frames_root, final_mask_dir, output_frames_dir)
            result_video = frames_to_video(output_frames_dir, os.path.join(videos_root, "output.mp4"), get_video_info(input_video)["fps"] or 25.0)
        elif "diffueraser" in method:
            runner = self._diffueraser_runner()
            mask_video_path = os.path.join(videos_root, "mask.mp4")
            mask_dir_to_video(final_mask_dir, mask_video_path, get_video_info(input_video)["fps"] or 25.0)
            args = runner.build_args(input_video, mask_video_path, videos_root, self.config["diffueraser"])
            result_video = runner.run(input_video, mask_video_path, videos_root, extra_args=args)
        elif "rose" in method:
            runner = self._rose_runner()
            result_video = runner.run(
                input_video=input_video,
                mask_video_or_dir=final_mask_dir,
                output_dir=videos_root,
                prompt=prompt,
                video_length=self.config["rose"].get("video_length", 17),
                sample_size=tuple(self.config["rose"].get("sample_size", [480, 720])),
            )

        if result_video:
            canonical_output = os.path.join(videos_root, "output.mp4")
            if os.path.abspath(result_video) != os.path.abspath(canonical_output):
                shutil.copy2(result_video, canonical_output)
                result_video = canonical_output

        return {
            "sequence": sequence,
            "method": method,
            "input_video": input_video,
            "object_mask_dir": object_mask_dir,
            "final_mask_dir": final_mask_dir,
            "result_video": result_video,
            "debug_video": debug_video,
            "videos_root": videos_root,
            "logs_root": logs_root,
        }
