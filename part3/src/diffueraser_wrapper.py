"""DiffuEraser subprocess wrapper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .io_utils import ensure_dir
from .mask_utils import mask_dir_to_video
from .video_utils import get_video_info


class DiffuEraserRunner:
    def __init__(self, diffueraser_dir: str, env_name: str | None = None, python_bin: str | None = None):
        self.repo_dir = os.path.abspath(diffueraser_dir)
        self.env_name = env_name
        self.python_bin = python_bin or sys.executable
        self.script = os.path.join(self.repo_dir, "run_diffueraser.py")
        if not os.path.isfile(self.script):
            raise FileNotFoundError(f"DiffuEraser script not found: {self.script}")

    def _validate_weights(self, base_model_path: str, vae_path: str, diffueraser_path: str, propainter_model_dir: str) -> None:
        missing = [path for path in [base_model_path, vae_path, diffueraser_path, propainter_model_dir] if not path or not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "DiffuEraser weights are missing. Set base_model_path, vae_path, "
                "diffueraser_path, and propainter_model_dir in the config before running inference."
            )

    def run(
        self,
        input_video: str,
        mask_video_or_dir: str,
        output_dir: str,
        extra_args: list[str] | None = None,
    ) -> str:
        ensure_dir(output_dir)
        log_dir = ensure_dir(os.path.join(output_dir, "_logs"))
        stdout_path = os.path.join(log_dir, "diffueraser_stdout.log")
        stderr_path = os.path.join(log_dir, "diffueraser_stderr.log")

        mask_video = mask_video_or_dir
        if os.path.isdir(mask_video_or_dir):
            info = get_video_info(input_video)
            mask_video = os.path.join(output_dir, "_mask_video.mp4")
            mask_dir_to_video(mask_video_or_dir, mask_video, info["fps"] or 25.0)

        cfg = extra_args or []
        cmd = [self.python_bin, self.script] + cfg
        with open(stdout_path, "w", encoding="utf-8") as stdout_file, open(stderr_path, "w", encoding="utf-8") as stderr_file:
            result = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"DiffuEraser failed with code {result.returncode}. See {stdout_path} and {stderr_path}."
            )
        out_path = os.path.join(output_dir, "diffueraser_result.mp4")
        if not os.path.isfile(out_path):
            raise RuntimeError(f"DiffuEraser did not produce {out_path}")
        return out_path

    def build_args(self, input_video: str, input_mask: str, output_dir: str, config: dict) -> list[str]:
        self._validate_weights(
            config.get("base_model_path", ""),
            config.get("vae_path", ""),
            config.get("diffueraser_path", ""),
            config.get("propainter_model_dir", ""),
        )
        return [
            "--input_video", os.path.abspath(input_video),
            "--input_mask", os.path.abspath(input_mask),
            "--video_length", str(config.get("video_length", 17)),
            "--mask_dilation_iter", str(config.get("mask_dilation_iter", 8)),
            "--max_img_size", str(config.get("max_img_size", 960)),
            "--save_path", os.path.abspath(output_dir),
            "--ref_stride", str(config.get("ref_stride", 10)),
            "--neighbor_length", str(config.get("neighbor_length", 10)),
            "--subvideo_length", str(config.get("subvideo_length", 50)),
            "--base_model_path", os.path.abspath(config.get("base_model_path", "")),
            "--vae_path", os.path.abspath(config.get("vae_path", "")),
            "--diffueraser_path", os.path.abspath(config.get("diffueraser_path", "")),
            "--propainter_model_dir", os.path.abspath(config.get("propainter_model_dir", "")),
        ]
