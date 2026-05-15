"""
ProPainter Video Inpainter
==========================
Wrapper around ProPainter's inference script for video inpainting.
Uses subprocess to call the official inference CLI, avoiding import conflicts.
"""

import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

from src.video_utils import frames_to_video, infer_fps


class ProPainterInpainter:
    """Video inpainting using ProPainter via its CLI."""

    def __init__(self, config: dict):
        self.config = config
        pp_cfg = config["propainter"]
        self.project_root = Path(__file__).resolve().parents[1]
        self.repo_dir = self._resolve_project_path(pp_cfg["repo_dir"])
        weights_dir = pp_cfg.get("weights_dir")
        if weights_dir is None:
            weights_dir = str(Path(pp_cfg["repo_dir"]) / "weights")
        self.weights_dir = self._resolve_project_path(weights_dir)
        self.fp16 = pp_cfg.get("fp16", True)
        self.neighbor_length = pp_cfg.get("neighbor_length", 10)
        self.ref_stride = pp_cfg.get("ref_stride", 10)
        self.subvideo_length = pp_cfg.get("subvideo_length", 80)
        self.output_video_fps = pp_cfg.get("output_video_fps", 30)
        self.gpu_id = config.get("gpu_id", 0)

        inference_script = self.repo_dir / "inference_propainter.py"
        if not inference_script.is_file():
            raise FileNotFoundError(
                f"ProPainter inference script not found: {inference_script}"
            )
        if not self.weights_dir.is_dir():
            raise FileNotFoundError(
                f"ProPainter weights directory not found: {self.weights_dir}"
            )

    def inpaint(self, frames_dir: str, masks_dir: str, output_dir: str, sequence_name: str | None = None):
        os.makedirs(output_dir, exist_ok=True)

        inference_script = self.repo_dir / "inference_propainter.py"
        pp_output = os.path.join(output_dir, "_propainter_tmp")

        cmd = [
            sys.executable, str(inference_script),
            "--video", os.path.abspath(frames_dir),
            "--mask", os.path.abspath(masks_dir),
            "--output", os.path.abspath(pp_output),
            "--neighbor_length", str(self.neighbor_length),
            "--ref_stride", str(self.ref_stride),
            "--subvideo_length", str(self.subvideo_length),
            "--save_frames",
        ]
        if self.fp16:
            cmd.append("--fp16")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        print(f"[ProPainter] Running inpainting ...")
        print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, cwd=str(self.repo_dir), env=env,
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            print("[ProPainter] STDERR:\n" + result.stderr)
            raise RuntimeError(
                f"ProPainter failed with return code {result.returncode}"
            )

        self._collect_results(pp_output, output_dir, frames_dir)

        if os.path.isdir(pp_output):
            shutil.rmtree(pp_output)

        fps = infer_fps(self.project_root, sequence_name, self.output_video_fps)
        video_path = Path(output_dir).parent / "inpainted.mp4"
        frames_to_video(output_dir, video_path, fps=fps)

        print(f"[ProPainter] Inpainted frames saved to {output_dir}")
        print(f"[ProPainter] Video saved to {video_path}")

    def _resolve_project_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _collect_results(self, pp_output: str, target_dir: str, frames_dir: str):
        result_frames = []
        for root, dirs, files in os.walk(pp_output):
            for f in sorted(files):
                if f.endswith((".png", ".jpg")) and "mask" not in f.lower():
                    result_frames.append(os.path.join(root, f))

        if not result_frames:
            print(f"[ProPainter] Warning: no output frames found in {pp_output}")
            orig_frames = sorted(
                glob.glob(os.path.join(frames_dir, "*.jpg"))
                + glob.glob(os.path.join(frames_dir, "*.png"))
            )
            for f in orig_frames:
                fname = os.path.splitext(os.path.basename(f))[0] + ".png"
                shutil.copy2(f, os.path.join(target_dir, fname))
            return

        orig_frames = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg"))
            + glob.glob(os.path.join(frames_dir, "*.png"))
        )

        if len(result_frames) == len(orig_frames):
            for src, orig in zip(result_frames, orig_frames):
                fname = os.path.splitext(os.path.basename(orig))[0] + ".png"
                shutil.copy2(src, os.path.join(target_dir, fname))
        else:
            for i, src in enumerate(result_frames):
                if i < len(orig_frames):
                    fname = os.path.splitext(
                        os.path.basename(orig_frames[i])
                    )[0] + ".png"
                else:
                    fname = f"{i:05d}.png"
                shutil.copy2(src, os.path.join(target_dir, fname))
