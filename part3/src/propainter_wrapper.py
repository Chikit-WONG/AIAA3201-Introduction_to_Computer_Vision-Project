"""Local ProPainter wrapper reused for Part 3."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys


class ProPainterRunner:
    def __init__(self, repo_dir: str, gpu_id: int = 0, fp16: bool = True, neighbor_length: int = 10,
                 ref_stride: int = 10, subvideo_length: int = 80):
        self.repo_dir = os.path.abspath(repo_dir)
        self.gpu_id = gpu_id
        self.fp16 = fp16
        self.neighbor_length = neighbor_length
        self.ref_stride = ref_stride
        self.subvideo_length = subvideo_length
        self.script = os.path.join(self.repo_dir, "inference_propainter.py")
        if not os.path.isfile(self.script):
            raise FileNotFoundError(f"ProPainter inference script not found: {self.script}")

    def run(self, frames_dir: str, masks_dir: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        tmp_output = os.path.join(output_dir, "_propainter_tmp")
        logs_dir = os.path.join(output_dir, "_logs")
        os.makedirs(logs_dir, exist_ok=True)
        stdout_path = os.path.join(logs_dir, "propainter_stdout.log")
        stderr_path = os.path.join(logs_dir, "propainter_stderr.log")
        cmd = [
            sys.executable, self.script,
            "--video", os.path.abspath(frames_dir),
            "--mask", os.path.abspath(masks_dir),
            "--output", os.path.abspath(tmp_output),
            "--neighbor_length", str(self.neighbor_length),
            "--ref_stride", str(self.ref_stride),
            "--subvideo_length", str(self.subvideo_length),
            "--save_frames",
        ]
        if self.fp16:
            cmd.append("--fp16")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        with open(stdout_path, "w", encoding="utf-8") as stdout_file, open(stderr_path, "w", encoding="utf-8") as stderr_file:
            result = subprocess.run(
                cmd,
                cwd=self.repo_dir,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"ProPainter failed with code {result.returncode}. "
                f"See {stdout_path} and {stderr_path}."
            )

        output_frames = []
        for root, _dirs, files in os.walk(tmp_output):
            for name in sorted(files):
                if name.endswith((".png", ".jpg")) and "mask" not in name.lower():
                    output_frames.append(os.path.join(root, name))

        src_frames = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg"))
            + glob.glob(os.path.join(frames_dir, "*.png"))
        )
        if not output_frames:
            raise RuntimeError(f"ProPainter produced no result frames under {tmp_output}")

        for index, src in enumerate(output_frames):
            if index < len(src_frames):
                basename = os.path.splitext(os.path.basename(src_frames[index]))[0] + ".png"
            else:
                basename = f"{index:05d}.png"
            shutil.copy2(src, os.path.join(output_dir, basename))

        shutil.rmtree(tmp_output, ignore_errors=True)
        return output_dir
