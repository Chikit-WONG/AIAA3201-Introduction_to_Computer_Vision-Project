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
        logs_dir = os.path.join(output_dir, "_logs")
        os.makedirs(logs_dir, exist_ok=True)

        retry_lengths = []
        for candidate in (
            self.subvideo_length,
            min(self.subvideo_length, 50),
            min(self.subvideo_length, 40),
            min(self.subvideo_length, 32),
            min(self.subvideo_length, 24),
            min(self.subvideo_length, 20),
            min(self.subvideo_length, 16),
            min(self.subvideo_length, 12),
            min(self.subvideo_length, 8),
        ):
            if candidate not in retry_lengths:
                retry_lengths.append(candidate)

        final_stdout_path = os.path.join(logs_dir, "propainter_stdout.log")
        final_stderr_path = os.path.join(logs_dir, "propainter_stderr.log")
        last_attempt = None
        for attempt_index, subvideo_length in enumerate(retry_lengths, start=1):
            tmp_output = os.path.join(output_dir, f"_propainter_tmp_attempt{attempt_index}")
            stdout_path = os.path.join(logs_dir, f"propainter_stdout_attempt{attempt_index}.log")
            stderr_path = os.path.join(logs_dir, f"propainter_stderr_attempt{attempt_index}.log")
            shutil.rmtree(tmp_output, ignore_errors=True)
            cmd = [
                sys.executable, self.script,
                "--video", os.path.abspath(frames_dir),
                "--mask", os.path.abspath(masks_dir),
                "--output", os.path.abspath(tmp_output),
                "--neighbor_length", str(self.neighbor_length),
                "--ref_stride", str(self.ref_stride),
                "--subvideo_length", str(subvideo_length),
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
            last_attempt = (result.returncode, stdout_path, stderr_path, tmp_output)
            shutil.copy2(stdout_path, final_stdout_path)
            shutil.copy2(stderr_path, final_stderr_path)
            if result.returncode == 0:
                break
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                stderr_text = f.read()
            oom = "outofmemoryerror" in stderr_text.lower() or "cuda out of memory" in stderr_text.lower()
            if not oom or attempt_index == len(retry_lengths):
                raise RuntimeError(
                    f"ProPainter failed with code {result.returncode}. "
                    f"See {final_stdout_path} and {final_stderr_path}."
                )
        else:
            raise RuntimeError(
                f"ProPainter failed with code {last_attempt[0] if last_attempt else 'unknown'}. "
                f"See {final_stdout_path} and {final_stderr_path}."
            )

        output_frames = []
        assert last_attempt is not None
        successful_tmp_output = last_attempt[3]
        for root, _dirs, files in os.walk(successful_tmp_output):
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

        for attempt_index in range(1, len(retry_lengths) + 1):
            shutil.rmtree(os.path.join(output_dir, f"_propainter_tmp_attempt{attempt_index}"), ignore_errors=True)
        return output_dir
