"""ROSE subprocess wrapper."""

from __future__ import annotations

import os
import subprocess
import sys

from .io_utils import ensure_dir
from .video_utils import (
    get_video_info,
    pad_video_to_16n_plus_1,
    read_video_frames,
    write_video,
)
from .mask_utils import mask_dir_to_video


class ROSERunner:
    def __init__(self, rose_dir: str, env_name: str | None = None, model_root: str | None = None,
                 transformer_root: str | None = None, python_bin: str | None = None):
        self.repo_dir = os.path.abspath(rose_dir)
        self.env_name = env_name
        self.model_root = os.path.abspath(model_root) if model_root else None
        self.transformer_root = os.path.abspath(transformer_root) if transformer_root else None
        self.python_bin = python_bin or sys.executable
        self.script = os.path.join(self.repo_dir, "inference.py")
        if not os.path.isfile(self.script):
            raise FileNotFoundError(f"ROSE inference script not found: {self.script}")

    def _validate_weights(self) -> None:
        model_root = self.model_root or os.path.join(self.repo_dir, "models", "Wan2.1-Fun-1.3B-InP")
        transformer_weight = self.transformer_root or os.path.join(self.repo_dir, "weights", "transformer")
        if not os.path.isdir(model_root) or not os.path.isdir(transformer_weight):
            raise FileNotFoundError(
                "ROSE base model or transformer weights are missing. Prepare the ROSE models/ and weights/ directories before running inference."
            )
        self._link_if_needed(model_root, os.path.join(self.repo_dir, "models", "Wan2.1-Fun-1.3B-InP"))
        self._link_if_needed(transformer_weight, os.path.join(self.repo_dir, "weights", "transformer"))

    @staticmethod
    def _link_if_needed(source: str, target: str) -> None:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.islink(target):
            current = os.path.realpath(target)
            if current == os.path.realpath(source):
                return
            os.unlink(target)
        elif os.path.exists(target):
            return
        os.symlink(source, target)

    def run(
        self,
        input_video: str,
        mask_video_or_dir: str,
        output_dir: str,
        prompt: str = "",
        video_length: int | None = None,
        sample_size: tuple[int, int] | None = None,
        extra_args: list[str] | None = None,
    ) -> str:
        self._validate_weights()
        ensure_dir(output_dir)
        log_dir = ensure_dir(os.path.join(output_dir, "_logs"))
        stdout_path = os.path.join(log_dir, "rose_stdout.log")
        stderr_path = os.path.join(log_dir, "rose_stderr.log")

        info = get_video_info(input_video)
        mask_video = mask_video_or_dir
        if os.path.isdir(mask_video_or_dir):
            mask_video = os.path.join(output_dir, "_mask_video.mp4")
            mask_dir_to_video(mask_video_or_dir, mask_video, info["fps"] or 25.0)

        padded_video = os.path.join(output_dir, "_padded_video.mp4")
        padded_mask = os.path.join(output_dir, "_padded_mask.mp4")
        padded_video, padded_length = pad_video_to_16n_plus_1(input_video, padded_video)
        padded_mask, _ = pad_video_to_16n_plus_1(mask_video, padded_mask)
        clip_length = int(video_length or padded_length)
        target_size = sample_size or (info["height"], info["width"])
        if clip_length <= 0:
            raise ValueError(f"Invalid ROSE clip length: {clip_length}")
        if (clip_length - 1) % 16 != 0:
            raise ValueError(f"ROSE clip length must satisfy 16n+1, got {clip_length}")

        if padded_length <= clip_length:
            return self._run_single_clip(
                padded_video=padded_video,
                padded_mask=padded_mask,
                output_dir=output_dir,
                prompt=prompt,
                clip_length=clip_length,
                sample_size=target_size,
                fps=info["fps"] or 25.0,
                original_length=info["frame_count"],
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                extra_args=extra_args,
            )

        return self._run_chunked(
            padded_video=padded_video,
            padded_mask=padded_mask,
            output_dir=output_dir,
            prompt=prompt,
            clip_length=clip_length,
            sample_size=target_size,
            fps=info["fps"] or 25.0,
            original_length=info["frame_count"],
            extra_args=extra_args,
        )

    def _build_cmd(
        self,
        video_path: str,
        mask_path: str,
        output_dir: str,
        prompt: str,
        clip_length: int,
        sample_size: tuple[int, int],
        extra_args: list[str] | None,
    ) -> list[str]:
        cmd = [
            self.python_bin, self.script,
            "--validation_videos", os.path.abspath(video_path),
            "--validation_masks", os.path.abspath(mask_path),
            "--validation_prompts", prompt,
            "--output_dir", os.path.abspath(output_dir),
            "--video_length", str(clip_length),
            "--sample_size",
            str(sample_size[0]),
            str(sample_size[1]),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def _run_single_clip(
        self,
        padded_video: str,
        padded_mask: str,
        output_dir: str,
        prompt: str,
        clip_length: int,
        sample_size: tuple[int, int],
        fps: float,
        original_length: int,
        stdout_path: str,
        stderr_path: str,
        extra_args: list[str] | None,
    ) -> str:
        cmd = self._build_cmd(
            video_path=padded_video,
            mask_path=padded_mask,
            output_dir=output_dir,
            prompt=prompt,
            clip_length=clip_length,
            sample_size=sample_size,
            extra_args=extra_args,
        )
        with open(stdout_path, "w", encoding="utf-8") as stdout_file, open(stderr_path, "w", encoding="utf-8") as stderr_file:
            result = subprocess.run(cmd, cwd=self.repo_dir, stdout=stdout_file, stderr=stderr_file, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ROSE failed with code {result.returncode}. See {stdout_path} and {stderr_path}.")
        out_path = self._find_output_video(output_dir)
        frames, _ = read_video_frames(out_path)
        if len(frames) > original_length:
            frames = frames[:original_length]
            write_video(frames, out_path, fps)
        return out_path

    def _run_chunked(
        self,
        padded_video: str,
        padded_mask: str,
        output_dir: str,
        prompt: str,
        clip_length: int,
        sample_size: tuple[int, int],
        fps: float,
        original_length: int,
        extra_args: list[str] | None,
    ) -> str:
        video_frames, _ = read_video_frames(padded_video)
        mask_frames, _ = read_video_frames(padded_mask)
        if len(video_frames) != len(mask_frames):
            raise RuntimeError("ROSE padded video/mask length mismatch.")

        stride = clip_length - 1
        chunk_root = ensure_dir(os.path.join(output_dir, "_chunks"))
        merged_frames = []
        chunk_stdout = os.path.join(output_dir, "_logs", "rose_stdout.log")
        chunk_stderr = os.path.join(output_dir, "_logs", "rose_stderr.log")
        with open(chunk_stdout, "w", encoding="utf-8") as stdout_file, open(chunk_stderr, "w", encoding="utf-8") as stderr_file:
            for chunk_index, start in enumerate(range(0, len(video_frames) - 1, stride)):
                end = start + clip_length
                chunk_video_frames = video_frames[start:end]
                chunk_mask_frames = mask_frames[start:end]
                if len(chunk_video_frames) < clip_length:
                    chunk_video_frames.extend([chunk_video_frames[-1].copy()] * (clip_length - len(chunk_video_frames)))
                    chunk_mask_frames.extend([chunk_mask_frames[-1].copy()] * (clip_length - len(chunk_mask_frames)))

                chunk_dir = ensure_dir(os.path.join(chunk_root, f"chunk_{chunk_index:03d}"))
                chunk_video_path = os.path.join(chunk_dir, "input.mp4")
                chunk_mask_path = os.path.join(chunk_dir, "mask.mp4")
                write_video(chunk_video_frames, chunk_video_path, fps)
                write_video(chunk_mask_frames, chunk_mask_path, fps)

                cmd = self._build_cmd(
                    video_path=chunk_video_path,
                    mask_path=chunk_mask_path,
                    output_dir=chunk_dir,
                    prompt=prompt,
                    clip_length=clip_length,
                    sample_size=sample_size,
                    extra_args=extra_args,
                )
                stdout_file.write(f"\n=== chunk {chunk_index:03d} [{start}:{end}] ===\n")
                stderr_file.write(f"\n=== chunk {chunk_index:03d} [{start}:{end}] ===\n")
                stdout_file.flush()
                stderr_file.flush()
                result = subprocess.run(cmd, cwd=self.repo_dir, stdout=stdout_file, stderr=stderr_file, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ROSE chunk {chunk_index:03d} failed with code {result.returncode}. "
                        f"See {chunk_stdout} and {chunk_stderr}."
                    )

                chunk_output_path = self._find_output_video(chunk_dir)
                chunk_output_frames, _ = read_video_frames(chunk_output_path)
                if chunk_index > 0:
                    chunk_output_frames = chunk_output_frames[1:]
                merged_frames.extend(chunk_output_frames)

        merged_frames = merged_frames[:original_length]
        out_path = os.path.join(output_dir, "output.mp4")
        write_video(merged_frames, out_path, fps)
        return out_path

    @staticmethod
    def _find_output_video(output_dir: str) -> str:
        candidates = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.endswith(".mp4")
        ]
        candidates = [path for path in candidates if not os.path.basename(path).startswith("_")]
        if not candidates:
            raise RuntimeError(f"ROSE produced no output video in {output_dir}")
        candidates.sort()
        return candidates[0]
