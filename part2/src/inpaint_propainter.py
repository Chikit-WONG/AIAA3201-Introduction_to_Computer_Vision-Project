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


class ProPainterInpainter:
    """Video inpainting using ProPainter via its CLI."""

    def __init__(self, config: dict):
        self.config = config
        pp_cfg = config["propainter"]
        self.repo_dir = os.path.abspath(pp_cfg["repo_dir"])
        self.weights_dir = os.path.abspath(pp_cfg["weights_dir"])
        self.fp16 = pp_cfg.get("fp16", True)
        self.neighbor_length = pp_cfg.get("neighbor_length", 10)
        self.ref_stride = pp_cfg.get("ref_stride", 10)
        self.subvideo_length = pp_cfg.get("subvideo_length", 80)
        self.gpu_id = config.get("gpu_id", 0)

        inference_script = os.path.join(self.repo_dir, "inference_propainter.py")
        if not os.path.isfile(inference_script):
            raise FileNotFoundError(
                f"ProPainter inference script not found: {inference_script}"
            )

    def inpaint(self, frames_dir: str, masks_dir: str, output_dir: str):
        """Run ProPainter inpainting.

        Args:
            frames_dir: directory with input frames (00000.jpg, ...).
            masks_dir: directory with binary masks (00000.png, ...).
                       White (255) = inpaint region, Black (0) = keep.
            output_dir: directory to save inpainted frames.
        """
        os.makedirs(output_dir, exist_ok=True)

        inference_script = os.path.join(self.repo_dir, "inference_propainter.py")

        # ProPainter saves results to a subfolder under --output
        # We'll use a temp output and then move files
        pp_output = os.path.join(output_dir, "_propainter_tmp")

        cmd = [
            sys.executable, inference_script,
            "--video", os.path.abspath(frames_dir),
            "--mask", os.path.abspath(masks_dir),
            "--output", os.path.abspath(pp_output),
            "--neighbor_length", str(self.neighbor_length),
            "--ref_stride", str(self.ref_stride),
            "--subvideo_length", str(self.subvideo_length),
        ]
        if self.fp16:
            cmd.append("--fp16")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        print(f"[ProPainter] Running inpainting ...")
        print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, cwd=self.repo_dir, env=env,
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            print(f"[ProPainter] STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"ProPainter failed with return code {result.returncode}"
            )

        # Move inpainted frames from ProPainter output to our output_dir
        # ProPainter typically saves to <output>/inpaint_out/<result_name>/
        self._collect_results(pp_output, output_dir, frames_dir)

        # Cleanup temp dir
        if os.path.isdir(pp_output):
            shutil.rmtree(pp_output)

        print(f"[ProPainter] Inpainted frames saved to {output_dir}")

    def _collect_results(self, pp_output: str, target_dir: str,
                         frames_dir: str):
        """Find and copy inpainted frames from ProPainter output structure."""
        # ProPainter outputs: <pp_output>/<subdir>/inpaint_out.png or
        # frame-by-frame in a subfolder. Search recursively for PNG/JPG files.
        result_frames = []
        for root, dirs, files in os.walk(pp_output):
            for f in sorted(files):
                if f.endswith((".png", ".jpg")) and "mask" not in f.lower():
                    result_frames.append(os.path.join(root, f))

        if not result_frames:
            print(f"[ProPainter] Warning: no output frames found in {pp_output}")
            # Fallback: copy original frames
            orig_frames = sorted(
                glob.glob(os.path.join(frames_dir, "*.jpg"))
                + glob.glob(os.path.join(frames_dir, "*.png"))
            )
            for f in orig_frames:
                fname = os.path.splitext(os.path.basename(f))[0] + ".png"
                shutil.copy2(f, os.path.join(target_dir, fname))
            return

        # Get original frame names for consistent naming
        orig_frames = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg"))
            + glob.glob(os.path.join(frames_dir, "*.png"))
        )

        if len(result_frames) == len(orig_frames):
            # Match by order
            for src, orig in zip(result_frames, orig_frames):
                fname = os.path.splitext(os.path.basename(orig))[0] + ".png"
                shutil.copy2(src, os.path.join(target_dir, fname))
        else:
            # Just copy what we have with original names
            for i, src in enumerate(result_frames):
                if i < len(orig_frames):
                    fname = os.path.splitext(
                        os.path.basename(orig_frames[i])
                    )[0] + ".png"
                else:
                    fname = f"{i:05d}.png"
                shutil.copy2(src, os.path.join(target_dir, fname))
