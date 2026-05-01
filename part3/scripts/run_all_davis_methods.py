#!/usr/bin/env python3
"""Run Part 3 on the full DAVIS dataset with direct python commands."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    chars = []
    for char in lowered:
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "default"


PART3_METHODS = [
    "sam3_diffueraser_object",
    "sam3_diffueraser_side_effect",
    "sam3_rose_object",
    "sam3_rose_side_effect",
]
BASE_METHOD = "sam3_diffueraser_object"


def discover_davis_sequences(davis_root: Path, resolution: str) -> list[str]:
    jpeg_root = davis_root / "JPEGImages" / resolution
    return sorted(path.name for path in jpeg_root.iterdir() if path.is_dir())


def first_mask_path(annotations_dir: Path) -> Path:
    candidates = sorted(annotations_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No annotation masks found under {annotations_dir}")
    return candidates[0]


def ensure_mask_alias(source_dir: Path, target_dir: Path) -> None:
    if target_dir.is_symlink() or target_dir.exists():
        if target_dir.is_symlink() and os.path.realpath(target_dir) == os.path.realpath(source_dir):
            return
        raise FileExistsError(f"Target already exists and is not the expected alias: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source_dir, target_dir, target_is_directory=True)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default="inputs/davis_videos")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-base-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--sequences", nargs="+", default=None)
    args = parser.parse_args()

    part3_dir = Path(__file__).resolve().parents[1]
    config_path = (part3_dir / args.config).resolve()
    config = load_yaml(config_path)
    davis_root = (part3_dir / config["davis"]["root"]).resolve()
    resolution = config["davis"]["resolution"]
    output_root = (part3_dir / config["output_root"]).resolve()
    input_dir = (part3_dir / args.input_dir).resolve()
    sequences = args.sequences or discover_davis_sequences(davis_root, resolution)

    annotations_root = davis_root / "Annotations" / resolution
    run_py = part3_dir / "run_part3.py"
    eval_py = part3_dir / "evaluate_part3.py"
    python_bin = sys.executable

    if not args.skip_base_run:
        print(f"Running {BASE_METHOD} on {len(sequences)} DAVIS sequences")
        for index, sequence in enumerate(sequences, start=1):
            input_video = input_dir / f"{sequence}.mp4"
            if not input_video.is_file():
                raise FileNotFoundError(
                    f"Missing DAVIS video for {sequence}: {input_video}. "
                    "Run scripts/prepare_davis_videos.py first."
                )
            init_mask = first_mask_path(annotations_root / sequence)
            print(f"[{index}/{len(sequences)}] {sequence}")
            run_command(
                [
                    python_bin,
                    str(run_py),
                    "--config",
                    str(config_path),
                    "--method",
                    BASE_METHOD,
                    "--sequence",
                    sequence,
                    "--input",
                    str(input_video),
                    "--init-mask",
                    str(init_mask),
                    "--gpu",
                    str(args.gpu),
                ],
                cwd=part3_dir,
            )

    print("Creating DAVIS-only mask aliases for the remaining Part 3 methods")
    for sequence in sequences:
        source_dir = output_root / "masks" / slugify(sequence) / slugify(BASE_METHOD) / "object_mask"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Base mask dir missing for {sequence}: {source_dir}")
        for method in PART3_METHODS[1:]:
            target_dir = output_root / "masks" / slugify(sequence) / slugify(method) / "object_mask"
            ensure_mask_alias(source_dir, target_dir)

    if args.skip_eval:
        return

    print("Evaluating DAVIS JM/JR for all Part 3 methods")
    for method in PART3_METHODS:
        run_command(
            [
                python_bin,
                str(eval_py),
                "--config",
                str(config_path),
                "--method",
                method,
                "--evaluate-davis",
                "--metrics-tag",
                method,
            ],
            cwd=part3_dir,
        )


if __name__ == "__main__":
    main()
