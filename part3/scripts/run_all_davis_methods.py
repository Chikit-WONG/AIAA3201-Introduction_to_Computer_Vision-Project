#!/usr/bin/env python3
"""Run Part 3 on the full DAVIS dataset with direct python commands."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PART3_DIR = Path(__file__).resolve().parents[1]
if str(PART3_DIR) not in sys.path:
    sys.path.insert(0, str(PART3_DIR))

from src.io_utils import part3_method_output_slug


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
    "sam3_propainter",
    "sam3_diffueraser_object",
    "sam3_diffueraser_side_effect",
    "sam3_rose_object",
    "sam3_rose_side_effect",
]
BASE_METHOD = "sam3_propainter"
SAMPLE_DATA_SEQUENCES = {"bmx-trees", "tennis"}


def discover_davis_sequences(davis_root: Path, resolution: str) -> list[str]:
    jpeg_root = davis_root / "JPEGImages" / resolution
    return sorted(path.name for path in jpeg_root.iterdir() if path.is_dir())


def first_mask_path(annotations_dir: Path) -> Path:
    candidates = sorted(annotations_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No annotation masks found under {annotations_dir}")
    return candidates[0]


def run_command(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_sample_data_outputs(output_root: Path, sample_root: Path, sequences: list[str]) -> None:
    sample_sequences = [sequence for sequence in sequences if sequence in SAMPLE_DATA_SEQUENCES]
    if not sample_sequences:
        return
    for branch in ("masks", "videos", "logs"):
        for sequence in sample_sequences:
            seq_slug = slugify(sequence)
            source = output_root / branch / seq_slug
            target = sample_root / branch / seq_slug
            if source.exists():
                shutil.rmtree(target, ignore_errors=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, symlinks=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default="inputs/davis_videos")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-base-run", action="store_true")
    parser.add_argument("--skip-derived-runs", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None, choices=PART3_METHODS)
    parser.add_argument(
        "--sample-copy-dir",
        default="results/Sample_Data",
        help="Copy bmx-trees and tennis outputs here after the DAVIS run.",
    )
    args = parser.parse_args()

    part3_dir = Path(__file__).resolve().parents[1]
    config_path = (part3_dir / args.config).resolve()
    config = load_yaml(config_path)
    davis_root = (part3_dir / config["davis"]["root"]).resolve()
    resolution = config["davis"]["resolution"]
    output_root = (part3_dir / config["output_root"]).resolve()
    input_dir = (part3_dir / args.input_dir).resolve()
    sequences = args.sequences or discover_davis_sequences(davis_root, resolution)
    methods = args.methods or PART3_METHODS

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

    if not args.skip-derived-runs:
        derived_methods = [method for method in methods if method != BASE_METHOD]
        if derived_methods:
            print("Running the remaining Part 3 DAVIS methods using the base object masks")
        for method in derived_methods:
            for index, sequence in enumerate(sequences, start=1):
                input_video = input_dir / f"{sequence}.mp4"
                init_mask = first_mask_path(annotations_root / sequence)
                source_dir = output_root / "masks" / slugify(sequence) / part3_method_output_slug(
                    BASE_METHOD, config.get("sam3", {}).get("version")
                ) / "object_mask"
                if not source_dir.is_dir():
                    raise FileNotFoundError(f"Base mask dir missing for {sequence}: {source_dir}")
                print(f"[{index}/{len(sequences)}] {sequence} -> {method}")
                run_command(
                    [
                        python_bin,
                        str(run_py),
                        "--config",
                        str(config_path),
                        "--method",
                        method,
                        "--sequence",
                        sequence,
                        "--input",
                        str(input_video),
                        "--init-mask",
                        str(init_mask),
                        "--allow-existing-masks",
                        "--existing-mask-dir",
                        str(source_dir),
                        "--gpu",
                        str(args.gpu),
                    ],
                    cwd=part3_dir,
                )

    if args.skip_eval:
        copy_sample_data_outputs(
            output_root,
            (part3_dir / args.sample_copy_dir / output_root.name).resolve(),
            sequences,
        )
        return

    print("Evaluating DAVIS JM/JR for all Part 3 methods")
    for method in methods:
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

    copy_sample_data_outputs(
        output_root,
        (part3_dir / args.sample_copy_dir / output_root.name).resolve(),
        sequences,
    )


if __name__ == "__main__":
    main()
