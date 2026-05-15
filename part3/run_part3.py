#!/usr/bin/env python3
"""Part 3 runner."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from src.io_utils import load_yaml, slugify
from src.pipeline import Part3Pipeline




SAMPLE_DATA_SEQUENCES = {"bmx-trees", "tennis", "bmx_trees"}


def sync_sample_data_sequence(part3_dir: str, output_root: str, sequence: str) -> None:
    if sequence not in SAMPLE_DATA_SEQUENCES:
        return

    output_path = Path(output_root)
    if not output_path.is_absolute():
        output_path = (Path(part3_dir) / output_path).resolve()
    else:
        output_path = output_path.resolve()

    parts = output_path.parts
    try:
        results_idx = parts.index("results")
    except ValueError:
        return

    relative = Path(*parts[results_idx:])
    if len(relative.parts) < 3 or relative.parts[0] != "results" or relative.parts[1] != "results_davis_full":
        return

    variant = relative.parts[2]
    seq_slug = slugify(sequence)
    target_root = output_path.parents[1] / "results_sample_data" / variant
    for branch in ("frames", "logs", "masks", "videos"):
        source = output_path / branch / seq_slug
        target = target_root / branch / seq_slug
        if source.exists():
            shutil.rmtree(target, ignore_errors=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, symlinks=True)


def main():
    parser = argparse.ArgumentParser(description="Run Part 3 video object removal pipeline.")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--method", "-m", required=True, choices=[
        "part2_sam2_propainter",
        "sam3_propainter",
        "sam3_diffueraser_object",
        "sam3_diffueraser_side_effect",
        "sam3_rose_object",
        "sam3_rose_side_effect",
    ])
    parser.add_argument("--sequence", "-s", required=True)
    parser.add_argument("--input", "-i", required=True, help="Input video path.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--inpaint-prompt", default=None)
    parser.add_argument(
        "--init-mask",
        default=None,
        help="Optional first-frame binary mask used to initialize SAM 3 with a GT-derived bbox/point prompt.",
    )
    parser.add_argument("--allow-existing-masks", action="store_true")
    parser.add_argument("--existing-mask-dir", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    part3_dir = os.path.dirname(config_path) if os.path.basename(os.path.dirname(config_path)) == "configs" else os.path.dirname(__file__)
    if os.path.basename(part3_dir) == "configs":
        part3_dir = os.path.dirname(part3_dir)
    else:
        part3_dir = os.path.dirname(os.path.abspath(__file__))

    config = load_yaml(config_path)
    if args.gpu is not None:
        config["gpu_id"] = args.gpu

    prompt = args.prompt
    if prompt is None:
        prompt = config.get("davis", {}).get("prompts", {}).get(args.sequence)
        if prompt is None:
            prompt = config.get("wild", {}).get("prompts", {}).get(args.sequence, "")

    inpaint_prompt = args.inpaint_prompt
    if inpaint_prompt is None:
        inpaint_prompt = config.get("rose", {}).get("prompts", {}).get(args.sequence)

    pipeline = Part3Pipeline(config, part3_dir=part3_dir)
    result = pipeline.run(
        method=args.method,
        sequence=args.sequence,
        input_video=os.path.abspath(args.input),
        prompt=prompt,
        inpaint_prompt=inpaint_prompt,
        init_mask_path=os.path.abspath(args.init_mask) if args.init_mask else None,
        allow_existing_masks=args.allow_existing_masks,
        existing_mask_dir=args.existing_mask_dir,
    )
    sync_sample_data_sequence(part3_dir, config.get("output_root", "results/results_wild_video"), args.sequence)

    print("Part 3 run completed:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
