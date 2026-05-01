#!/usr/bin/env python3
"""Part 3 runner."""

from __future__ import annotations

import argparse
import os

from src.io_utils import load_yaml
from src.pipeline import Part3Pipeline


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
            prompt = config.get("wild", {}).get("prompts", {}).get(args.sequence, "person")

    pipeline = Part3Pipeline(config, part3_dir=part3_dir)
    result = pipeline.run(
        method=args.method,
        sequence=args.sequence,
        input_video=os.path.abspath(args.input),
        prompt=prompt,
        init_mask_path=os.path.abspath(args.init_mask) if args.init_mask else None,
        allow_existing_masks=args.allow_existing_masks,
        existing_mask_dir=args.existing_mask_dir,
    )
    print("Part 3 run completed:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
