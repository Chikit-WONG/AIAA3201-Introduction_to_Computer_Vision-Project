#!/usr/bin/env python3
"""
Video Object Removal Pipeline — CLI Entry Point
================================================
Usage:
    # Run SAM 2 + ProPainter on all DAVIS val sequences
    python run.py --method sam2

    # Run VGGT4D + ProPainter on all DAVIS val sequences
    python run.py --method vggt4d

    # Process specific sequences
    python run.py --method sam2 --sequences bmx-trees tennis

    # Custom output directory
    python run.py --method sam2 --output results/sam2_experiment
"""

import argparse
import os
import yaml

from src.pipeline import VideoRemovalPipeline


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Part 2: Advanced Video Object Removal (VGGT4D / SAM 2 + ProPainter)",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--method", "-m",
        choices=["vggt4d", "sam2"],
        default=None,
        help="Mask extraction method (overrides config).",
    )
    parser.add_argument(
        "--davis-root", "-d",
        default=None,
        help="DAVIS dataset root (overrides config).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output root directory (default: results/<method>).",
    )
    parser.add_argument(
        "--sequences", "-s",
        nargs="+",
        default=None,
        help="Specific sequences to process (overrides config).",
    )
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=None,
        help="GPU device ID (overrides config).",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.method:
        config["method"] = args.method
    if args.davis_root:
        config["davis"]["root"] = args.davis_root
    if args.gpu is not None:
        config["gpu_id"] = args.gpu

    method = config["method"]
    davis_root = config["davis"]["root"]
    resolution = config["davis"]["resolution"]
    sequences = args.sequences or config["davis"].get("sequences", [])

    output_root = args.output or os.path.join(
        config.get("output_root", "results"), method
    )

    print(f"Method:      {method}")
    print(f"DAVIS root:  {davis_root}")
    print(f"Resolution:  {resolution}")
    print(f"Sequences:   {len(sequences)}")
    print(f"Output:      {output_root}")
    print(f"GPU:         {config.get('gpu_id', 0)}")

    # Build and run pipeline
    pipeline = VideoRemovalPipeline(config)
    pipeline.process_davis(davis_root, output_root, sequences, resolution)


if __name__ == "__main__":
    main()
