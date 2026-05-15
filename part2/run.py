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
import shutil
from pathlib import Path
import yaml

from src.pipeline import VideoRemovalPipeline
from src.video_utils import pack_result_videos as pack_result_videos_root


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




SAMPLE_DATA_SEQUENCES = {"bmx-trees", "tennis"}


def sync_sample_data_sequence(output_root: str, sequence: str, method: str) -> None:
    if sequence not in SAMPLE_DATA_SEQUENCES:
        return

    output_path = Path(output_root).resolve()
    parts = output_path.parts
    try:
        results_idx = parts.index("results")
    except ValueError:
        return

    relative = Path(*parts[results_idx:])
    if len(relative.parts) < 3 or relative.parts[0] != "results" or relative.parts[1] != "results_davis_full":
        return

    source = output_path / sequence
    target = output_path.parents[1] / "results_sample_data" / method / sequence
    if not source.is_dir():
        return
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


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
    if not sequences:
        jpeg_root = os.path.join(davis_root, "JPEGImages", resolution)
        if not os.path.isdir(jpeg_root):
            print(f"[ERROR] DAVIS JPEGImages not found at: {jpeg_root}")
            return
        sequences = sorted(
            d for d in os.listdir(jpeg_root)
            if os.path.isdir(os.path.join(jpeg_root, d))
        )

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

    for sequence in sequences:
        sync_sample_data_sequence(output_root, sequence, method)

    pack_result_videos_root(Path(output_root).resolve(), Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
