#!/usr/bin/env python3
"""
Video Object Removal Pipeline — CLI Entry Point
================================================
Usage:
    # Process DAVIS sequences (default: bmx-trees, tennis)
    python run.py --davis

    # Process specific sequences
    python run.py --davis --sequences bmx-trees tennis car-shadow

    # Disable temporal propagation (spatial-only ablation)
    python run.py --davis --no-temporal

    # Process a single video file
    python run.py --input path/to/video.mp4 --output results/my_video

    # Use custom config
    python run.py --davis --config configs/custom.yaml
"""

import argparse
import os
import yaml

from src.pipeline import VideoPipeline


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Video Object Removal Pipeline (Part 1 — Hand-crafted)",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--davis",
        action="store_true",
        help="Process DAVIS dataset sequences.",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to a video file (mp4/avi). Ignored if --davis is set.",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output root directory.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="DAVIS sequence names to process (overrides config).",
    )

    # Temporal propagation toggle
    temporal_group = parser.add_mutually_exclusive_group()
    temporal_group.add_argument(
        "--temporal",
        action="store_true",
        default=None,
        help="Enable temporal background propagation.",
    )
    temporal_group.add_argument(
        "--no-temporal",
        action="store_true",
        default=None,
        help="Disable temporal propagation (spatial-only mode).",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides for temporal propagation
    if args.temporal:
        config["temporal_propagation"]["enabled"] = True
    elif args.no_temporal:
        config["temporal_propagation"]["enabled"] = False

    tp_status = "ON" if config["temporal_propagation"]["enabled"] else "OFF"
    print(f"Temporal propagation: {tp_status}")

    # Build pipeline
    pipeline = VideoPipeline(config)

    if args.davis:
        # DAVIS mode
        davis_cfg = config["davis"]
        davis_root = davis_cfg["root"]
        resolution = davis_cfg["resolution"]
        sequences = args.sequences or davis_cfg.get("sequences") or []

        jpeg_root = os.path.join(davis_root, "JPEGImages", resolution)

        if not sequences:
            # Auto-discover all sequences
            if os.path.isdir(jpeg_root):
                sequences = sorted([
                    d for d in os.listdir(jpeg_root)
                    if os.path.isdir(os.path.join(jpeg_root, d))
                ])
            else:
                print(f"[ERROR] DAVIS JPEGImages not found at: {jpeg_root}")
                return

        print(f"Processing {len(sequences)} DAVIS sequence(s): {sequences}")
        for seq in sequences:
            seq_dir = os.path.join(jpeg_root, seq)
            if not os.path.isdir(seq_dir):
                print(f"[WARN] Sequence directory not found: {seq_dir}, skipping.")
                continue

            out_dir = os.path.join(args.output, seq)
            print(f"\n{'='*60}")
            print(f"Sequence: {seq}")
            print(f"{'='*60}")
            pipeline.process_davis_sequence(seq_dir, out_dir)

        print(f"\nAll sequences processed. Results in: {args.output}")
    elif args.input:
        # Single video file mode
        if not os.path.isfile(args.input):
            print(f"[ERROR] Video file not found: {args.input}")
            return
        print(f"Processing video: {args.input}")
        pipeline.process_video_file(args.input, args.output)
    else:
        parser.print_help()
        print("\nError: specify --davis or --input <video_path>.")


if __name__ == "__main__":
    main()
