#!/usr/bin/env python3
"""Evaluate Part 3 outputs."""

from __future__ import annotations

import argparse
import os
from glob import glob

from src.io_utils import load_yaml, part3_method_output_slug, slugify
from src.metrics_mask import evaluate_mask_sequence
from src.metrics_video import evaluate_video_quality
from src.report_tables import save_records_csv, save_records_json


def resolve_pred_video(output_root: str, sequence: str, method: str, sam3_version: str | None = None) -> str:
    video_dir = os.path.join(output_root, "videos", slugify(sequence), part3_method_output_slug(method, sam3_version))
    preferred = [
        os.path.join(video_dir, "output.mp4"),
        os.path.join(video_dir, "diffueraser_result.mp4"),
        os.path.join(video_dir, "example-1.mp4"),
    ]
    for path in preferred:
        if os.path.isfile(path):
            return path
    candidates = sorted(
        path for path in glob(os.path.join(video_dir, "*.mp4"))
        if not os.path.basename(path).startswith("_")
    )
    if not candidates:
        raise FileNotFoundError(f"No predicted video found under {video_dir}")
    return candidates[0]


def discover_davis_sequences(part3_dir: str, config: dict) -> list[str]:
    davis_cfg = config["davis"]
    ann_root = os.path.join(part3_dir, davis_cfg["root"], "Annotations", davis_cfg["resolution"])
    if not os.path.isdir(ann_root):
        raise FileNotFoundError(f"DAVIS annotations root not found: {ann_root}")
    return sorted(
        name for name in os.listdir(ann_root)
        if os.path.isdir(os.path.join(ann_root, name))
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 3 outputs.")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--method", "-m", required=True)
    parser.add_argument("--sequence", "-s", nargs="+", default=None)
    parser.add_argument("--evaluate-davis", action="store_true")
    parser.add_argument("--evaluate-wild", action="store_true")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--metrics-tag", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    part3_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.abspath(os.path.join(part3_dir, config.get("output_root", "outputs")))
    metrics_dir = os.path.join(output_root, "metrics")
    metrics_suffix = f"__{slugify(args.metrics_tag)}" if args.metrics_tag else ""
    records = []

    if args.evaluate_davis:
        sequences = args.sequence or config["davis"].get("sequences") or discover_davis_sequences(part3_dir, config)
        for seq in sequences:
            pred_mask_dir = os.path.join(
                output_root,
                "masks",
                slugify(seq),
                part3_method_output_slug(args.method, config.get("sam3", {}).get("version")),
                "object_mask",
            )
            gt_mask_dir = os.path.join(part3_dir, config["davis"]["root"], "Annotations", config["davis"]["resolution"], seq)
            metrics = evaluate_mask_sequence(pred_mask_dir, gt_mask_dir)
            record = {"sequence": seq, "method": args.method, **metrics}
            records.append(record)
        save_records_csv(os.path.join(metrics_dir, f"davis_mask_metrics{metrics_suffix}.csv"), ["sequence", "method", "JM", "JR", "num_frames", "threshold"], records)
        save_records_json(os.path.join(metrics_dir, f"davis_mask_metrics{metrics_suffix}.json"), records)

    if args.evaluate_wild:
        sequences = args.sequence or list(config["wild"]["pairs"].keys())
        wild_records = []
        for seq in sequences:
            pred_video = resolve_pred_video(output_root, seq, args.method, config.get("sam3", {}).get("version"))
            gt_name = config["wild"]["pairs"][seq]
            gt_video = os.path.join(part3_dir, config["wild"]["clean_gt_dir"], f"{gt_name}.mp4")
            align = config["wild"].get("align_for_metrics", True) and not args.no_align
            metrics = evaluate_video_quality(pred_video, gt_video, align=align)
            record = {"sequence": seq, "method": args.method, **metrics}
            wild_records.append(record)
        save_records_csv(os.path.join(metrics_dir, f"wild_video_metrics{metrics_suffix}.csv"), ["sequence", "method", "PSNR", "SSIM", "num_frames", "aligned"], wild_records)
        save_records_json(os.path.join(metrics_dir, f"wild_video_metrics{metrics_suffix}.json"), wild_records)


if __name__ == "__main__":
    main()
