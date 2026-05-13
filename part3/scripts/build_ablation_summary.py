#!/usr/bin/env python3
"""Collect SAM 3 vs SAM 3.1 ablation metrics into one table."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def load_json(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(records: list[dict], metric_keys: list[str], prefix: str) -> dict[str, float | int | str]:
    if not records:
        return {}
    row: dict[str, float | int | str] = {}
    row[f"num_{prefix}_sequences"] = len(records)
    for key in metric_keys:
        values = [float(record[key]) for record in records if key in record]
        if values:
            row[f"avg_{prefix}_{key}"] = sum(values) / len(values)
    return row


def write_csv(path: str, fieldnames: list[str], records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_markdown(path: str, fieldnames: list[str], records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for record in records:
            values = [str(record.get(field, "")) for field in fieldnames]
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part3-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()

    part3_root = os.path.abspath(args.part3_root)
    variants = {
        "sam3": os.path.join(part3_root, "results_debug", "ablation", "sam3", "metrics"),
        "sam3.1": os.path.join(part3_root, "results_debug", "ablation", "sam3_1", "metrics"),
    }

    rows = []
    for variant, metrics_dir in variants.items():
        davis_records = load_json(os.path.join(metrics_dir, "davis_mask_metrics.json"))
        wild_records = load_json(os.path.join(metrics_dir, "wild_video_metrics.json"))
        row = {"variant": variant}
        row.update(aggregate(davis_records, ["JM", "JR"], "davis"))
        row.update(aggregate(wild_records, ["PSNR", "SSIM"], "wild"))
        rows.append(row)

    fieldnames = [
        "variant",
        "num_davis_sequences",
        "avg_davis_JM",
        "avg_davis_JR",
        "num_wild_sequences",
        "avg_wild_PSNR",
        "avg_wild_SSIM",
    ]
    output_dir = os.path.join(part3_root, "results_debug", "ablation", "summary")
    write_csv(os.path.join(output_dir, "sam3_vs_sam3_1_ablation.csv"), fieldnames, rows)
    write_markdown(os.path.join(output_dir, "sam3_vs_sam3_1_ablation.md"), fieldnames, rows)
    with open(os.path.join(output_dir, "sam3_vs_sam3_1_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
