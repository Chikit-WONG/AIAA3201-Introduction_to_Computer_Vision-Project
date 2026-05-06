#!/usr/bin/env python3
"""Aggregate full Part 3 metrics into comparison tables."""

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


def average(records: list[dict], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record]
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part3-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()

    root = os.path.abspath(args.part3_root)
    variants = {
        "sam3": os.path.join(root, "outputs", "full", "sam3", "metrics"),
        "sam3.1": os.path.join(root, "outputs", "full", "sam3_1", "metrics"),
    }
    methods = [
        "sam3_diffueraser_object",
        "sam3_diffueraser_side_effect",
        "sam3_rose_object",
        "sam3_rose_side_effect",
    ]

    rows = []
    for variant, metrics_dir in variants.items():
        for method in methods:
            tag = method
            davis_records = load_json(os.path.join(metrics_dir, f"davis_mask_metrics__{tag}.json"))
            wild_records = load_json(os.path.join(metrics_dir, f"wild_video_metrics__{tag}.json"))
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "num_davis_sequences": len(davis_records),
                    "avg_davis_JM": average(davis_records, "JM"),
                    "avg_davis_JR": average(davis_records, "JR"),
                    "num_wild_sequences": len(wild_records),
                    "avg_wild_PSNR": average(wild_records, "PSNR"),
                    "avg_wild_SSIM": average(wild_records, "SSIM"),
                }
            )

    fieldnames = [
        "variant",
        "method",
        "num_davis_sequences",
        "avg_davis_JM",
        "avg_davis_JR",
        "num_wild_sequences",
        "avg_wild_PSNR",
        "avg_wild_SSIM",
    ]
    output_dir = os.path.join(root, "outputs", "full", "summary")
    write_csv(os.path.join(output_dir, "part3_full_summary.csv"), fieldnames, rows)
    write_md(os.path.join(output_dir, "part3_full_summary.md"), fieldnames, rows)
    with open(os.path.join(output_dir, "part3_full_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
