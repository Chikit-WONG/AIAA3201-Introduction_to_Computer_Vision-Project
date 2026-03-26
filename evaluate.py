#!/usr/bin/env python3
"""
Evaluation Script — Standalone
==============================
Evaluate pipeline outputs against DAVIS ground truth.

Usage:
    # Evaluate default sequences (bmx-trees, tennis)
    python evaluate.py --pred results --davis-root data/DAVIS

    # Evaluate all sequences found in results/
    python evaluate.py --pred results --davis-root data/DAVIS --all

    # Evaluate specific sequences
    python evaluate.py --pred results --davis-root data/DAVIS --sequences bmx-trees tennis car-shadow
"""

import argparse
import json
import os

from src.evaluation import evaluate_dataset, print_results_table


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video object removal results against DAVIS GT.",
    )
    parser.add_argument(
        "--pred", "-p",
        required=True,
        help="Root directory of pipeline outputs (contains <seq>/masks/ and <seq>/frames/).",
    )
    parser.add_argument(
        "--davis-root", "-d",
        default="data/DAVIS",
        help="Root directory of DAVIS dataset.",
    )
    parser.add_argument(
        "--resolution", "-r",
        default="480p",
        choices=["480p", "Full-Resolution"],
        help="DAVIS resolution to use.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Specific sequences to evaluate. Default: auto-discover from --pred.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all sequences found in the prediction directory.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Path to save results as JSON file.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.pred):
        print(f"[ERROR] Prediction directory not found: {args.pred}")
        return

    sequences = args.sequences
    if not args.all and sequences is None:
        # Default: evaluate what's available
        sequences = None  # evaluate_dataset will auto-discover

    print(f"DAVIS root:  {args.davis_root}")
    print(f"Predictions: {args.pred}")
    print(f"Resolution:  {args.resolution}")
    print()

    results = evaluate_dataset(
        pred_root=args.pred,
        davis_root=args.davis_root,
        resolution=args.resolution,
        sequences=sequences,
    )

    print_results_table(results)

    if args.save_json:
        # Remove non-serialisable data
        serialisable = {}
        for k, v in results.items():
            serialisable[k] = {
                mk: mv for mk, mv in v.items()
                if mk != "per_frame_iou"
            }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.save_json}")


if __name__ == "__main__":
    main()
