#!/usr/bin/env python3
"""
Evaluation Script — Part 2
===========================
Evaluate pipeline outputs against DAVIS ground truth.

Usage:
    # Evaluate single pipeline
    python evaluate.py --pred results/sam2 --davis-root ../data/DAVIS

    # Evaluate specific sequences
    python evaluate.py --pred results/sam2 --sequences bmx-trees tennis

    # Compare two pipelines side-by-side
    python evaluate.py --pred results/vggt4d --pred2 results/sam2

    # Save results to JSON
    python evaluate.py --pred results/sam2 --save-json results/sam2_metrics.json
"""

import argparse
import json
import os

from src.evaluation import (
    evaluate_dataset,
    print_results_table,
    print_comparison_table,
)


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
        "--pred2",
        default=None,
        help="Second prediction directory for comparison (optional).",
    )
    parser.add_argument(
        "--davis-root", "-d",
        default="../data/DAVIS",
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
        help="Specific sequences to evaluate (default: auto-discover).",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Path to save results as JSON file.",
    )
    parser.add_argument(
        "--name1",
        default=None,
        help="Display name for first pipeline (default: inferred from path).",
    )
    parser.add_argument(
        "--name2",
        default=None,
        help="Display name for second pipeline (default: inferred from path).",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.pred):
        print(f"[ERROR] Prediction directory not found: {args.pred}")
        return

    # Evaluate first pipeline
    print(f"DAVIS root:   {args.davis_root}")
    print(f"Predictions:  {args.pred}")
    print(f"Resolution:   {args.resolution}")
    print()

    results_1 = evaluate_dataset(
        pred_root=args.pred,
        davis_root=args.davis_root,
        resolution=args.resolution,
        sequences=args.sequences,
    )

    if args.pred2 and os.path.isdir(args.pred2):
        # Comparison mode
        print(f"Predictions2: {args.pred2}")
        print()

        results_2 = evaluate_dataset(
            pred_root=args.pred2,
            davis_root=args.davis_root,
            resolution=args.resolution,
            sequences=args.sequences,
        )

        name1 = args.name1 or os.path.basename(os.path.normpath(args.pred))
        name2 = args.name2 or os.path.basename(os.path.normpath(args.pred2))
        print_comparison_table(results_1, results_2, name1, name2)

        # Also print individual tables
        print(f"\n--- {name1} ---")
        print_results_table(results_1)
        print(f"\n--- {name2} ---")
        print_results_table(results_2)
    else:
        # Single pipeline
        print_results_table(results_1)

    # Save JSON
    if args.save_json:
        serialisable = {}
        for k, v in results_1.items():
            serialisable[k] = {
                mk: mv for mk, mv in v.items()
                if mk != "per_frame_iou"
            }
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.save_json}")


if __name__ == "__main__":
    main()
