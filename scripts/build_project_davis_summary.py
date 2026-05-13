#!/usr/bin/env python3
"""Aggregate full-DAVIS JM/JR results across Part 1, Part 2, and Part 3."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results_davis_summary"


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def load_dict_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_list_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def summarize_dict_metrics(part: str, method: str, path: Path) -> dict | None:
    records = load_dict_metrics(path)
    if not records:
        return None
    sequence_records = {
        key: value
        for key, value in records.items()
        if key != "average" and isinstance(value, dict)
    }
    jm_values = [float(item["JM"]) for item in sequence_records.values() if "JM" in item]
    jr_values = [float(item["JR"]) for item in sequence_records.values() if "JR" in item]
    return {
        "part": part,
        "method": method,
        "variant": "",
        "num_sequences": len(sequence_records),
        "avg_JM": average(jm_values),
        "avg_JR": average(jr_values),
        "source": str(path.relative_to(PROJECT_ROOT)),
        "notes": "",
    }


def summarize_list_metrics(part: str, variant: str, method: str, path: Path) -> dict | None:
    records = load_list_metrics(path)
    if not records:
        return None
    jm_values = [float(item["JM"]) for item in records if "JM" in item]
    jr_values = [float(item["JR"]) for item in records if "JR" in item]
    return {
        "part": part,
        "method": method,
        "variant": variant,
        "num_sequences": len(records),
        "avg_JM": average(jm_values),
        "avg_JR": average(jr_values),
        "source": str(path.relative_to(PROJECT_ROOT)),
        "notes": "DAVIS rows reuse the same SAM 3 object masks; inpainting backend is not part of JM/JR.",
    }


def fmt(value: float | None, highlight: bool = False) -> str:
    if value is None:
        return ""
    text = f"{value:.4f}"
    return f"**{text}**" if highlight else text


def row_label(row: dict) -> str:
    return f"{row['part']} / {row['method']} {row['variant']}".strip()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_jm = max((row["avg_JM"] for row in rows if row["avg_JM"] is not None), default=None)
    max_jr = max((row["avg_JR"] for row in rows if row["avg_JR"] is not None), default=None)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Full DAVIS Summary\n\n")
        f.write("This table follows the rerun protocol that keeps only DAVIS `JM` and `JR`.\n\n")
        f.write("| Part | Method | Variant | #Seq | Avg JM | Avg JR | Notes |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in rows:
            jm_highlight = max_jm is not None and row["avg_JM"] == max_jm
            jr_highlight = max_jr is not None and row["avg_JR"] == max_jr
            f.write(
                "| "
                + " | ".join(
                    [
                        row["part"],
                        row["method"],
                        row["variant"],
                        str(row["num_sequences"]),
                        fmt(row["avg_JM"], jm_highlight),
                        fmt(row["avg_JR"], jr_highlight),
                        row["notes"],
                    ]
                )
                + " |\n"
            )

        if rows:
            best_jm = [row for row in rows if row["avg_JM"] == max_jm]
            best_jr = [row for row in rows if row["avg_JR"] == max_jr]
            f.write("\n## Notes\n\n")
            f.write(f"- Best `JM`: {', '.join(row_label(row) for row in best_jm)}\n")
            f.write(f"- Best `JR`: {', '.join(row_label(row) for row in best_jr)}\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-sam3-1", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []

    part1_files = [
        ("temporal_aligned", PROJECT_ROOT / "part1/results_davis_full/temporal_aligned_metrics.json"),
        ("temporal_no_align", PROJECT_ROOT / "part1/results_davis_full/temporal_no_align_metrics.json"),
        ("spatial_only", PROJECT_ROOT / "part1/results_davis_full/spatial_only_metrics.json"),
    ]
    part2_files = [
        ("sam2_propainter", PROJECT_ROOT / "part2/results_davis_full/sam2_metrics.json"),
        ("vggt4d_propainter", PROJECT_ROOT / "part2/results_davis_full/vggt4d_metrics.json"),
    ]
    part3_files = [
        ("sam3", "sam3_propainter", first_existing_path(
            PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3/metrics/davis_mask_metrics__sam3_propainter.json",
            PROJECT_ROOT / "part3/outputs/davis_full/sam3/metrics/davis_mask_metrics__sam3_propainter.json",
            PROJECT_ROOT / "part3/outputs_davis_full/sam3/metrics/davis_mask_metrics__sam3_propainter.json",
        )),
        ("sam3", "sam3_diffueraser_object", first_existing_path(
            PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3/metrics/davis_mask_metrics__sam3_diffueraser_object.json",
            PROJECT_ROOT / "part3/outputs/davis_full/sam3/metrics/davis_mask_metrics__sam3_diffueraser_object.json",
            PROJECT_ROOT / "part3/outputs_davis_full/sam3/metrics/davis_mask_metrics__sam3_diffueraser_object.json",
        )),
        ("sam3", "sam3_diffueraser_side_effect", first_existing_path(
            PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3/metrics/davis_mask_metrics__sam3_diffueraser_side_effect.json",
            PROJECT_ROOT / "part3/outputs/davis_full/sam3/metrics/davis_mask_metrics__sam3_diffueraser_side_effect.json",
            PROJECT_ROOT / "part3/outputs_davis_full/sam3/metrics/davis_mask_metrics__sam3_diffueraser_side_effect.json",
        )),
        ("sam3", "sam3_rose_object", first_existing_path(
            PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3/metrics/davis_mask_metrics__sam3_rose_object.json",
            PROJECT_ROOT / "part3/outputs/davis_full/sam3/metrics/davis_mask_metrics__sam3_rose_object.json",
            PROJECT_ROOT / "part3/outputs_davis_full/sam3/metrics/davis_mask_metrics__sam3_rose_object.json",
        )),
        ("sam3", "sam3_rose_side_effect", first_existing_path(
            PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3/metrics/davis_mask_metrics__sam3_rose_side_effect.json",
            PROJECT_ROOT / "part3/outputs/davis_full/sam3/metrics/davis_mask_metrics__sam3_rose_side_effect.json",
            PROJECT_ROOT / "part3/outputs_davis_full/sam3/metrics/davis_mask_metrics__sam3_rose_side_effect.json",
        )),
    ]
    if args.include_sam3_1:
        part3_files.extend([
            ("sam3.1", "sam3_propainter", first_existing_path(
                PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3_1/metrics/davis_mask_metrics__sam3_propainter.json",
                PROJECT_ROOT / "part3/outputs/davis_full/sam3_1/metrics/davis_mask_metrics__sam3_propainter.json",
                PROJECT_ROOT / "part3/outputs_davis_full/sam3_1/metrics/davis_mask_metrics__sam3_propainter.json",
            )),
            ("sam3.1", "sam3_diffueraser_object", first_existing_path(
                PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3_1/metrics/davis_mask_metrics__sam3_diffueraser_object.json",
                PROJECT_ROOT / "part3/outputs/davis_full/sam3_1/metrics/davis_mask_metrics__sam3_diffueraser_object.json",
                PROJECT_ROOT / "part3/outputs_davis_full/sam3_1/metrics/davis_mask_metrics__sam3_diffueraser_object.json",
            )),
            ("sam3.1", "sam3_diffueraser_side_effect", first_existing_path(
                PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3_1/metrics/davis_mask_metrics__sam3_diffueraser_side_effect.json",
                PROJECT_ROOT / "part3/outputs/davis_full/sam3_1/metrics/davis_mask_metrics__sam3_diffueraser_side_effect.json",
                PROJECT_ROOT / "part3/outputs_davis_full/sam3_1/metrics/davis_mask_metrics__sam3_diffueraser_side_effect.json",
            )),
            ("sam3.1", "sam3_rose_object", first_existing_path(
                PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3_1/metrics/davis_mask_metrics__sam3_rose_object.json",
                PROJECT_ROOT / "part3/outputs/davis_full/sam3_1/metrics/davis_mask_metrics__sam3_rose_object.json",
                PROJECT_ROOT / "part3/outputs_davis_full/sam3_1/metrics/davis_mask_metrics__sam3_rose_object.json",
            )),
            ("sam3.1", "sam3_rose_side_effect", first_existing_path(
                PROJECT_ROOT / "part3/results/DAVIS_Dataset/sam3_1/metrics/davis_mask_metrics__sam3_rose_side_effect.json",
                PROJECT_ROOT / "part3/outputs/davis_full/sam3_1/metrics/davis_mask_metrics__sam3_rose_side_effect.json",
                PROJECT_ROOT / "part3/outputs_davis_full/sam3_1/metrics/davis_mask_metrics__sam3_rose_side_effect.json",
            )),
        ])

    for method, path in part1_files:
        row = summarize_dict_metrics("part1", method, path)
        if row:
            rows.append(row)
    for method, path in part2_files:
        row = summarize_dict_metrics("part2", method, path)
        if row:
            rows.append(row)
    for variant, method, path in part3_files:
        row = summarize_list_metrics("part3", variant, method, path)
        if row:
            rows.append(row)

    rows.sort(key=lambda row: (row["part"], row["method"], row["variant"]))

    fieldnames = ["part", "method", "variant", "num_sequences", "avg_JM", "avg_JR", "source", "notes"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "project_davis_summary.csv", rows, fieldnames)
    (OUTPUT_DIR / "project_davis_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(OUTPUT_DIR / "project_davis_summary.md", rows)
    print(f"Wrote {OUTPUT_DIR / 'project_davis_summary.md'}")


if __name__ == "__main__":
    main()
