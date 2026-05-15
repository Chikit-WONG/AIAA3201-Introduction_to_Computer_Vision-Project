"""Simple report table writers."""

from __future__ import annotations

import csv
import os

from .io_utils import ensure_dir, save_json


def save_records_csv(path: str, fieldnames: list[str], records: list[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def save_records_json(path: str, records: list[dict]) -> None:
    save_json(path, records)

