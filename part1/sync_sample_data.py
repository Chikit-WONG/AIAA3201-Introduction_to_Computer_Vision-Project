#!/usr/bin/env python3
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "results_davis_full"
DST = ROOT / "results" / "results_sample_data"
SEQS = ["bmx-trees", "tennis"]
VARIANTS = ["temporal_aligned", "temporal_no_align", "spatial_only"]

for variant in VARIANTS:
    for seq in SEQS:
        src = SRC / variant / seq
        dst = DST / variant / seq
        if src.exists():
            shutil.rmtree(dst, ignore_errors=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)

print(DST)
