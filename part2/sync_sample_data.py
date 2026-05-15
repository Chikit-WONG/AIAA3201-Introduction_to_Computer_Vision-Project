#!/usr/bin/env python3
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "results_davis_full"
DST = ROOT / "results" / "results_sample_data"
SEQS = ["bmx-trees", "tennis"]
METHODS = ["sam2", "vggt4d"]

for method in METHODS:
    for seq in SEQS:
        src = SRC / method / seq
        dst = DST / method / seq
        if src.exists():
            shutil.rmtree(dst, ignore_errors=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)

print(DST)
