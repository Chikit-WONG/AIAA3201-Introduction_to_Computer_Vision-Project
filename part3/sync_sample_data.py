#!/usr/bin/env python3
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "results_davis_full"
DST = ROOT / "results" / "results_sample_data"
SEQS = ["bmx_trees", "tennis"]

for variant_dir in [p for p in SRC.iterdir() if p.is_dir()]:
    for category in ["frames", "logs", "masks", "videos", "metrics"]:
        for seq in SEQS:
            src = variant_dir / category / seq
            dst = DST / variant_dir.name / category / seq
            if src.exists():
                shutil.rmtree(dst, ignore_errors=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst, symlinks=True)

print(DST)
