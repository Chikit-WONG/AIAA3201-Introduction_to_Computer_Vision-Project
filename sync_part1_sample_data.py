from pathlib import Path
import shutil
ROOT = Path('/hpc2hdd/home/yuxuanzhao/xuhaodong/AIAA3201-Introduction_to_Computer_Vision-Project/part1')
SRC = ROOT / 'results' / 'results_davis_full'
DST = ROOT / 'results' / 'results_sample_data'
SEQS = ['bmx-trees', 'tennis']
VARIANTS = ['temporal_aligned', 'temporal_no_align', 'spatial_only']
for variant in VARIANTS:
    for seq in SEQS:
        src = SRC / variant / seq
        dst = DST / variant / seq
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
print(DST)
