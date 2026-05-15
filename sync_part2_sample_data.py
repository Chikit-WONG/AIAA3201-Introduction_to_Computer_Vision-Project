from pathlib import Path
import shutil
ROOT = Path('/hpc2hdd/home/yuxuanzhao/xuhaodong/AIAA3201-Introduction_to_Computer_Vision-Project/part2')
SRC = ROOT / 'results' / 'results_davis_full'
DST = ROOT / 'results' / 'results_sample_data'
SEQS = ['bmx-trees', 'tennis']
METHODS = ['sam2', 'vggt4d']
for method in METHODS:
    for seq in SEQS:
        src = SRC / method / seq
        dst = DST / method / seq
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
print(DST)
