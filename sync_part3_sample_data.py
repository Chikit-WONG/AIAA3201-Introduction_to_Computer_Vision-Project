from pathlib import Path
import shutil
ROOT = Path('/hpc2hdd/home/yuxuanzhao/xuhaodong/AIAA3201-Introduction_to_Computer_Vision-Project/part3')
SRC = ROOT / 'results' / 'results_davis_full'
DST = ROOT / 'results' / 'results_sample_data'
SEQS = ['bmx_trees', 'tennis']
VARIANTS = [p.name for p in SRC.iterdir() if p.is_dir()]
for variant in VARIANTS:
    for category in ['frames','logs','masks','videos']:
        for seq in SEQS:
            src = SRC / variant / category / seq
            dst = DST / variant / category / seq
            if not src.exists():
                continue
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
print(DST)
