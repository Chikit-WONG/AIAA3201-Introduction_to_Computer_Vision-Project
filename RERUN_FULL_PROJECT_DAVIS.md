# Full DAVIS Rerun Guide

中文版本: [RERUN_FULL_PROJECT_DAVIS-CN.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/RERUN_FULL_PROJECT_DAVIS-CN.md)

This guide is for rerunning the whole project with direct `python` commands instead of `sbatch`.

## Scope

- Rerun `part1`, `part2`, and `part3` on the full DAVIS dataset that is already extracted under `data/DAVIS/JPEGImages/480p/`.
- Use only DAVIS `JM` and `JR` as the final project metrics.
- Do not use the old Wild Video results in the final comparison table.

## Why This Rerun Exists

- The earlier Part 1 and Part 2 Wild Video setup was not reliable enough for final reporting.
- In Part 3, `Wild_Video1` had an input / ground-truth swap issue.
- Because of that, the clean rerun target is now the full DAVIS dataset only.

## Important Rules

- Run these commands on a machine that already has a GPU, or inside an already allocated compute node.
- Do not run heavy GPU jobs directly on an HPC login node.
- For this rerun, the final summary table keeps only DAVIS `JM` and `JR`.
- Ignore the old Wild Video sections in the older `part1/README.md` and `part2/README.md` when reproducing the final report.

## Output Targets

- `part1/results_davis_full/`
- `part2/results_davis_full/`
- `part3/outputs_davis_full/`
- `results_davis_summary/project_davis_summary.md`
- `results_davis_summary/project_davis_summary.csv`
- `results_davis_summary/project_davis_summary.json`

## Part 1

Create the Part 1 environment:

```bash
cd part1
conda env create -f environment.yml
conda activate cv
```

Run the three Part 1 conditions on the full DAVIS dataset:

```bash
python run.py --davis --config configs/davis_all_temporal_aligned.yaml --output results_davis_full/temporal_aligned
python evaluate.py --pred results_davis_full/temporal_aligned --davis-root ../data/DAVIS --save-json results_davis_full/temporal_aligned_metrics.json

python run.py --davis --config configs/davis_all_temporal_no_align.yaml --output results_davis_full/temporal_no_align
python evaluate.py --pred results_davis_full/temporal_no_align --davis-root ../data/DAVIS --save-json results_davis_full/temporal_no_align_metrics.json

python run.py --davis --config configs/davis_all_spatial_only.yaml --output results_davis_full/spatial_only
python evaluate.py --pred results_davis_full/spatial_only --davis-root ../data/DAVIS --save-json results_davis_full/spatial_only_metrics.json
```

Notes:

- These configs auto-discover all sequences under `../data/DAVIS/JPEGImages/480p/`.
- The updated Part 1 evaluation now keeps only `JM` and `JR`.

## Part 2

Create the Part 2 environment:

```bash
cd part2
conda env create -f environment.yml
conda activate cv2
bash setup.sh
```

Run both Part 2 methods on the full DAVIS dataset:

```bash
python run.py --config configs/davis_all.yaml --method sam2 --output results_davis_full/sam2 --gpu 0
python evaluate.py --pred results_davis_full/sam2 --davis-root ../data/DAVIS --save-json results_davis_full/sam2_metrics.json

python run.py --config configs/davis_all.yaml --method vggt4d --output results_davis_full/vggt4d --gpu 0
python evaluate.py --pred results_davis_full/vggt4d --davis-root ../data/DAVIS --save-json results_davis_full/vggt4d_metrics.json
```

Notes:

- `configs/davis_all.yaml` is for the full-dataset rerun and auto-discovers all available sequences.
- The updated Part 2 evaluation now keeps only `JM` and `JR`.

## Part 3

Create the Part 3 environment:

```bash
cd part3
conda env create -f environment.yml
conda activate cv2
```

Prepare external repositories and checkpoints first:

- `external/repository/sam3`
- `external/repository/DiffuEraser`
- `external/repository/ROSE`
- `../part2/external/ProPainter`
- `models/sam3/`
- `models/sam3.1/`
- `models/diffuEraser/`
- `models/Wan2.1-Fun-1.3B-InP/`
- `models/ROSE_transformer/`
- `models/sd-vae-ft-mse/`
- `models/stable-diffusion-v1-5/`

Default model download source:

- Prefer `ModelScope` first.
- `SAM 3` and `SAM 3.1` can be downloaded from `ModelScope` in the current setup, so Hugging Face access approval is not required if you stay on the default path.
- If a required checkpoint is still not covered by the current `ModelScope` helper, use the optional upstream `Hugging Face` path described in `part3/README.md`.

Recommended model preparation commands:

```bash
bash setup.sh
```

This is the recommended path. It clones the required repositories, downloads the currently confirmed `ModelScope` items, and prepares the remaining model directories under `part3/models/`.

If the repositories are already cloned and you want only the model step:

```bash
bash scripts/download_models.sh
```

Generate DAVIS mp4 files once:

```bash
python scripts/prepare_davis_videos.py --davis-root ../data/DAVIS --output-dir inputs/davis_videos
```

Run SAM 3 on the full DAVIS dataset:

```bash
python scripts/run_all_davis_methods.py --config configs/sam3_davis_all.yaml --gpu 0
```

Run SAM 3.1 on the full DAVIS dataset:

```bash
python scripts/run_all_davis_methods.py --config configs/sam3_1_davis_all.yaml --gpu 0
```

Notes:

- This script uses the first-frame DAVIS annotation to initialize SAM 3 with a GT-derived bbox and point prompt.
- For the DAVIS-only rerun, Part 3 `JM/JR` is evaluated from the mask output only.
- Therefore, the four Part 3 method rows reuse the same object masks on DAVIS. The inpainting backend is not part of DAVIS `JM/JR`.
- For model downloads, the default preference is `ModelScope`. In the current setup, `SAM 3` and `SAM 3.1` are available there, while `Hugging Face` remains an optional fallback for the remaining upstream-only items.

## Build the Final Cross-Part Table

From the project root:

```bash
python scripts/build_project_davis_summary.py
```

This writes:

- `results_davis_summary/project_davis_summary.md`
- `results_davis_summary/project_davis_summary.csv`
- `results_davis_summary/project_davis_summary.json`

## Manual Wild Video Check Before Any Future Wild Rerun

Do this before trusting any future Wild Video experiment:

1. Verify whether `data/Wild_Video/input_with_person/Wild_Video1.mp4` is really the input video.
2. Verify whether `data/Wild_Video/clean_gt_no_person/Wild_Video1-Ground_Truth.mp4` is really the clean ground truth.
3. Only rerun Wild Video after the swap issue has been fixed.

## Recommended Rerun Order

1. `part1`
2. `part2`
3. `part3` `sam3`
4. `part3` `sam3.1`
5. `scripts/build_project_davis_summary.py`
