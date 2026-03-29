# Plan: Complete Part 2 — Video Object Removal (VGGT4D/SAM2 + ProPainter)

## Context
Part 2 of the AIAA3201 Computer Vision project requires running two AI-driven pipelines on DAVIS 2017 dataset:
- **Pipeline A**: VGGT4D (training-free mask extraction) + ProPainter (video inpainting)
- **Pipeline B**: SAM 2 (promptable segmentation) + ProPainter (video inpainting)

All source code is already written (`run.py`, `evaluate.py`, `src/*.py`). External repos and model weights are downloaded. DAVIS dataset exists. The remaining work is: fix environment, fix code bugs, create SLURM scripts, run pipelines on mandatory sequences (bmx-trees, tennis), evaluate, and clean up.

**Project root**: `/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project`
**Part2 dir**: `<root>/part2`

---

## Pre-flight Check Results

| Item | Status | Note |
|------|--------|------|
| Conda env `cv2` | Exists | PyTorch 2.10.0+cu126 (newer than required, OK) |
| External repos | All cloned | VGGT4D, SAM2, ProPainter in `external/` |
| Model weights | All downloaded | VGGT4D 1.4GB, SAM2 855MB, ProPainter 3 files |
| DAVIS dataset | Exists | At `../data/DAVIS` with JPEGImages + Annotations |
| `einops` package | **MISSING** | Required by VGGT4D + ProPainter |
| `timm` package | **MISSING** | Required by ProPainter |
| `sam2` package | **NOT INSTALLED** | Must `pip install -e external/sam2` |
| `hydra-core` package | **MISSING** | Required by SAM2 (installed as SAM2 dependency) |
| `--save_frames` flag | **MISSING in code** | ProPainter won't save individual frames without it |
| `gpu_id` default | **Wrong (4)** | Must be 0 under SLURM |
| Junk files | 2 files | `=0.21.0` and `=4.8.0` pip artifacts to delete |

---

## Step 1: Fix Environment (on login node)

Install missing packages in `cv2` conda env:
```bash
conda activate cv2
pip install einops timm imageio-ffmpeg
cd part2/external/sam2 && pip install -e . && cd ../../..
```
Verify: `python -c "import einops, timm, sam2; print('OK')"`

## Step 2: Fix Code Bugs

### 2a. Add `--save_frames` to ProPainter command
**File**: `part2/src/inpaint_propainter.py:60`
Add `"--save_frames",` to the `cmd` list after `"--subvideo_length", str(...)` and before the `fp16` check. Without this, ProPainter only outputs a video file (`inpaint_out.mp4`), not individual frames, and `_collect_results()` finds nothing — falling back to copying original (un-inpainted) frames.

### 2b. Change default gpu_id
**File**: `part2/configs/default.yaml:46`
Change `gpu_id: 4` to `gpu_id: 0`. Under SLURM `--gres=gpu:1`, the allocated GPU is always device 0.

## Step 3: Clean Up Junk Files

Delete pip artifacts in part2 root:
```bash
rm "part2/=0.21.0" "part2/=4.8.0"
```

## Step 4: Create SLURM Job Scripts

Create scripts in `part2/` following the reference pattern (`run_qwen3_quick_start.sh`). All use:
- `#SBATCH -p debug` + `--gres=gpu:1` + `--time=00:30:00`
- `source conda.sh && conda activate cv2`
- `module load cuda/12.6` (matches PyTorch cu126)
- Output logs to `part2/temp/`

### `slurm_sam2.sh` — Pipeline B (mandatory sequences: bmx-trees, tennis)
```bash
python run.py --method sam2 --gpu 0 --sequences bmx-trees tennis
```

### `slurm_vggt4d.sh` — Pipeline A (mandatory sequences: bmx-trees, tennis)
```bash
python run.py --method vggt4d --gpu 0 --sequences bmx-trees tennis
```

### `slurm_sam2_all.sh` — Pipeline B (all 30 sequences)
Uses all sequences from config. May exceed 30-min limit; will batch if needed.

### `slurm_vggt4d_all.sh` — Pipeline A (all 30 sequences)
Same pattern.

### `slurm_eval.sh` — Evaluation (no GPU needed)
```bash
python evaluate.py --pred results/vggt4d --pred2 results/sam2 \
  --davis-root ../data/DAVIS --save-json results/comparison_metrics.json
```

## Step 5: Update README.md

Add a "Running on SLURM Cluster" section after Quick Start section (after line 63), explaining:
- How to submit: `sbatch slurm_sam2.sh`
- How to monitor: `squeue -u $USER`
- How to check output: `cat temp/sam2_output.txt`
- The `--gpu 0` requirement under SLURM
- Batching strategy for 30-min debug limit

Keep all existing README content intact.

## Step 6: Update .gitignore

Add to `<root>/.gitignore`:
```
temp/
*.code-workspace
*.pth
slurm-*.out
```

## Step 7: Run Pipelines

Execution order:
1. `sbatch slurm_sam2.sh` (mandatory: bmx-trees, tennis)
2. `sbatch slurm_vggt4d.sh` (mandatory: bmx-trees, tennis)
3. Wait for completion, check `temp/*.txt` logs
4. Verify results exist: `ls results/{sam2,vggt4d}/{bmx-trees,tennis}/{masks,frames}/`
5. `sbatch slurm_eval.sh` (after both pipelines complete)

## Step 8: Verify Results

Expected output structure:
```
results/sam2/bmx-trees/masks/     # Binary masks (0/255 PNG)
results/sam2/bmx-trees/frames/    # Inpainted frames (PNG)
results/vggt4d/bmx-trees/masks/
results/vggt4d/bmx-trees/frames/
```

---

## Files Summary

### Modify
| File | Change |
|------|--------|
| `part2/src/inpaint_propainter.py` | Add `--save_frames` to cmd list |
| `part2/configs/default.yaml` | `gpu_id: 4` -> `gpu_id: 0` |
| `part2/README.md` | Add SLURM instructions section |
| `<root>/.gitignore` | Add `temp/`, `*.code-workspace`, `*.pth`, `slurm-*.out` |

### Create
| File | Purpose |
|------|---------|
| `part2/slurm_sam2.sh` | SLURM job: SAM2 pipeline (mandatory seqs) |
| `part2/slurm_vggt4d.sh` | SLURM job: VGGT4D pipeline (mandatory seqs) |
| `part2/slurm_sam2_all.sh` | SLURM job: SAM2 pipeline (all 30 seqs) |
| `part2/slurm_vggt4d_all.sh` | SLURM job: VGGT4D pipeline (all 30 seqs) |
| `part2/slurm_eval.sh` | SLURM job: evaluation |
| `part2/temp/` | Directory for SLURM logs |

### Delete
| File | Reason |
|------|--------|
| `part2/=0.21.0` | pip artifact |
| `part2/=4.8.0` | pip artifact (empty) |

---

## Risks

1. **VGGT4D OOM**: Loads all frames at once into GPU. Long sequences (80+ frames at 480p) may exceed A40 48GB. Mitigation: start with short mandatory sequences.
2. **30-min time limit**: All 30 sequences may take ~60-90min per pipeline. Mitigation: mandatory-first approach, then batch remaining.
3. **PyTorch version**: 2.10.0 vs required 2.5.1 — should be backward compatible. Will test with mandatory sequences first.
