# Part 2: Advanced Video Object Removal

## Overview

This project implements two advanced video object removal pipelines and compares them on the DAVIS 2017 dataset:

| Pipeline | Mask Extraction | Video Inpainting |
|----------|----------------|------------------|
| **Pipeline A** | VGGT4D (training-free, ViT attention mining) | ProPainter |
| **Pipeline B** | SAM 2 (promptable foundation model) | ProPainter |

**Evaluation Metrics**: JM (Jaccard Mean), JR (Jaccard Recall), PSNR, SSIM

## Quick Start

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate cv2
```
or
```bash
conda create -n cv2 python=3.10 -y
conda activate cv2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Setup External Dependencies

```bash
bash setup.sh
```

This clones VGGT4D, SAM 2, and ProPainter repos, installs them, and downloads pretrained weights.

### 3. Run Pipelines

```bash
# Pipeline A: VGGT4D + ProPainter
python run.py --method vggt4d --gpu 4

# Pipeline B: SAM 2 + ProPainter
python run.py --method sam2 --gpu 4

# Process specific sequences only
python run.py --method sam2 --sequences bmx-trees tennis --gpu 4
```

### Running on SLURM Cluster

If you are running on a SLURM-managed HPC cluster (e.g., HKUST-GZ HPC), use the provided SLURM scripts instead of running `python run.py` directly on the login node. GPU resources must be requested via SLURM job submission.

```bash
# Submit Pipeline B (SAM2 + ProPainter) on mandatory sequences
sbatch slurm_sam2.sh

# Submit Pipeline A (VGGT4D + ProPainter) on mandatory sequences
sbatch slurm_vggt4d.sh

# Submit all 30 DAVIS sequences (may need longer partition if >30 min)
sbatch slurm_sam2_all.sh
sbatch slurm_vggt4d_all.sh

# Monitor job status
squeue -u $USER

# Check output logs
cat temp/sam2_output.txt
cat temp/sam2_err.txt
```

**Notes:**
- The SLURM scripts use the `debug` partition with a 30-minute time limit and 1 GPU.
- Under SLURM `--gres=gpu:1`, the allocated GPU is always device 0, so `--gpu 0` is used.
- If processing all 30 sequences exceeds the 30-minute limit, split sequences across multiple jobs using `--sequences seq1 seq2 ...`.
- For longer jobs, consider using a different partition (e.g., `i64m1tga800u` for up to 7 days).
- Evaluation does not require a GPU and can be run on the login node or via `sbatch slurm_eval.sh`.

### 4. Evaluate

```bash
# Evaluate individual pipelines
python evaluate.py --pred results/vggt4d --davis-root ../data/DAVIS
python evaluate.py --pred results/sam2 --davis-root ../data/DAVIS

# Compare two pipelines side-by-side
python evaluate.py --pred results/vggt4d --pred2 results/sam2 --davis-root ../data/DAVIS

# Save results to JSON
python evaluate.py --pred results/sam2 --save-json results/sam2_metrics.json
```

## Project Structure

```
part2/
├── environment.yml           # Conda environment definition
├── requirements.txt          # Pip dependencies
├── setup.sh                  # Clone repos + download weights
├── run.py                    # Main CLI for running pipelines
├── evaluate.py               # Evaluation CLI
├── README.md                 # This file
├── configs/
│   └── default.yaml          # Configuration file
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # Pipeline orchestrator
│   ├── mask_vggt4d.py        # VGGT4D mask extraction
│   ├── mask_sam2.py          # SAM 2 mask extraction
│   ├── inpaint_propainter.py # ProPainter inpainting wrapper
│   └── evaluation.py         # Metric computation (JM, JR, PSNR, SSIM)
└── external/                 # Cloned external repos (git-ignored)
    ├── VGGT4D/
    ├── sam2/
    └── ProPainter/
```

## Output Structure

```
results/<method>/<sequence>/
├── masks/      # Predicted binary masks (255=foreground, 0=background)
└── frames/     # Inpainted frames (PNG)
```

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **JM** | Mask Quality | Mean IoU across all frames |
| **JR** | Mask Quality | Fraction of frames with IoU ≥ 0.5 |
| **PSNR** | Video Quality | Peak Signal-to-Noise Ratio |
| **SSIM** | Video Quality | Structural Similarity Index |

## Experiment Results

Evaluated on DAVIS 2017 validation set mandatory sequences (`bmx-trees`, `tennis`) using an NVIDIA A40 GPU.

### Quantitative Comparison

| Pipeline | Sequence | JM ↑ | JR ↑ | PSNR ↑ | SSIM ↑ |
|----------|----------|-------|-------|--------|--------|
| **VGGT4D + ProPainter** | bmx-trees | 0.0306 | 0.0000 | 16.06 | 0.5713 |
| **VGGT4D + ProPainter** | tennis | 0.0354 | 0.0000 | 21.98 | 0.7854 |
| **VGGT4D + ProPainter** | **Average** | **0.0330** | **0.0000** | **19.02** | **0.6783** |
| **SAM 2 + ProPainter** | bmx-trees | 0.7412 | 0.9500 | 27.25 | 0.9744 |
| **SAM 2 + ProPainter** | tennis | 0.9354 | 1.0000 | 22.21 | 0.9528 |
| **SAM 2 + ProPainter** | **Average** | **0.8383** | **0.9750** | **24.73** | **0.9636** |

Full results saved in `results/comparison_metrics.json`.

### Runtime

| Pipeline | Sequence | Mask Extraction | Inpainting | Total |
|----------|----------|----------------|------------|-------|
| VGGT4D + ProPainter | bmx-trees (80 frames) | 55.3s | 49.1s | 104.4s |
| VGGT4D + ProPainter | tennis (70 frames) | 28.9s | 38.9s | 67.8s |
| SAM 2 + ProPainter | bmx-trees (80 frames) | 28.8s | 52.4s | 81.2s |
| SAM 2 + ProPainter | tennis (70 frames) | 8.5s | 36.5s | 45.0s |

### Analysis

**Pipeline B (SAM 2 + ProPainter)** significantly outperforms Pipeline A on mask quality (JM: 0.84 vs 0.03). SAM 2 is a purpose-built video segmentation model that leverages ground-truth first-frame prompts to track objects reliably across the video. The high-quality masks lead to accurate inpainting by ProPainter, achieving strong PSNR and SSIM scores.

**Pipeline A (VGGT4D + ProPainter)** produces near-zero mask quality (JM: 0.033, JR: 0.000). VGGT4D is primarily designed for 4D scene reconstruction; its dynamic mask output is a secondary by-product derived from depth confidence thresholding, which is not reliable for foreground segmentation. As a result, ProPainter receives incorrect masks and cannot perform meaningful inpainting.

**Takeaway**: For the video object removal task, the quality of the mask extractor is the dominant factor. SAM 2 with first-frame GT prompts is highly effective, while using VGGT4D's depth-confidence fallback is insufficient. Potential improvement for VGGT4D: use its motion cues as prompts for SAM 2/SAM 3 to obtain accurate masks (Part 3 Direction A).

## References

- **VGGT4D**: Hu et al., "Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction", arXiv:2511.19971
- **SAM 2**: Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714
- **ProPainter**: Zhou et al., "ProPainter: Improving Propagation and Transformer for Video Inpainting", ICCV 2023
- **DAVIS 2017**: Pont-Tuset et al., "The 2017 DAVIS Challenge on Video Object Segmentation", arXiv:1704.00675
