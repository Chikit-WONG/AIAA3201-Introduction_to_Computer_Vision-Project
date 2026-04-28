# AIAA3201 — Video Object Removal & Inpainting (Part 1: Hand-crafted)

## Overview

A complete video dynamic object removal and background restoration pipeline using
classic computer vision techniques:

1. **Mask Extraction**: YOLOv8-Seg instance segmentation
2. **Dynamic Filtering**: Lucas-Kanade sparse optical flow
3. **Mask Dilation**: Morphological operations for motion blur coverage
4. **Temporal Background Propagation** *(toggleable)*: Borrow clean pixels from neighbouring frames
5. **Spatial Inpainting Fallback**: `cv2.inpaint` (Telea / Navier-Stokes)

Evaluated on **DAVIS 2017** using JM, JR, PSNR, and SSIM metrics.

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate cv
```

### 2. Download DAVIS 2017 Dataset

Download the DAVIS 2017 Semi-supervised TrainVal (480p) and extract to `../data/DAVIS/`:

```
../data/DAVIS/
├── JPEGImages/480p/<sequence_name>/
└── Annotations/480p/<sequence_name>/
```

## Usage

### Process DAVIS Sequences (default: bmx-trees, tennis)

```bash
python run.py --davis
```

### Process Specific Sequences

```bash
python run.py --davis --sequences bmx-trees tennis car-shadow
```

### Disable Temporal Propagation (ablation)

```bash
python run.py --davis --no-temporal --output results_no_temporal
```

### Process a Single Video File

```bash
python run.py --input path/to/video.mp4 --output results/my_video
```

### Running on SLURM Cluster

GPU resources on the HPC cluster must be requested via SLURM. Use the provided script to process all Wild Videos in one job.

```bash
# Submit Wild Video job (6 videos: ride1-3, run1-3)
sbatch slurm_wild.sh

# Monitor job status
squeue -u $USER

# Check output logs
cat temp/wild_output.txt
cat temp/wild_err.txt
```

**Notes:**
- Uses the `debug` partition (30-minute limit, 1 GPU).
- Results are saved to `results/wild_video/<seq>/` (frames + output.mp4).

### Custom Config

```bash
python run.py --davis --config configs/custom.yaml
```

## Evaluation

```bash
# Evaluate all processed sequences
python evaluate.py --pred results --davis-root ../data/DAVIS

# Save results as JSON
python evaluate.py --pred results --davis-root ../data/DAVIS --save-json results/metrics.json
```

### Metrics

| Metric | Description | Reference |
|--------|-------------|-----------|
| **JM** (IoU Mean) | Average IoU between predicted and GT masks | VGGT4D |
| **JR** (IoU Recall) | Fraction of frames with IoU ≥ 0.5 | VGGT4D |
| **PSNR** | Peak Signal-to-Noise Ratio of inpainted frames | ProPainter |
| **SSIM** | Structural Similarity Index of inpainted frames | ProPainter |

## Project Structure

```
├── configs/
│   └── default.yaml           # Pipeline configuration
├── data/
│   └── DAVIS/                 # DAVIS 2017 dataset
├── src/
│   ├── mask_extraction.py     # YOLOv8-Seg + optical flow + dilation
│   ├── inpainting.py          # Temporal propagation + cv2.inpaint
│   ├── evaluation.py          # JM, JR, PSNR, SSIM metrics
│   └── pipeline.py            # Main pipeline orchestrator
├── run.py                     # CLI entry point
├── evaluate.py                # Standalone evaluation script
├── environment.yml            # Conda environment definition
└── README.md
```

## Configuration

Key settings in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temporal_propagation.enabled` | `true` | **Core switch** — enable/disable temporal propagation |
| `temporal_propagation.window_size` | `5` | Number of frames to look ahead/behind |
| `mask.target_classes` | `[0]` | COCO class IDs (0=person) |
| `dynamic_filter.motion_threshold` | `2.0` | Pixel displacement threshold for static filtering |
| `dilation.kernel_size` | `15` | Morphological dilation kernel size |
| `spatial_inpaint.method` | `telea` | `telea` or `ns` (Navier-Stokes) |

## Dependencies

- Python 3.10
- OpenCV ≥ 4.8
- Ultralytics (YOLOv8) ≥ 8.1
- PyTorch ≥ 2.1
- scikit-image ≥ 0.21
- NumPy, PyYAML, tqdm, Pillow

## Results

Evaluated on DAVIS 2017 validation sequences (bmx-trees, tennis) using an NVIDIA A40 GPU.
`PSNR_masked` / `SSIM_masked` are computed only inside the inpainted region (masked bounding box), giving a fairer measure of inpainting quality unaffected by the large unchanged background.

### DAVIS 2017 — Ablation Study

| Condition | Sequence | JM ↑ | JR ↑ | PSNR ↑ | SSIM ↑ | PSNR_masked ↑ | SSIM_masked ↑ |
|-----------|----------|-------|-------|--------|--------|--------------|--------------|
| **Spatial only** | bmx-trees | 0.2684 | 0.0125 | 49.56 | 0.9817 | 13.99 | 0.4929 |
| **Spatial only** | tennis | 0.4924 | 0.4000 | 22.58 | 0.9471 | 14.52 | 0.6536 |
| **Spatial only** | **Average** | **0.3804** | **0.2063** | **36.07** | **0.9644** | **14.26** | **0.5733** |
| Temporal (no align) | bmx-trees | 0.2684 | 0.0125 | 49.35 | 0.9780 | 13.69 | 0.4080 |
| Temporal (no align) | tennis | 0.4924 | 0.4000 | 22.37 | 0.9456 | 14.31 | 0.6437 |
| Temporal (no align) | **Average** | **0.3804** | **0.2063** | **35.86** | **0.9618** | **14.00** | **0.5259** |
| **Temporal + Flow Align** | bmx-trees | 0.2684 | 0.0125 | **51.51** | **0.9852** | **16.84** | **0.6054** |
| **Temporal + Flow Align** | tennis | 0.4924 | 0.4000 | **22.54** | **0.9471** | **14.48** | **0.6545** |
| **Temporal + Flow Align** | **Average** | **0.3804** | **0.2063** | **37.03** | **0.9662** | **15.66** | **0.6300** |

### Analysis

**Why temporal (no align) underperforms spatial-only:**
Direct pixel copy from neighbour frames without geometric correction introduces two artefacts in sequences with camera motion (bmx-trees has ~84% temporal fill rate):
1. Misaligned textures — the background content is correct but at the wrong position, creating sharp but wrong-position edges that are penalised more by PSNR (L2) than smooth `cv2.inpaint` blending.
2. Ghost objects — neighbour masks are inconsistent frame-to-frame; pixels marked "clean" in a neighbour may still contain partial foreground, importing ghost artefacts.

**Why temporal + optical-flow alignment is the best:**
Before copying each neighbour frame's pixels, Farneback dense optical flow warps the neighbour to the current frame's coordinate system. This eliminates the misalignment, allowing borrowed pixels to land at geometrically correct positions. The improvement is most dramatic for bmx-trees (significant camera motion): **masked PSNR improves +3.15 dB** vs no-align (16.84 vs 13.69). For tennis (mostly static camera), the gain is smaller (+0.17 dB masked) as alignment matters less when global motion is small.

### Wild Video

Self-recorded footage (6 clips: ride1-3, run1-3) at HKUST(GZ) campus, processed using `slurm_wild.sh`.

| Sequence | Frames | Output |
|----------|--------|--------|
| ride1 | 146 | `results/wild_video/ride1/output.mp4` |
| ride2 | 84 | `results/wild_video/ride2/output.mp4` |
| ride3 | 157 | `results/wild_video/ride3/output.mp4` |
| run1 | 111 | `results/wild_video/run1/output.mp4` |
| run2 | 128 | `results/wild_video/run2/output.mp4` |
| run3 | 134 | `results/wild_video/run3/output.mp4` |

Each sequence folder contains:
```
results/wild_video/<seq>/
├── frames/         # Inpainted frames (PNG)
├── masks/          # Predicted binary masks (PNG)
├── visualization/
└── output.mp4      # Final inpainted video
```

*(Quantitative metrics not available for Wild Video — no ground-truth annotations.)*
