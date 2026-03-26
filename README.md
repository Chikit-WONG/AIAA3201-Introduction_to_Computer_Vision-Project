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

Download the DAVIS 2017 Semi-supervised TrainVal (480p) and extract to `data/DAVIS/`:

```
data/DAVIS/
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

### Custom Config

```bash
python run.py --davis --config configs/custom.yaml
```

## Evaluation

```bash
# Evaluate all processed sequences
python evaluate.py --pred results --davis-root data/DAVIS

# Save results as JSON
python evaluate.py --pred results --davis-root data/DAVIS --save-json results/metrics.json
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

### Temporal Propagation ON (default)

| Sequence | JM | JR | PSNR | SSIM |
|----------|------|------|-------|--------|
| bmx-trees | 0.2671 | 0.0125 | 49.44 | 0.9784 |
| tennis | 0.4922 | 0.4000 | 22.37 | 0.9457 |
| **AVERAGE** | **0.3796** | **0.2063** | **35.91** | **0.9621** |

### Temporal Propagation OFF (spatial-only ablation)

| Sequence | JM | JR | PSNR | SSIM |
|----------|------|------|-------|--------|
| bmx-trees | 0.2671 | 0.0125 | 49.62 | 0.9819 |
| tennis | 0.4922 | 0.4000 | 22.58 | 0.9473 |
| **AVERAGE** | **0.3796** | **0.2063** | **36.10** | **0.9646** |
