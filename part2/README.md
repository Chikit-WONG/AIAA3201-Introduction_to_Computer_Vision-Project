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

## References

- **VGGT4D**: Hu et al., "Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction", arXiv:2511.19971
- **SAM 2**: Ravi et al., "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714
- **ProPainter**: Zhou et al., "ProPainter: Improving Propagation and Transformer for Video Inpainting", ICCV 2023
- **DAVIS 2017**: Pont-Tuset et al., "The 2017 DAVIS Challenge on Video Object Segmentation", arXiv:1704.00675
