# Part 3

Part 3 implements the final video object removal pipeline for the course project. The pipeline uses `SAM 3` or `SAM 3.1` for video mask generation, then applies one of two inpainting backends:

- `DiffuEraser`
- `ROSE`

It also supports a side-effect-aware mask expansion stage for shadows, reflections, mirror traces, and residual artifacts.

## Goal

The goal of Part 3 is to remove people from videos while improving over the Part 2 baseline by:

- upgrading the segmenter from `SAM 2` to `SAM 3` / `SAM 3.1`
- adding diffusion-based inpainting backends
- explicitly handling side effects around the removed object

## Implemented Methods

The main entry point is [run_part3.py](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/run_part3.py). It currently supports:

- `part2_sam2_propainter`
- `sam3_propainter`
- `sam3_diffueraser_object`
- `sam3_diffueraser_side_effect`
- `sam3_rose_object`
- `sam3_rose_side_effect`

In practice, the final Part 3 experiments focus on:

- `sam3_diffueraser_object`
- `sam3_diffueraser_side_effect`
- `sam3_rose_object`
- `sam3_rose_side_effect`

The same full pipeline was also run with `SAM 3.1` through dedicated configs.

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate cv2
```

If you prefer manual installation, use:

```bash
conda create -n cv2 python=3.10 -y
conda activate cv2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_part3.txt
```

This matches the environment style used in Part 1 and Part 2. If your local CUDA / PyTorch stack is different, adjust the PyTorch install command accordingly.

### 2. Setup External Dependencies

```bash
bash scripts/setup_external_repos.sh
```

This prepares the external repositories used by Part 3:

- `SAM 3`
- `DiffuEraser`
- `ROSE`
- `ProPainter` reuse path from Part 2

### 3. Prepare Model Checkpoints

Model checkpoints are configured by absolute paths in the YAML files. In the current environment they are expected under:

- `/hpc2hdd/home/ckwong627/workdir/models/sam3`
- `/hpc2hdd/home/ckwong627/workdir/models/sam3.1`
- `/hpc2hdd/home/ckwong627/workdir/models/diffuEraser`
- `/hpc2hdd/home/ckwong627/workdir/models/sd-vae-ft-mse`
- `/hpc2hdd/home/ckwong627/workdir/models/Wan2.1-Fun-1.3B-InP`
- `/hpc2hdd/home/ckwong627/workdir/models/ROSE_transformer`

## Data

### DAVIS

DAVIS is used for mask evaluation on:

- `bmx-trees`
- `tennis`

Ground-truth annotations are read from:

- `../data/DAVIS`

### Wild Video

Paired wild videos are used for restoration quality evaluation:

- `Wild_Video1`
- `Wild_Video2`

Input and ground truth are configured in [configs/default.yaml](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/configs/default.yaml):

- input: `../data/Wild_Video/input_with_person`
- clean GT: `../data/Wild_Video/clean_gt_no_person`

Clean ground truth is used only for evaluation, not for inference.

## Directory Layout

Key files and folders:

- [configs/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/configs)
  - pipeline configs for default, full runs, and ablation runs
- [src/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/src)
  - pipeline implementation, wrappers, metrics, and utility code
- [slurm_scripts/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts)
  - SLURM scripts for smoke tests, method runs, and evaluation
- [outputs_ablation/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_ablation)
  - `SAM 3` vs `SAM 3.1` smoke ablation results
- [outputs_full/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_full)
  - final full-run outputs and summary tables
- [plan/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/plan)
  - planning documents and execution specifications

## External Dependencies

Part 3 depends on:

- `SAM 3` repository under `external/repository/sam3`
- `DiffuEraser` under `external/repository/DiffuEraser`
- `ROSE` under `external/repository/ROSE`
- `ProPainter` reused from `../part2/external/ProPainter`

The helper script [scripts/setup_external_repos.sh](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/scripts/setup_external_repos.sh) is provided for repository setup.

This project was developed in an HPC environment and some backends may require dedicated environments or backend-specific Python binaries. For example, the full `ROSE` configs point to a dedicated interpreter through `rose.python_bin`.

## Running the Pipeline

### Local Run

Example:

```bash
python run_part3.py \
  --config configs/sam3_full.yaml \
  --method sam3_diffueraser_side_effect \
  --sequence Wild_Video1 \
  --input ../data/Wild_Video/input_with_person/Wild_Video1.mp4
```

### SLURM Run

The preferred HPC entry points are:

- [slurm_scripts/run_part3_method.slurm](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts/run_part3_method.slurm)
- [slurm_scripts/eval_part3_metrics.slurm](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts/eval_part3_metrics.slurm)

Example:

```bash
CONFIG=configs/sam3_full.yaml \
METHOD=sam3_rose_side_effect \
SEQUENCE=Wild_Video1 \
INPUT_VIDEO=../data/Wild_Video/input_with_person/Wild_Video1.mp4 \
sbatch slurm_scripts/run_part3_method.slurm
```

## Evaluation

Use [evaluate_part3.py](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/evaluate_part3.py).

### DAVIS

DAVIS evaluation reports:

- `JM`
- `JR`

Example:

```bash
python evaluate_part3.py \
  --config configs/sam3_full.yaml \
  --method sam3_diffueraser_object \
  --evaluate-davis \
  --sequence bmx-trees tennis \
  --metrics-tag sam3_diffueraser_object
```

### Wild Video

Wild-video evaluation reports:

- `PSNR`
- `SSIM`

Example:

```bash
python evaluate_part3.py \
  --config configs/sam3_full.yaml \
  --method sam3_rose_side_effect \
  --evaluate-wild \
  --no-align \
  --metrics-tag sam3_rose_side_effect
```

## Output Structure

Important output roots:

- `outputs/`
  - local smoke and intermediate runs
- `outputs_ablation/`
  - `SAM 3` vs `SAM 3.1` smoke ablation
- `outputs_full/`
  - final full experiments

Inside `outputs_full/<variant>/`, the most important folders are:

- `videos/`
- `masks/`
- `metrics/`

Final summary tables are stored in:

- [outputs_full/summary/part3_full_summary.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_full/summary/part3_full_summary.md)
- [outputs_full/summary/part3_results_table.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_full/summary/part3_results_table.md)

Chinese versions are also provided in the same folder.

## Final Results

Summary of the final table:

- Best DAVIS mask metrics: `sam3.1` variants
- Best wild-video `PSNR`: `sam3_diffueraser_side_effect`
- Best wild-video `SSIM`: `sam3_rose_side_effect`
- Best balanced final recommendation: `sam3.1_rose_side_effect`

Current final summary:

| variant | method | avg_davis_JM | avg_davis_JR | avg_wild_PSNR | avg_wild_SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| sam3 | sam3_diffueraser_object | 0.6953 | 0.8625 | 14.3225 | 0.2270 |
| sam3 | sam3_diffueraser_side_effect | 0.6953 | 0.8625 | **14.3236** | 0.2271 |
| sam3 | sam3_rose_object | 0.6953 | 0.8625 | 14.2992 | 0.2294 |
| sam3 | sam3_rose_side_effect | 0.6953 | 0.8625 | 14.3121 | **0.2303** |
| sam3.1 | sam3_diffueraser_object | **0.6969** | **0.8750** | 14.3219 | 0.2270 |
| sam3.1 | sam3_diffueraser_side_effect | **0.6969** | **0.8750** | 14.3231 | 0.2271 |
| sam3.1 | sam3_rose_object | **0.6969** | **0.8750** | 14.3081 | 0.2302 |
| sam3.1 | sam3_rose_side_effect | **0.6969** | **0.8750** | 14.3179 | 0.2302 |

## Notes

- DAVIS metrics are identical across methods within the same variant because DAVIS evaluation reuses the same `object_mask` outputs.
- `PSNR` is a pixel-wise reconstruction metric.
- `SSIM` is a local structural similarity metric.
- Some legacy smoke-check folders such as `wild_video1_short33_check` and `wild_video1_short33_chunked_check` are kept for debugging and validation history; they are not the final full-run results.
