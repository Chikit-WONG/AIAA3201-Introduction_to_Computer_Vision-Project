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

This script clones the external repositories used by Part 3 into the exact locations expected by the configs:

- `SAM 3` -> `part3/external/repository/sam3`
- `DiffuEraser` -> `part3/external/repository/DiffuEraser`
- `ROSE` -> `part3/external/repository/ROSE`
- `ProPainter` -> `part2/external/ProPainter`

Repository URLs:

- `SAM 3`: `https://github.com/facebookresearch/sam3.git`
- `DiffuEraser`: `https://github.com/lixiaowen-xw/DiffuEraser.git`
- `ROSE`: `https://github.com/Kunbyte-AI/ROSE.git`
- `ProPainter`: `https://github.com/sczhou/ProPainter.git`

If you prefer manual cloning, run:

```bash
git clone https://github.com/facebookresearch/sam3.git \
  part3/external/repository/sam3

git clone https://github.com/lixiaowen-xw/DiffuEraser.git \
  part3/external/repository/DiffuEraser

git clone https://github.com/Kunbyte-AI/ROSE.git \
  part3/external/repository/ROSE

git clone https://github.com/sczhou/ProPainter.git \
  part2/external/ProPainter
```

### 3. Prepare Model Checkpoints

Automatic download script:

```bash
bash scripts/download_models.sh
```

The default model root is the relative path `models/` under `part3/`.
The script uses **ModelScope by default** for the weights whose ModelScope mirrors are explicitly confirmed in this project setup.

Model checkpoints are configured by relative paths in the YAML files. They are expected under:

- `models/sam3`
- `models/sam3.1`
- `models/diffuEraser`
- `models/sd-vae-ft-mse`
- `models/stable-diffusion-v1-5`
- `models/Wan2.1-Fun-1.3B-InP`
- `models/ROSE_transformer`

Model sources and download targets:

| Component | Source URL | Download-to Directory | Notes |
| --- | --- | --- | --- |
| `SAM 3` checkpoint bundle | `https://huggingface.co/facebook/sam3` | `models/sam3` | Upstream-only in this setup. Current config uses `sam3.pt`. |
| `SAM 3.1` checkpoint bundle | `https://huggingface.co/facebook/sam3.1` | `models/sam3.1` | Upstream-only in this setup. Current config uses `sam3.1_multiplex.pt`. |
| `DiffuEraser` weights | `https://www.modelscope.cn/models/xingzi/diffuEraser` | `models/diffuEraser` | Downloaded by `scripts/download_models.sh`. |
| `sd-vae-ft-mse` | `https://huggingface.co/stabilityai/sd-vae-ft-mse` | `models/sd-vae-ft-mse` | Used by `DiffuEraser`. Manual upstream download. |
| `stable-diffusion-v1-5` base model | `https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5` | `models/stable-diffusion-v1-5` | Used by `DiffuEraser`. Manual upstream download. |
| `Wan2.1-Fun-1.3B-InP` base model | `https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP` | `models/Wan2.1-Fun-1.3B-InP` | Downloaded by `scripts/download_models.sh`. Large download. |
| `ROSE` transformer weights | `https://huggingface.co/Kunbyte/ROSE` | `models/ROSE_transformer` | Upstream-only in this setup. Finetuned transformer for `ROSE`. |

Manual download commands:

```bash
pip install modelscope

python - <<'PY'
from modelscope import snapshot_download
snapshot_download('xingzi/diffuEraser', local_dir='models/diffuEraser')
snapshot_download('PAI/Wan2.1-Fun-1.3B-InP', local_dir='models/Wan2.1-Fun-1.3B-InP')
PY
```

Manual upstream downloads for the remaining weights:

```bash
hf auth login

hf download facebook/sam3 --local-dir models/sam3

hf download facebook/sam3.1 --local-dir models/sam3.1

hf download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae-ft-mse

hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir models/stable-diffusion-v1-5

hf download Kunbyte/ROSE --local-dir models/ROSE_transformer
```

Notes:

- Default source is ModelScope where this project has a confirmed ModelScope mirror.
- `facebook/sam3` and `facebook/sam3.1` require access approval on Hugging Face first.
- `stable-diffusion-v1-5` and `Wan2.1-Fun-1.3B-InP` are large repositories, so make sure enough disk space is available before downloading.
- The `ROSE` configs in this project point to a dedicated Python interpreter through `rose.python_bin`. The model files above are still downloaded into the shared model root.

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

The helper script [scripts/download_models.sh](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/scripts/download_models.sh) is provided for model downloads.

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
