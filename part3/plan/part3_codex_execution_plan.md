# AIAA3201 Project 3 Part 3 Execution Plan

**Project:** Video Object Removal & Inpainting  
**Branch:** `ckw`  
**Target folder to implement:** `part3/`  
**Main idea:** Use **SAM 3 + DiffuEraser + ROSE** to improve Part 2 by removing not only the foreground dynamic object, but also object-induced side effects such as **shadow, reflection, mirror image, and residual traces**.

---

## 0. Important Context for Codex

We have already completed:

- `part1/`: hand-crafted baseline.
- `part2/`: SOTA reproduction, including SAM 2 / VGGT4D style mask extraction and ProPainter-style video inpainting.

Now we need to complete **Part 3: Exploration / Optimization / Extension**.

The Part 3 plan is:

1. Use **SAM 3** to generate better concept-guided masks for dynamic objects.
2. Use **DiffuEraser** as a diffusion-based video inpainting method.
3. Use **ROSE** as a side-effect-aware video object removal method, especially for:
   - shadows,
   - reflections,
   - mirror images,
   - residual side effects caused by the removed object.
4. Evaluate:
   - **DAVIS / mandatory sample sequences:** calculate **JM** and **JR** only, because we have mask ground truth but no clean inpainted background ground truth.
   - **New paired wild video:** calculate **PSNR** and **SSIM** only, because we have a clean background video after removing the moving person, but we do not have manually annotated mask ground truth.

Very important:

> Do **not** use the clean wild-video ground truth during inference.  
> The clean wild-video ground truth can only be used for evaluation.

---

## 1. Official External Repositories and Papers

### 1.1 SAM 3

- GitHub: https://github.com/facebookresearch/sam3 /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/external/repository/sam3
- arXiv paper: https://arxiv.org/abs/2511.16719 /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/external/paper/Carion 等 - 2026 - SAM 3 Segment Anything with Concepts.pdf
- Official page: https://ai.meta.com/sam3/

Purpose in this project:

> Use SAM 3 for text/concept-guided dynamic object segmentation in video. For example, prompt with `"person"`, `"moving person"`, `"bicycle"`, `"tennis player"`, etc.

Expected output:

- Binary masks per frame.
- Optional visualized overlay video.
- Mask video compatible with DiffuEraser / ROSE / ProPainter.

---

### 1.2 DiffuEraser

- GitHub: https://github.com/lixiaowen-xw/diffueraser /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/external/repository/DiffuEraser
- arXiv paper: https://arxiv.org/abs/2501.10018 /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/external/paper/Li 等 - 2025 - DiffuEraser A Diffusion Model for Video Inpainting.pdf
- Project page: https://lixiaowen-xw.github.io/DiffuEraser-page/

Purpose in this project:

> Replace or complement ProPainter with a diffusion-based video inpainting method. This is mainly used to test whether diffusion-based inpainting gives better content completeness and temporal consistency.

Expected input:

- Input video.
- Binary mask video / mask frames.

Expected output:

- Inpainted video.

---

### 1.3 ROSE

- GitHub: https://github.com/Kunbyte-AI/ROSE /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/external/repository/ROSE
- arXiv paper: https://arxiv.org/abs/2508.18633 /hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/external/paper/Miao 等 - 2025 - ROSE Remove Objects with Side Effects in Videos.pdf
- Project page: https://rose2025-inpaint.github.io/

Purpose in this project:

> Use ROSE for side-effect-aware object removal. ROSE is especially important because our Part 3 focuses on shadow, reflection, and mirror-image removal.

Expected input:

- Input video.
- Mask video.
- Optional prompt.

Expected output:

- Side-effect-aware erased video.

---

## 2. Research Story for Part 3

The report story should be:

```text
Part 1: Traditional method can remove simple objects but produces blurry results.
Part 2: SAM 2 + ProPainter improves object removal but still mainly removes the foreground object.
Part 3: We observe that shadows, reflections, and mirror images often remain after object removal.
        Therefore, we explore SAM 3 + diffusion-based erasing methods, especially ROSE,
        to remove object-induced side effects.
```

The Part 3 research question is:

> Can concept-guided segmentation and diffusion-based erasing improve video object removal, especially when the target object causes shadows, reflections, or mirror effects?

---

## 3. Required Repository Structure

Create the following folder structure:

```text
part3/
├── README.md (English version, have link point to Chinese version)
├── README-CN.md (Chinese version, have link point to English version)
├── requirements_part3.txt
├── run_part3.py
├── evaluate_part3.py
├── make_videos_zip.py
├── configs/
│   ├── default.yaml
│   ├── davis_bmx_trees.yaml
│   ├── davis_tennis.yaml
│   └── wild_paired.yaml
├── scripts/
│   ├── setup_external_repos.sh
│   ├── run_sam3_masks.sh
│   ├── run_diffueraser.sh
│   ├── run_rose.sh
│   ├── run_all_part3.sh
│   ├── evaluate_all.sh
│   └── make_qualitative_figures.sh
├── src/
│   ├── __init__.py
│   ├── io_utils.py
│   ├── video_utils.py
│   ├── mask_utils.py
│   ├── sam3_wrapper.py
│   ├── side_effect_mask.py
│   ├── diffueraser_wrapper.py
│   ├── rose_wrapper.py
│   ├── metrics_mask.py
│   ├── metrics_video.py
│   ├── alignment.py
│   ├── visualization.py
│   └── report_tables.py
├── external/
│   ├── paper/
│   ├── repository/
│       ├── README.md (English version, have link point to Chinese version)
|  	  	├── README-CN.md (Chinese version, have link point to English version)
│       ├── sam3/              # cloned manually or by setup script
│       ├── diffueraser/       # cloned manually or by setup script
│       └── ROSE/              # cloned manually or by setup script
├── outputs/
│   ├── masks/
│   ├── videos/
│   ├── metrics/
│   ├── figures/
│   └── logs/
└── slurm/
    ├── run_sam3.slurm
    ├── run_diffueraser.slurm
    ├── run_rose.slurm
    └── run_all_part3.slurm
```

Do not break existing `part1/` and `part2/`.

---

## 4. Data Layout Requirements

### 4.1 DAVIS / Sample Data

DAVIS dataset maybe this structure:

```text
/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/data/DAVIS/
├── JPEGImages/480p/
│   ├── bmx-trees/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   └── tennis/
│       ├── 00000.jpg
│       ├── 00001.jpg
│       └── ...
├── ImageSets/
└── Annotations/480p/
    ├── bmx-trees/
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── ...
    └── tennis/
        ├── 00000.png
        ├── 00001.png
        └── ...
```

For DAVIS and mandatory sample sequences:

- Evaluate **mask quality only**:
  - `JM`: mean IoU.
  - `JR`: recall of IoU above threshold, default threshold = 0.5.
- Do **not** compute PSNR / SSIM unless there is a clean inpainted-background ground truth.

---

### 4.2 Paired Wild Video

Wild video dataset maybe this structure this structure:

```text
/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/data/Wild_Video/
├── README.md
├── input_with_person/
├── clean_gt_no_person/
└── metadata.yaml
```

`metadata.yaml` should include:

```yaml
name: wild_side_effect
input_videos: input_with_person/
clean_gt_videos: clean_gt_no_person/
fps: 30
resolution: [height, width]
camera: fixed
has_mask_gt: false
has_clean_gt: true
notes: "Paired wild video. Input has moving person; clean_gt has person removed / absent."
```

For wild paired video:

- Evaluate **video quality only**:
  - PSNR.
  - SSIM.
- Do **not** compute JM / JR because no human-labeled mask ground truth exists.

Important implementation detail:

> The wild videos should be aligned before PSNR / SSIM calculation. If the camera is fixed and frames are already aligned, direct comparison is acceptable. If there is slight shift, use ECC or homography alignment.

---

## 5. Methods to Implement and Compare

Implement at least these methods:

| Method ID | Mask Source | Side-effect Mask | Inpainting / Erasing | Purpose |
|---|---|---|---|---|
| `part2_sam2_propainter` | existing Part 2 SAM 2 mask | no | ProPainter | strong baseline |
| `sam3_propainter` | SAM 3 | no | ProPainter | isolate mask upgrade effect |
| `sam3_diffueraser_object` | SAM 3 | no | DiffuEraser | test diffusion video inpainting |
| `sam3_diffueraser_side_effect` | SAM 3 | yes | DiffuEraser | test explicit side-effect mask expansion |
| `sam3_rose_object` | SAM 3 | no or minimal | ROSE | test side-effect-aware erasing |
| `sam3_rose_side_effect` | SAM 3 | yes | ROSE | strongest Part 3 method |

If time is limited, prioritize:

1. `part2_sam2_propainter`
2. `sam3_diffueraser_object`
3. `sam3_diffueraser_side_effect`
4. `sam3_rose_object`
5. `sam3_rose_side_effect`

The main proposed method should be:

```text
sam3_rose_side_effect
```

or, if ROSE works better with object-only masks:

```text
sam3_rose_object
```

In the report, choose the best-performing one as the final Part 3 method.

---

## 6. Implementation Phase 1: Setup External Repositories

Create:

```bash
part3/scripts/setup_external_repos.sh
```

The script should:

```bash
#!/usr/bin/env bash
set -e

mkdir -p part3/external
mkdir -p part3/external/repository
cd part3/external/repository

if [ ! -d "sam3" ]; then
  git clone https://github.com/facebookresearch/sam3.git
fi

if [ ! -d "diffueraser" ]; then
  git clone https://github.com/lixiaowen-xw/diffueraser.git
fi

if [ ! -d "ROSE" ]; then
  git clone https://github.com/Kunbyte-AI/ROSE.git
fi

echo "External repositories cloned."
```

I have already manually clone the repositories, so you don't need to clone them again.

Also create:

```text
part3/external/repository/README.md
part3/external/repository/README-CN.md
```

Content should explain:

- External repos are not our own code.
- They are used as third-party methods.
- Model checkpoints should not be committed to GitHub.
- Store large checkpoints outside the repo or under ignored paths.

---

## 7. Implementation Phase 2: SAM 3 Mask Generation

### 7.1 File to implement

Create:

```text
part3/src/sam3_wrapper.py
```

Expected class:

```python
class SAM3MaskGenerator:
    def __init__(self, sam3_dir: str, checkpoint: str | None = None, device: str = "cuda"):
        ...

    def generate_video_masks(
        self,
        input_video: str,
        output_mask_dir: str,
        prompt: str,
        output_overlay_video: str | None = None,
        output_mask_video: str | None = None,
    ) -> dict:
        ...
```

Expected behavior:

1. Read input video.
2. Run SAM 3 video predictor using text prompt.
3. Save binary masks as PNG frames:
   ```text
   part3/outputs/masks/{sequence}/sam3/{prompt}/00000.png
   part3/outputs/masks/{sequence}/sam3/{prompt}/00001.png
   ...
   ```
4. Save optional mask video:
   ```text
   part3/outputs/masks/{sequence}/sam3_mask.mp4
   ```
5. Save optional overlay video:
   ```text
   part3/outputs/masks/{sequence}/sam3_overlay.mp4
   ```
6. Return a dictionary:
   ```python
   {
       "sequence": "...",
       "prompt": "...",
       "num_frames": ...,
       "mask_dir": "...",
       "mask_video": "...",
       "overlay_video": "..."
   }
   ```

### 7.2 Prompts to support

Config should allow prompt per sequence:

```yaml
prompts:
  bmx-trees: "person riding bicycle"
  tennis: "tennis player"
  wild_side_effect: "person"
```

Also allow CLI override:

```bash
python part3/run_part3.py \
  --sequence wild_side_effect \
  --input part3/data/wild/input_with_person.mp4 \
  --prompt "person" \
  --mask_source sam3
```

### 7.3 If SAM 3 checkpoint is unavailable

Do not silently replace SAM 3 with SAM 2.

Instead:

1. Print a clear error:
   ```text
   SAM 3 checkpoint is unavailable. Please request access and set SAM3_CHECKPOINT or SAM3_MODEL_ID.
   ```
2. Allow a fallback only if explicitly requested:
   ```bash
   --allow_existing_masks --existing_mask_dir path/to/masks
   ```

This keeps the experiment honest.

---

## 8. Implementation Phase 3: Side-Effect Mask Construction

### 8.1 Goal

Object-only masks may remove the person but leave:

- shadow on the ground,
- reflection on glass/floor/water,
- mirror person in a mirror,
- lighting distortion,
- residual blurred body boundary.

So implement side-effect mask expansion.

Create:

```text
part3/src/side_effect_mask.py
```

Expected functions:

```python
def build_side_effect_mask(
    frame,
    object_mask,
    config: dict,
    prev_frame=None,
    next_frame=None,
    prev_mask=None,
    next_mask=None,
):
    ...
```

Return:

```python
{
    "object_mask": object_mask,
    "expanded_mask": expanded_mask,
    "shadow_mask": shadow_mask,
    "reflection_mask": reflection_mask,
    "final_mask": final_mask,
}
```

### 8.2 Simple required components

Implement these first:

#### A. Morphological expansion

```python
expanded = dilate(object_mask, radius=config["dilate_radius"])
```

Default:

```yaml
side_effect:
  enable: true
  dilate_radius: 21
  close_radius: 11
  fill_holes: true
```

This catches object boundaries, motion blur, feet, and nearby shadows.

#### B. Directional shadow expansion

For shadows on the ground, expand the object mask downward:

```python
shifted = shift_mask(object_mask, dx=0, dy=config["shadow_down_shift"])
shadow_region = dilate(shifted, radius=config["shadow_radius"])
```

Default:

```yaml
side_effect:
  shadow_down_shift: 35
  shadow_radius: 45
```

#### C. Optional manual ROI

For mirror/reflection scenes, support polygon ROI in YAML:

```yaml
side_effect:
  reflection_rois:
    - name: mirror_left
      polygon:
        - [100, 50]
        - [300, 50]
        - [300, 420]
        - [100, 420]
```

Inside ROI, we can include reflection candidates using either:

- low-level frame difference,
- temporal motion difference,
- manual ROI union with a shifted/flipped version of object mask,
- or simple ROI mask if reflection is stable and small.

Do not use clean ground truth to generate side-effect masks.

### 8.3 Optional reflection heuristic

If a mirror plane is specified:

```yaml
side_effect:
  mirror:
    enable: true
    axis: vertical
    x_center: 640
    roi_name: mirror_left
```

Then generate approximate mirror mask:

```python
reflection_mask = flip_object_mask_across_vertical_axis(object_mask, x_center)
reflection_mask = reflection_mask & roi_mask
reflection_mask = dilate(reflection_mask, radius=reflection_radius)
```

This is optional but useful for mirror videos.

### 8.4 Save all intermediate masks

For each sequence, save:

```text
part3/outputs/masks/{sequence}/object_mask/
part3/outputs/masks/{sequence}/expanded_mask/
part3/outputs/masks/{sequence}/shadow_mask/
part3/outputs/masks/{sequence}/reflection_mask/
part3/outputs/masks/{sequence}/final_side_effect_mask/
```

Also save:

```text
part3/outputs/masks/{sequence}/mask_debug_grid.mp4
```

The debug video should show:

```text
Original | Object mask | Shadow/reflection mask | Final mask overlay
```

This will be very useful for the report.

---

## 9. Implementation Phase 4: DiffuEraser Wrapper

Create:

```text
part3/src/diffueraser_wrapper.py
```

Expected class:

```python
class DiffuEraserRunner:
    def __init__(self, diffueraser_dir: str, env_name: str | None = None):
        ...

    def run(
        self,
        input_video: str,
        mask_video_or_dir: str,
        output_dir: str,
        extra_args: list[str] | None = None,
    ) -> str:
        ...
```

Implementation:

1. Convert mask frames to DiffuEraser-compatible format.
2. Convert input frames/video if needed.
3. Call DiffuEraser with `subprocess.run`.
4. Save the output video under:

```text
part3/outputs/videos/{sequence}/sam3_diffueraser_object/output.mp4
part3/outputs/videos/{sequence}/sam3_diffueraser_side_effect/output.mp4
```

Because external repos may change their CLI, implement the wrapper defensively:

- Check that `run_diffueraser.py` exists.
- If expected CLI arguments differ, print a helpful error.
- Do not crash silently.
- Save stdout/stderr logs to:

```text
part3/outputs/logs/{sequence}/diffueraser_*.log
```

---

## 10. Implementation Phase 5: ROSE Wrapper

Create:

```text
part3/src/rose_wrapper.py
```

Expected class:

```python
class ROSERunner:
    def __init__(self, rose_dir: str, env_name: str | None = None):
        ...

    def run(
        self,
        input_video: str,
        mask_video_or_dir: str,
        output_dir: str,
        prompt: str = "",
        video_length: int | None = None,
        sample_size: tuple[int, int] | None = None,
        extra_args: list[str] | None = None,
    ) -> str:
        ...
```

The official ROSE inference interface is approximately:

```bash
python inference.py \
  --validation_videos PATH_TO_INPUT_VIDEO \
  --validation_masks PATH_TO_MASK_VIDEO \
  --validation_prompts "" \
  --output_dir PATH_TO_OUTPUT_DIR \
  --video_length 17 \
  --sample_size 480 720
```

Important ROSE detail:

> ROSE expects video length to be `16n + 1`.

So implement helper:

```python
def split_video_into_rose_chunks(video_path, mask_path, chunk_size=17, overlap=4):
    ...
```

If input video length is not `16n + 1`, either:

1. Pad the last frames, or
2. Split into overlapping chunks of 17 / 33 / 49 frames, then stitch results.

For the first implementation, simple padding is acceptable. But the script should document the behavior.

Save output:

```text
part3/outputs/videos/{sequence}/sam3_rose_object/output.mp4
part3/outputs/videos/{sequence}/sam3_rose_side_effect/output.mp4
```

Save logs:

```text
part3/outputs/logs/{sequence}/rose_*.log
```

---

## 11. Implementation Phase 6: Evaluation

Create:

```text
part3/evaluate_part3.py
part3/src/metrics_mask.py
part3/src/metrics_video.py
part3/src/alignment.py
part3/src/report_tables.py
```

---

### 11.1 Mask Metrics for DAVIS: JM and JR

Implement in:

```text
part3/src/metrics_mask.py
```

Functions:

```python
def compute_iou(pred_mask, gt_mask) -> float:
    ...

def compute_jm(pred_mask_dir, gt_mask_dir) -> float:
    ...

def compute_jr(pred_mask_dir, gt_mask_dir, threshold: float = 0.5) -> float:
    ...

def evaluate_mask_sequence(pred_mask_dir, gt_mask_dir, threshold: float = 0.5) -> dict:
    return {
        "JM": ...,
        "JR": ...,
        "num_frames": ...,
        "threshold": threshold
    }
```

Important details:

- Convert all masks to binary.
- If DAVIS has multiple object labels, treat all non-zero pixels as foreground unless the config specifies an object ID.
- Resize predicted mask to GT size if needed.
- Align frame names by sorted order.

Output:

```text
part3/outputs/metrics/davis_mask_metrics.csv
part3/outputs/metrics/davis_mask_metrics.json
```

Table columns:

```text
sequence, method, JM, JR, num_frames
```

Methods to evaluate on DAVIS:

- `part2_sam2_propainter` mask if available.
- `sam3_object_mask`.
- `sam3_side_effect_mask` only if it makes sense; note that side-effect mask may not match DAVIS object GT and can reduce IoU, so the main fair comparison should be object masks only.

---

### 11.2 Video Metrics for Wild Paired Video: PSNR and SSIM

Implement in:

```text
part3/src/metrics_video.py
```

Functions:

```python
def compute_psnr(pred_frame, gt_frame) -> float:
    ...

def compute_ssim(pred_frame, gt_frame) -> float:
    ...

def evaluate_video_quality(pred_video, gt_video, align: bool = True) -> dict:
    return {
        "PSNR": ...,
        "SSIM": ...,
        "num_frames": ...,
        "aligned": align
    }
```

Use:

- `skimage.metrics.peak_signal_noise_ratio`
- `skimage.metrics.structural_similarity`

If `skimage` is unavailable, add it to `requirements_part3.txt`.

Important details:

- Match FPS and frame count.
- Resize predicted video to GT resolution if necessary.
- If `--align` is enabled:
  - Use ECC alignment or homography alignment.
  - Implement safe fallback: if alignment fails, log warning and use unaligned frames.
- Do not use GT video for any inference-time mask generation or inpainting.

Output:

```text
part3/outputs/metrics/wild_video_metrics.csv
part3/outputs/metrics/wild_video_metrics.json
```

Table columns:

```text
sequence, method, PSNR, SSIM, num_frames, aligned
```

Methods to evaluate on wild:

- `part2_sam2_propainter`
- `sam3_diffueraser_object`
- `sam3_diffueraser_side_effect`
- `sam3_rose_object`
- `sam3_rose_side_effect`

---

## 12. Implementation Phase 7: Visualization

Create:

```text
part3/src/visualization.py
part3/scripts/make_qualitative_figures.sh
```

Generate these figures:

### Figure A: Part 3 pipeline

```text
Input video
  ↓
SAM 3 concept-guided object mask
  ↓
Side-effect mask expansion
  ↓
DiffuEraser / ROSE
  ↓
Object + shadow/reflection/mirror removed video
```

Save:

```text
part3/outputs/figures/part3_pipeline.png
```

### Figure B: Mask comparison

For DAVIS and wild, create grid:

```text
Original frame | SAM 3 object mask | Side-effect mask | Final mask overlay
```

Save:

```text
part3/outputs/figures/{sequence}_mask_comparison.png
```

### Figure C: Inpainting comparison

For wild:

```text
Input | Part 2 SAM2+ProPainter | SAM3+DiffuEraser | SAM3+ROSE | Clean GT
```

Save:

```text
part3/outputs/figures/wild_inpainting_comparison.png
```

### Figure D: Side-effect failure cases

Create one figure for:

- shadow remains,
- reflection remains,
- mirror person remains,
- object boundary ghost remains.

Save:

```text
part3/outputs/figures/side_effect_failure_cases.png
```

---

## 13. Main CLI Design

Implement:

```text
part3/run_part3.py
```

Example usage:

```bash
# Generate SAM 3 masks only
python part3/run_part3.py \
  --sequence wild_side_effect \
  --input part3/data/wild/input_with_person.mp4 \
  --prompt "person" \
  --stage mask \
  --method sam3

# Generate side-effect masks
python part3/run_part3.py \
  --sequence wild_side_effect \
  --input part3/data/wild/input_with_person.mp4 \
  --mask_dir part3/outputs/masks/wild_side_effect/sam3/person \
  --stage side_effect_mask \
  --config part3/configs/wild_paired.yaml

# Run DiffuEraser object-only
python part3/run_part3.py \
  --sequence wild_side_effect \
  --input part3/data/wild/input_with_person.mp4 \
  --mask_dir part3/outputs/masks/wild_side_effect/object_mask \
  --stage inpaint \
  --method sam3_diffueraser_object

# Run DiffuEraser side-effect mask
python part3/run_part3.py \
  --sequence wild_side_effect \
  --input part3/data/wild/input_with_person.mp4 \
  --mask_dir part3/outputs/masks/wild_side_effect/final_side_effect_mask \
  --stage inpaint \
  --method sam3_diffueraser_side_effect

# Run ROSE object-only
python part3/run_part3.py \
  --sequence wild_side_effect \
  --input part3/data/wild/input_with_person.mp4 \
  --mask_dir part3/outputs/masks/wild_side_effect/object_mask \
  --stage inpaint \
  --method sam3_rose_object

# Run ROSE side-effect mask
python part3/run_part3.py \
  --sequence wild_side_effect \
  --input part3/data/wild/input_with_person.mp4 \
  --mask_dir part3/outputs/masks/wild_side_effect/final_side_effect_mask \
  --stage inpaint \
  --method sam3_rose_side_effect
```

Also support:

```bash
python part3/run_part3.py --config part3/configs/wild_paired.yaml --run_all
```

---

## 14. Evaluation CLI Design

Implement:

```text
part3/evaluate_part3.py
```

Example usage:

```bash
# DAVIS mask evaluation
python part3/evaluate_part3.py \
  --task mask \
  --sequence bmx-trees \
  --pred_mask_dir part3/outputs/masks/bmx-trees/sam3/person_riding_bicycle \
  --gt_mask_dir part3/data/davis/Annotations/480p/bmx-trees \
  --method sam3_object_mask

# Wild video evaluation
python part3/evaluate_part3.py \
  --task video \
  --sequence wild_side_effect \
  --pred_video part3/outputs/videos/wild_side_effect/sam3_rose_side_effect/output.mp4 \
  --gt_video part3/data/wild/clean_gt_no_person.mp4 \
  --method sam3_rose_side_effect \
  --align
```

Also support:

```bash
python part3/evaluate_part3.py --config part3/configs/wild_paired.yaml --evaluate_all
```

---

## 15. Config Examples

### 15.1 `part3/configs/default.yaml`

```yaml
project:
  name: AIAA3201_Project3_Part3
  output_root: part3/outputs

external:
  sam3_dir: part3/external/sam3
  diffueraser_dir: part3/external/diffueraser
  rose_dir: part3/external/ROSE

runtime:
  device: cuda
  fps: null
  max_frames: null
  save_intermediate: true

sam3:
  checkpoint: null
  prompt: "person"
  allow_existing_masks: false

side_effect:
  enable: true
  dilate_radius: 21
  close_radius: 11
  fill_holes: true
  shadow_down_shift: 35
  shadow_radius: 45
  reflection_radius: 25
  reflection_rois: []
  mirror:
    enable: false
    axis: vertical
    x_center: null
    roi_name: null

diffueraser:
  enabled: true
  extra_args: []

rose:
  enabled: true
  video_length: 17
  sample_size: [480, 720]
  prompt: ""

evaluation:
  mask_iou_threshold: 0.5
  video_align: true
```

### 15.2 `part3/configs/wild_paired.yaml`

```yaml
sequence:
  name: wild_side_effect
  input_video: part3/data/wild/input_with_person.mp4
  clean_gt_video: part3/data/wild/clean_gt_no_person.mp4
  has_mask_gt: false
  has_clean_gt: true

sam3:
  prompt: "person"

side_effect:
  enable: true
  dilate_radius: 25
  close_radius: 15
  fill_holes: true
  shadow_down_shift: 40
  shadow_radius: 55
  reflection_radius: 35
  reflection_rois:
    # Fill this manually if the wild video contains mirror/reflection area.
    # Coordinates are [x, y].
    - name: mirror_or_reflection_area
      polygon:
        - [0, 0]
        - [0, 0]
        - [0, 0]
        - [0, 0]
  mirror:
    enable: false
    axis: vertical
    x_center: null
    roi_name: mirror_or_reflection_area

methods:
  - sam3_diffueraser_object
  - sam3_diffueraser_side_effect
  - sam3_rose_object
  - sam3_rose_side_effect

evaluation:
  video_align: true
```

---

## 16. README Requirements

Create:

```text
part3/README.md
```

The README must include:

1. Part 3 objective.
2. Explanation of why Part 3 focuses on side effects:
   - shadow,
   - reflection,
   - mirror image.
3. External method links:
   - SAM 3,
   - DiffuEraser,
   - ROSE.
4. Setup instructions.
5. Data layout.
6. How to run each method.
7. How to evaluate.
8. How to reproduce figures.
9. Known limitations.

README should clearly state:

```text
For DAVIS / sample data, we report JM and JR for mask quality.
For paired wild video, we report PSNR and SSIM for video quality.
We do not compute PSNR/SSIM on DAVIS because it lacks clean inpainted-background ground truth.
We do not compute JM/JR on wild video because it lacks manually annotated mask ground truth.
```

---

## 17. SLURM Scripts

The following content still needs to be revised. For instance, determining the partitions and checking if the time needs to be modified are tasks that you should undertake.

Create SLURM scripts under:

```text
part3/slurm/
```

### 17.1 `run_sam3.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=part3_sam3
#SBATCH --output=part3/outputs/logs/slurm_sam3_%j.out
#SBATCH --error=part3/outputs/logs/slurm_sam3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate sam3

python part3/run_part3.py --config part3/configs/wild_paired.yaml --stage mask --method sam3
```

### 17.2 `run_diffueraser.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=part3_diffueraser
#SBATCH --output=part3/outputs/logs/slurm_diffueraser_%j.out
#SBATCH --error=part3/outputs/logs/slurm_diffueraser_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00

source ~/.bashrc
conda activate diffueraser

python part3/run_part3.py --config part3/configs/wild_paired.yaml --stage inpaint --method sam3_diffueraser_side_effect
```

### 17.3 `run_rose.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=part3_rose
#SBATCH --output=part3/outputs/logs/slurm_rose_%j.out
#SBATCH --error=part3/outputs/logs/slurm_rose_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00

source ~/.bashrc
conda activate rose

python part3/run_part3.py --config part3/configs/wild_paired.yaml --stage inpaint --method sam3_rose_side_effect
```

---

## 18. Expected Final Outputs

After implementation, the following outputs should exist:

```text
part3/outputs/
├── masks/
│   ├── bmx-trees/
│   ├── tennis/
│   └── wild_side_effect/
├── videos/
│   ├── bmx-trees/
│   ├── tennis/
│   └── wild_side_effect/
├── metrics/
│   ├── davis_mask_metrics.csv
│   ├── davis_mask_metrics.json
│   ├── wild_video_metrics.csv
│   └── wild_video_metrics.json
├── figures/
│   ├── part3_pipeline.png
│   ├── wild_inpainting_comparison.png
│   ├── wild_mask_comparison.png
│   └── side_effect_failure_cases.png
└── logs/
```

Also create:

``` bash
videos.zip
```

containing all processed mandatory dataset videos.

---

## 19. Report Tables to Generate

The following is just a preliminary proposal. Once the code and results are finalized, I will review whether any modifications are necessary and then guide you to write the report according to the CVPR template.

### 19.1 DAVIS / Sample Mask Quality Table

```text
Table: Mask quality on DAVIS / mandatory sample sequences

Method | bmx-trees JM | bmx-trees JR | tennis JM | tennis JR | Average JM | Average JR
```

Compare:

- Part 2 SAM 2 mask.
- SAM 3 object mask.
- Optional: SAM 3 side-effect mask, but mark it clearly because side-effect mask may intentionally include regions outside object GT.

---

### 19.2 Wild Paired Video Quality Table

```text
Table: Video quality on paired wild video

Method | Uses side-effect mask? | PSNR ↑ | SSIM ↑
```

Compare:

- Part 2 SAM2 + ProPainter.
- SAM3 + DiffuEraser object mask.
- SAM3 + DiffuEraser side-effect mask.
- SAM3 + ROSE object mask.
- SAM3 + ROSE side-effect mask.

---

### 19.3 Side-Effect Qualitative Table

```text
Table: Side-effect removal analysis

Method | Shadow removal | Reflection removal | Mirror image removal | Temporal consistency | Main failure
```

Use qualitative labels:

```text
Poor / Medium / Good / Best
```

This is acceptable because shadow/reflection/mirror removal may not have ground-truth mask annotation.

---

## 20. Success Criteria

The Part 3 implementation is considered successful if:

1. `part3/` can run independently without breaking `part1/` or `part2/`.
2. SAM 3 masks are generated and saved.
3. At least one DiffuEraser result video is generated.
4. At least one ROSE result video is generated.
5. Wild paired video PSNR and SSIM are computed.
6. DAVIS / sample JM and JR are computed.
7. Qualitative figures clearly show:
   - object removed,
   - shadow/reflection/mirror side effect reduced,
   - comparison with Part 2 baseline.
8. `part3/README.md` and  `part3/README-CN.md` explains how to reproduce everything.

---

## 21. Minimal Viable Part 3

If time or GPU resources are limited, implement this minimum version:

```text
1. SAM 3 object mask generation.
2. Side-effect mask expansion by dilation + downward shadow expansion.
3. DiffuEraser with object mask.
4. DiffuEraser with side-effect mask.
5. ROSE with object mask or side-effect mask.
6. Wild PSNR / SSIM evaluation.
7. DAVIS JM / JR mask evaluation.
8. Qualitative comparison figures.
```

This is enough to tell the intended Part 3 story.

---

## 22. Do Not Do These

Do not:

1. Use wild clean GT to generate masks.
2. Compute PSNR / SSIM on DAVIS original videos as if they were clean background.
3. Compute JM / JR on wild video without mask annotation.
4. Commit model checkpoints or large output videos to GitHub.
5. Overwrite existing Part 1 / Part 2 results.
6. Hide failures. If ROSE or DiffuEraser fails on some videos, save failure cases and discuss them in the report.

---

## 24. Final Reproduction Commands

At the end, these commands should work:

```bash
# 1. Setup external repos
bash part3/scripts/setup_external_repos.sh

# 2. Run all Part 3 experiments
bash part3/scripts/run_all_part3.sh

# 3. Evaluate all results
bash part3/scripts/evaluate_all.sh

# 4. Generate qualitative figures
bash part3/scripts/make_qualitative_figures.sh

# 5. Package videos for submission
python part3/make_videos_zip.py \
  --input_dir part3/outputs/videos \
  --output videos.zip
```

---

## 25. Final Notes for Codex

When implementing:

- Prefer clean, modular Python code.
- Add helpful error messages.
- Save intermediate outputs.
- Make scripts reproducible.
- Keep paths configurable.
- Do not assume checkpoints are already present.
- Do not make silent fallbacks that would make the experiment dishonest.
- Use existing Part 2 outputs when useful, but keep Part 3 results separate.
- Write logs for every external model call.
- Make final metrics tables automatically generated as `.csv`, `.json`, and optionally `.md`.

The final Part 3 should clearly answer:

> Compared with Part 2, can SAM 3 + DiffuEraser / ROSE better remove dynamic objects and their side effects such as shadows, reflections, and mirror images?
