# Part 3 Final Execution Specification

## Summary

This document is the final execution specification for Part 3. It is derived from the confirmed inputs below and is intended to guide implementation without replacing any existing planning documents:

- `part3/plan/create_part3.md`
- `part3/plan/part3_codex_execution_plan.md`
- `part3/plan/Notes_for_Attention.md`
- `part3/plan/Slurm_Partition_Selection_Strategy.md`
- `data/Wild_Video/README.md`

Part 3 extends the earlier project stages with a stronger focus on removing object-induced side effects in video, especially shadows, reflections, mirror images, and residual traces. The main technical direction remains `SAM 3 + DiffuEraser + ROSE`, with `part2` used as the implementation reference and baseline source.

This specification defines what the implementation must produce under `part3/`, how it should be run on the SLURM cluster, which methods must be prioritized, how evaluation must be separated between DAVIS and paired wild video, and which fallbacks are allowed or forbidden.

## Environment and HPC Constraints

- GPU jobs must run on compute nodes. Do not run GPU inference directly on the login node.
- The default first choice for smoke tests and environment checks is the `debug` partition.
- If a formal job is also expected to fit within the `debug` limits, it should still be considered for `debug` first rather than being reserved only for smoke tests.
- If `debug` is clearly disadvantaged by live queue conditions or user-level QOS limits, switch partitions according to `part3/plan/Slurm_Partition_Selection_Strategy.md`.
- Partition selection must follow this fixed order:
  - first check whether the job fits `debug`
  - then search lower-priority shared partitions
  - then medium-priority exclusive partitions
  - then high-priority emergency partitions
- Do not default to `long_cpu` or `long_gpu`. Use them only when the runtime truly requires the longer limit or the live queue is materially better.
- Do not default to A800 if A40 is sufficient.
- Do not default to GPU if the task can be completed on CPU.
- Short formal jobs that fit the `debug` limits should still prefer `debug` unless real-time queue pressure makes another suitable partition clearly better.
- The implementation should assume that `debug` is free but constrained by both runtime and QOS. It must not assume unlimited parallelism on `debug`.

### Asset Download Rules

- If extra models need to be downloaded, create a dedicated subdirectory under `/hpc2hdd/home/ckwong627/workdir/models` first.
- Download with:

```bash
hf download <repo-or-file> --local-dir /hpc2hdd/home/ckwong627/workdir/models/<target_dir>
```

- The same rule applies to any additional dataset download requested by the implementation.
- Any download step must record:
  - what was downloaded
  - approximate storage usage
  - approximate runtime resource requirements
- Large checkpoints must not be committed into the repository.

## Implementation Changes

- Treat `part3/plan/Notes_for_Attention.md` as an official requirement source rather than an unverified note.
- Keep `part3` as an independent implementation area. Do not complete Part 3 by silently modifying `part2` behavior.
- Reuse the design patterns of `part2`, especially CLI organization, pipeline orchestration style, evaluation style, and output conventions, while keeping `part1/` and `part2/` intact.
- Preserve the main Part 3 method story:
  - baseline comparison method: `part2_sam2_propainter`
  - key method to establish diffusion-based object removal: `sam3_diffueraser_object`
  - main proposed methods: `sam3_diffueraser_side_effect`, `sam3_rose_object`, `sam3_rose_side_effect`
- The final Part 3 report method should be chosen from the ROSE-based variants unless there is an explicit documented reason not to do so.
- Do not silently fall back from SAM 3 to SAM 2 when SAM 3 checkpoints are unavailable.
- The only allowed fallback for missing SAM 3 checkpoints is an explicit existing-mask path provided by the user or CLI.

### Data and Evaluation Boundaries

- DAVIS should initially focus on `bmx-trees` and `tennis`.
- Wild paired evaluation should use the current `data/Wild_Video` videos. The repository note only confirms that these are the new wild videos and should be used; it does not establish a richer schema than the existing plan documents.
- Clean wild ground truth under `data/Wild_Video/clean_gt_no_person` is for evaluation only.
- Clean wild ground truth must not be used during mask generation, side-effect mask expansion, prompt construction, or video inpainting inference.
- DAVIS evaluation is mask-centric:
  - compute `JM`
  - compute `JR`
- Paired wild video evaluation is video-quality-centric:
  - compute `PSNR`
  - compute `SSIM`

## Public Interfaces / CLI / Config

### `run_part3.py`

- Acts as the main Part 3 execution entry point.
- Organizes execution by sequence, method, prompt, and mask source.
- Supports configuration defaults plus CLI overrides.
- On the SLURM cluster, GPU-related usage examples should prefer `sbatch`.
- Direct `python` GPU execution should only be documented for non-cluster environments or when already running inside an allocated compute node.

### `evaluate_part3.py`

- Evaluates DAVIS using `JM` and `JR` only.
- Evaluates paired wild videos using `PSNR` and `SSIM` only.
- May optionally align predicted and ground-truth wild frames before metric computation.
- Must document that inference-time use of clean GT is prohibited.

### `SAM3MaskGenerator`

- Responsible for prompt-driven SAM 3 video mask generation.
- Must fail with a clear error if the SAM 3 checkpoint is unavailable.
- Must not replace SAM 3 with SAM 2 silently.
- May allow an explicit existing-mask fallback only when requested by configuration or CLI.

### `build_side_effect_mask(...)`

- Builds side-effect-aware masks from object masks.
- Must enable morphological expansion by default.
- Must enable downward shadow expansion by default.
- May optionally support reflection ROIs and mirror-axis heuristics through configuration.

### `DiffuEraserRunner`

- Wraps DiffuEraser execution.
- Must save stdout and stderr logs.
- Must report a helpful error if the upstream CLI shape does not match expected arguments.
- Must not fail silently.

### `ROSERunner`

- Wraps ROSE execution.
- Version 1 must use padding as the default strategy to satisfy the `16n + 1` frame-length constraint.
- Chunking may be added later as an optimization, but it is not the default v1 behavior.
- Must save stdout and stderr logs.

### Default Configuration

- Initial DAVIS sequences: `bmx-trees`, `tennis`
- Wild video roots:
  - `data/Wild_Video/input_with_person`
  - `data/Wild_Video/clean_gt_no_person`
- Smoke tests should default to `debug`.
- Formal jobs should follow the ordered partition strategy from `part3/plan/Slurm_Partition_Selection_Strategy.md`.

## Test Plan

### File Checks

- `part3/plan/spec_part3_execution_plan.md` exists.
- `part3/plan/spec_part3_execution_plan_CN.md` exists.
- Both files are UTF-8 encoded.

### Structure Checks

- Both documents contain the same section structure.
- The Chinese version is a full equivalent translation, not a summary.

### Content Checks

- Both documents explicitly reference:
  - `Notes_for_Attention.md`
  - `Slurm_Partition_Selection_Strategy.md`
- Both documents state that `debug` is not only for smoke tests and may also be used for formal jobs that fit its limits.
- Both documents define the partition escalation order and mention queue/QOS-based switching conditions.
- Both documents define the asset download directory and resource-recording requirement.
- Both documents state that wild clean GT is for evaluation only.
- Both documents define `part2` reuse boundaries and the rule that `part3` remains an independent implementation area.

### Repository Consistency Checks

- Paths, class names, method names, and script names match the repository plan and current naming.
- The old assumption that `Notes_for_Attention.md` was missing is not carried forward.
- `data/Wild_Video/README.md` is not overstated as a complete data specification, since it currently only gives a minimal confirmation to use the new wild videos.

## Assumptions and Defaults

- This English document is the source draft; the Chinese document must remain fully equivalent.
- These files are final execution specifications and do not replace:
  - `create_part3.md`
  - `part3_codex_execution_plan.md`
  - `Notes_for_Attention.md`
  - `Slurm_Partition_Selection_Strategy.md`
- When earlier planning text conflicts with `Notes_for_Attention.md`, the confirmed notes take precedence.
- Wild-data assumptions rely mainly on the repository layout and the main Part 3 planning documents because `data/Wild_Video/README.md` currently contains only a minimal instruction.
- This specification defines what must be written and kept in the plan documents; it does not itself implement the Part 3 code pipeline.
