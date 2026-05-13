# Part 3 Output Layout

The formal output policy is:

- keep all final experiment outputs under `part3/results/`
- keep all non-final smoke checks and archived logs under `part3/artifacts_debug/`

Current layout:

- `results/Wild_Video/`
  - final full-run results used in the report
  - contains `sam3/`, `sam3_1/`, and `summary/`
- `results_debug/ablation/`
  - `SAM 3` vs `SAM 3.1` smoke ablation outputs
- `results/DAVIS_Dataset/`
  - reserved for the future full-DAVIS rerun outputs
- `artifacts_debug/checks/`
  - short-clip smoke checks such as `wild_video1_short33_check`
- `artifacts_debug/legacy_outputs/`
  - older local runs created before the cleanup
- `artifacts_debug/slurm_job_logs/`
  - archived `.out` / `.err` files from SLURM runs

This split keeps the reported results easy to find while preserving old debugging artifacts for traceability.
