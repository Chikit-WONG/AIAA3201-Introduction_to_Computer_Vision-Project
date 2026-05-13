#!/usr/bin/env bash
# Edit this file when moving the project to another machine.
# Most project paths are already relative; the values below are only for
# machine-specific locations that cannot be inferred safely.

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export REPO_ROOT
export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
export PART1_CONDA_ENV="${PART1_CONDA_ENV:-cv}"
export PART2_CONDA_ENV="${PART2_CONDA_ENV:-cv2}"
export PART3_CONDA_ENV="${PART3_CONDA_ENV:-cv2}"
export LEGACY_MODELS_ROOT="${LEGACY_MODELS_ROOT:-$HOME/workdir/models}"
export CUDA_MODULE_NAME="${CUDA_MODULE_NAME:-cuda/12.6}"
export PYTORCH_CUDA_ALLOC_CONF_DEFAULT="${PYTORCH_CUDA_ALLOC_CONF_DEFAULT:-expandable_segments:True}"
