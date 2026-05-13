#!/bin/bash
# Submit full-DAVIS SAM3 sequence-matrix runs through the debug partition.
# Each array task runs one DAVIS sequence with the 5-method matrix.
# Concurrency is capped to 2 to stay within the debug GPU QOS.
#SBATCH -J p3_davis_dbg
#SBATCH -p debug
#SBATCH --array=0-0%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o artifacts_debug/slurm_job_logs/p3_davis_debug_%A_%a.out
#SBATCH -e artifacts_debug/slurm_job_logs/p3_davis_debug_%A_%a.err

set -euo pipefail

PART3_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
ROOT="$PART3_ROOT"
cd "$ROOT"
mkdir -p artifacts_debug/slurm_job_logs

: "${CONFIG:?CONFIG is required}"
: "${SEQUENCE_LIST:?SEQUENCE_LIST is required}"

if [[ ! -f "$SEQUENCE_LIST" ]]; then
  echo "Missing SEQUENCE_LIST: $SEQUENCE_LIST" >&2
  exit 1
fi

SEQUENCE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")"
if [[ -z "$SEQUENCE" ]]; then
  echo "No sequence found for task ${SLURM_ARRAY_TASK_ID} in $SEQUENCE_LIST" >&2
  exit 1
fi

INPUT_VIDEO="inputs/davis_videos/${SEQUENCE}.mp4"
INIT_MASK="../data/DAVIS/Annotations/480p/${SEQUENCE}/00000.png"

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Missing input video: $INPUT_VIDEO" >&2
  exit 1
fi

if [[ ! -f "$INIT_MASK" ]]; then
  echo "Missing init mask: $INIT_MASK" >&2
  exit 1
fi

export CONFIG
export SEQUENCE
export INPUT_VIDEO
export INIT_MASK

bash slurm_scripts/run_part3_sequence_matrix_debug.slurm
