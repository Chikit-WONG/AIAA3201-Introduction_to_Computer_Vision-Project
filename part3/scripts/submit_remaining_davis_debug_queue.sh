#!/bin/bash
set -euo pipefail

PART3_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PART3_ROOT/.." && pwd)"
cd "$PART3_ROOT"

SEQUENCE_LIST="${1:-artifacts_debug/remaining_davis_sequences_sam3_debug.txt}"
CONFIG="${2:-configs/sam3_davis_all.yaml}"
MAX_DEBUG_JOBS="${MAX_DEBUG_JOBS:-4}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG_FILE="${LOG_FILE:-artifacts_debug/submit_remaining_davis_debug_queue.log}"

mkdir -p "$(dirname "$LOG_FILE")"

if [[ ! -f "$SEQUENCE_LIST" ]]; then
  echo "Missing sequence list: $SEQUENCE_LIST" >&2
  exit 1
fi

slugify() {
  python - "$1" <<'PY'
import sys
from src.io_utils import slugify
print(slugify(sys.argv[1]))
PY
}

is_completed() {
  local sequence="$1"
  local seq_slug
  seq_slug="$(slugify "$sequence")"
  [[ -f "results/results_davis_full/sam3/videos/$seq_slug/sam3_rose_side_effect/output.mp4" ]]
}

current_debug_jobs() {
  squeue -h -u "$USER" -p debug | wc -l
}

submit_one() {
  local sequence="$1"
  local input_video="inputs/davis_videos/${sequence}.mp4"
  local init_mask="../data/DAVIS/Annotations/480p/${sequence}/00000.png"

  if [[ ! -f "$input_video" ]]; then
    echo "[$(date '+%F %T')] skip ${sequence}: missing input video ${input_video}" | tee -a "$LOG_FILE"
    return 0
  fi
  if [[ ! -f "$init_mask" ]]; then
    echo "[$(date '+%F %T')] skip ${sequence}: missing init mask ${init_mask}" | tee -a "$LOG_FILE"
    return 0
  fi

  local submit_out
  if submit_out="$(
    sbatch \
      --export=ALL,CONFIG="$CONFIG",SEQUENCE="$sequence",INPUT_VIDEO="$input_video",INIT_MASK="$init_mask" \
      slurm_scripts/run_part3_sequence_matrix_debug.slurm 2>&1
  )"; then
    echo "[$(date '+%F %T')] submit ${sequence}: ${submit_out}" | tee -a "$LOG_FILE"
    return 0
  fi

  echo "[$(date '+%F %T')] retry ${sequence}: ${submit_out}" | tee -a "$LOG_FILE"
  return 1
}

while IFS= read -r sequence || [[ -n "$sequence" ]]; do
  [[ -z "$sequence" ]] && continue

  if is_completed "$sequence"; then
    echo "[$(date '+%F %T')] skip ${sequence}: already completed" | tee -a "$LOG_FILE"
    continue
  fi

  while true; do
    active_jobs="$(current_debug_jobs | tr -d ' ')"
    if [[ "${active_jobs:-0}" -lt "$MAX_DEBUG_JOBS" ]]; then
      break
    fi
    echo "[$(date '+%F %T')] wait: debug jobs ${active_jobs}/${MAX_DEBUG_JOBS}" | tee -a "$LOG_FILE"
    sleep "$POLL_SECONDS"
  done

  while true; do
    if submit_one "$sequence"; then
      break
    fi
    sleep "$POLL_SECONDS"
  done
  sleep 1
done < "$SEQUENCE_LIST"

echo "[$(date '+%F %T')] all remaining sequences submitted from ${SEQUENCE_LIST}" | tee -a "$LOG_FILE"
