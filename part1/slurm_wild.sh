#!/bin/bash
#SBATCH -p i64m1tga40u
#SBATCH -o temp/wild_output.txt
#SBATCH -e temp/wild_err.txt
#SBATCH -n 1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH -J p1_wild

PART1_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$PART1_ROOT/.." && pwd)"
source "$REPO_ROOT/local_paths.sh"
cd "$PART1_ROOT"

source "$CONDA_SH"
conda activate "$PART1_CONDA_ENV"
module load "$CUDA_MODULE_NAME"

VARIANT="${VARIANT:-temporal_aligned}"
VIDEO_DIR="${VIDEO_DIR:-../data/Wild_Video/input_with_person}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/results_wild_video}"

case "$VARIANT" in
  temporal_aligned)
    CONFIG="configs/wild.yaml"
    EXTRA_ARGS=()
    ;;
  temporal_no_align)
    CONFIG="configs/wild_no_align.yaml"
    EXTRA_ARGS=()
    ;;
  spatial_only)
    CONFIG="configs/wild.yaml"
    EXTRA_ARGS=(--no-temporal)
    ;;
  *)
    echo "Unknown VARIANT: $VARIANT"
    exit 1
    ;;
esac

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

shopt -s nullglob
for video in "$VIDEO_DIR"/*.mp4; do
  seq="$(basename "$video" .mp4)"
  echo "Running $VARIANT on $seq"
  python run.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --input "$video" --output "$OUTPUT_ROOT/$VARIANT/$seq"
done

echo ""
echo "Job ended at $(date)"
conda deactivate
