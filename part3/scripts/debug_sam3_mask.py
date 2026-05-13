#!/usr/bin/env python3
"""Run a SAM 3 mask-only diagnostic on one video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PART3_DIR = Path(__file__).resolve().parents[1]
if str(PART3_DIR) not in sys.path:
    sys.path.insert(0, str(PART3_DIR))

from src.io_utils import ensure_dir, load_yaml, resolve_path, slugify
from src.sam3_wrapper import SAM3MaskGenerator


def artifact_prefix(version: str) -> str:
    normalized = version.strip().lower()
    if normalized in {"sam3.1", "sam3_1", "sam31"}:
        return "sam3_1"
    return "sam3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--init-mask", default=None)
    parser.add_argument("--output-root", default="results_debug/mask_check")
    args = parser.parse_args()

    part3_dir = PART3_DIR
    config = load_yaml(part3_dir / args.config)
    sam3_cfg = config["sam3"]
    sequence_slug = slugify(args.sequence)
    variant = sam3_cfg.get("version", "sam3").replace(".", "_")
    output_root = part3_dir / args.output_root / variant / sequence_slug
    mask_dir = ensure_dir(output_root / "object_mask")
    videos_dir = ensure_dir(output_root / "videos")

    generator = SAM3MaskGenerator(
        sam3_dir=resolve_path(str(part3_dir), sam3_cfg["repo_dir"]),
        checkpoint=resolve_path(str(part3_dir), sam3_cfg["checkpoint"]) if sam3_cfg.get("checkpoint") else None,
        device=config.get("device", "cuda"),
        version=sam3_cfg.get("version", "sam3"),
        frame_index=sam3_cfg.get("frame_index", 0),
        compile=sam3_cfg.get("compile", False),
        async_loading_frames=sam3_cfg.get("async_loading_frames", False),
        use_fa3=sam3_cfg.get("use_fa3"),
        use_rope_real=sam3_cfg.get("use_rope_real"),
    )
    result = generator.generate_video_masks(
        input_video=str((part3_dir / args.input).resolve()),
        output_mask_dir=mask_dir,
        prompt=args.prompt,
        init_mask_path=str((part3_dir / args.init_mask).resolve()) if args.init_mask else None,
        output_overlay_video=str(Path(videos_dir) / f"{artifact_prefix(sam3_cfg.get('version', 'sam3'))}_overlay.mp4"),
        output_mask_video=str(Path(videos_dir) / f"{artifact_prefix(sam3_cfg.get('version', 'sam3'))}_mask.mp4"),
    )
    print("SAM 3 mask diagnostic completed:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
