#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from src.video_utils import pack_result_videos


def main() -> None:
    parser = argparse.ArgumentParser(description='Create inpainted.mp4 / mask.mp4 / overlay.mp4 for existing Part 1 results.')
    parser.add_argument('--root', default='results', help='Root directory to scan for result folders.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--default-fps', type=float, default=30.0)
    args = parser.parse_args()

    part1_root = Path(__file__).resolve().parent
    root = (part1_root / args.root).resolve()
    count = pack_result_videos(root, part1_root, overwrite=args.overwrite, default_fps=args.default_fps)
    print(f'packed_videos={count}')


if __name__ == '__main__':
    main()
