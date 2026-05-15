from __future__ import annotations

from pathlib import Path
import cv2


def infer_fps(part2_root: Path, sequence_name: str, default_fps: float = 30.0) -> float:
    wild_candidate = part2_root.parent / 'data' / 'Wild_Video' / 'input_with_person' / f'{sequence_name}.mp4'
    if wild_candidate.is_file():
        cap = cv2.VideoCapture(str(wild_candidate))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 1e-6:
                return fps
        finally:
            cap.release()
    return default_fps


def frames_to_video(frames_dir: Path, output_path: Path, fps: float = 30.0) -> None:
    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not frame_paths:
        raise FileNotFoundError(f'No frames found in {frames_dir}')
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f'Failed to read frame: {frame_paths[0]}')
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open video writer for {output_path}')
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise RuntimeError(f'Failed to read frame: {frame_path}')
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()


def pack_result_videos(root: Path, part_root: Path, overwrite: bool = False, default_fps: float = 30.0) -> int:
    created = 0
    for dirpath in root.rglob('*'):
        if not dirpath.is_dir():
            continue
        base = dirpath.name
        if base == 'frames':
            output_name = 'inpainted.mp4'
        elif base == 'masks':
            output_name = 'mask.mp4'
        elif base in {'overlay', 'overlays'}:
            output_name = 'overlay.mp4'
        else:
            continue
        output_path = dirpath.parent / output_name
        if output_path.exists() and not overwrite:
            continue
        seq_name = dirpath.parent.name
        fps = infer_fps(part_root, seq_name, default_fps)
        try:
            frames_to_video(dirpath, output_path, fps=fps)
            created += 1
        except FileNotFoundError:
            continue
    return created
