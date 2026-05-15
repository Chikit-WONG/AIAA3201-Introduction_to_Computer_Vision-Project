"""I/O helpers for Part 3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | os.PathLike[str], payload: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_text(path: str | os.PathLike[str], text: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    chars = []
    for char in lowered:
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_"}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "default"


def sam3_variant_slug(version: str | None) -> str:
    normalized = str(version or "sam3").strip().lower()
    if normalized in {"sam3.1", "sam3_1", "sam31"}:
        return "sam3_1"
    return "sam3"


def part3_method_output_slug(method: str, sam3_version: str | None = None) -> str:
    if method.startswith("sam3_") and sam3_variant_slug(sam3_version) == "sam3_1":
        suffix = method[len("sam3_"):]
        return slugify(f"sam3_1_{suffix}")
    return slugify(method)


def resolve_path(base_dir: str, maybe_relative: str) -> str:
    if not maybe_relative:
        return maybe_relative
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.abspath(os.path.join(base_dir, maybe_relative))
