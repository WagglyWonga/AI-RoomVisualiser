"""Image and file helpers for the room visualiser project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask_image(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    if expected_size is not None:
        expected_width, expected_height = expected_size
        if mask.shape[1] != expected_width or mask.shape[0] != expected_height:
            raise ValueError(
                f"Mask size {mask.shape[1]}x{mask.shape[0]} does not match expected "
                f"image size {expected_width}x{expected_height}."
            )

    return mask > 127


def save_image(path: Path, image: np.ndarray) -> None:
    if image.ndim == 2:
        Image.fromarray(image).save(path)
        return

    Image.fromarray(image.astype(np.uint8)).save(path)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
