"""Floor texture tiling helpers."""

from __future__ import annotations

import math

import cv2
import numpy as np


def create_tiled_texture_canvas(
    flooring_image: np.ndarray,
    canvas_size: tuple[int, int],
    *,
    texture_scale: float = 1.0,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    if texture_scale <= 0:
        raise ValueError("texture_scale must be greater than zero.")

    canvas_width, canvas_height = canvas_size
    sample_height, sample_width = flooring_image.shape[:2]
    tile_width = max(8, int(round(sample_width * texture_scale)))
    tile_height = max(8, int(round(sample_height * texture_scale)))

    tile = cv2.resize(flooring_image, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR)
    repeats_x = max(1, math.ceil(canvas_width / tile_width) + 1)
    repeats_y = max(1, math.ceil(canvas_height / tile_height) + 1)

    tiled = np.tile(tile, (repeats_y, repeats_x, 1))
    tiled = tiled[:canvas_height, :canvas_width]

    if not rotation_degrees:
        return tiled

    center = (canvas_width / 2.0, canvas_height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    return cv2.warpAffine(
        tiled,
        matrix,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
