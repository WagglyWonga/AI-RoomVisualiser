"""Manual-mask Phase 1 pipeline for floor replacement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .compositing.compositor import (
    apply_shading,
    build_shading_map,
    composite_floor,
    feather_mask,
    match_masked_luminance,
)
from .geometry.homography import (
    estimate_texture_canvas_size,
    infer_quad_from_mask,
    order_quad_points,
    rasterize_polygon,
    warp_texture_to_room,
)
from .io_utils import ensure_directory, load_json_file, load_mask_image, load_rgb_image, save_image, save_json_file
from .materials.tiler import create_tiled_texture_canvas


@dataclass(slots=True)
class ManualVisualiserConfig:
    room_image: Path
    flooring_image: Path
    output_dir: Path
    mask_image: Path | None = None
    polygon_file: Path | None = None
    quad_file: Path | None = None
    texture_scale: float = 1.0
    texture_rotation_degrees: float = 0.0
    feather_radius: int = 10
    shading_strength: float = 0.7
    save_debug_images: bool = True


@dataclass(slots=True)
class ManualVisualiserResult:
    output_dir: Path
    final_image_path: Path
    metadata_path: Path
    debug_paths: dict[str, Path]


@dataclass(slots=True)
class FloorReplacementArtifacts:
    final_rgb: np.ndarray
    quad: np.ndarray
    homography: np.ndarray
    debug_images: dict[str, np.ndarray]


def _parse_points_file(path: Path) -> np.ndarray:
    payload = load_json_file(path)
    if isinstance(payload, dict):
        points = payload.get("points")
    else:
        points = payload

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected point data shaped like [[x, y], ...] in {path}.")

    return array


def _draw_quad_overlay(room_rgb: np.ndarray, quad: np.ndarray) -> np.ndarray:
    overlay = room_rgb.copy()
    points = np.round(quad).astype(np.int32)
    cv2.polylines(overlay, [points], isClosed=True, color=(0, 255, 0), thickness=4)
    return overlay


def _mask_preview(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def _resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.name:
        return ensure_directory(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_directory(output_dir / f"manual-visualise-{timestamp}")


def _build_mask_and_quad(
    room_size: tuple[int, int],
    *,
    mask_image: Path | None,
    polygon_file: Path | None,
    quad_file: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    room_width, room_height = room_size
    polygon_points = _parse_points_file(polygon_file) if polygon_file is not None else None

    if mask_image is not None:
        mask = load_mask_image(mask_image, expected_size=room_size)
    elif polygon_points is not None:
        mask = rasterize_polygon(room_size, polygon_points)
    else:
        raise ValueError("Provide either a mask image or a polygon file for manual visualisation.")

    if not np.any(mask):
        raise ValueError("The supplied floor mask is empty.")

    if quad_file is not None:
        quad = order_quad_points(_parse_points_file(quad_file))
    elif polygon_points is not None and polygon_points.shape[0] == 4:
        quad = order_quad_points(polygon_points)
    else:
        quad = infer_quad_from_mask(mask)

    quad[:, 0] = np.clip(quad[:, 0], 0, room_width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, room_height - 1)
    return mask, quad


def build_floor_replacement_artifacts(
    room_rgb: np.ndarray,
    flooring_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    quad: np.ndarray | None,
    preserve_mask: np.ndarray | None = None,
    texture_scale: float,
    texture_rotation_degrees: float,
    feather_radius: int,
    shading_strength: float,
) -> FloorReplacementArtifacts:
    room_height, room_width = room_rgb.shape[:2]
    resolved_quad = infer_quad_from_mask(mask) if quad is None else quad

    canvas_size = estimate_texture_canvas_size((room_height, room_width), resolved_quad)
    tiled_texture = create_tiled_texture_canvas(
        flooring_rgb,
        canvas_size,
        texture_scale=texture_scale,
        rotation_degrees=texture_rotation_degrees,
    )
    warped_texture, homography = warp_texture_to_room(tiled_texture, resolved_quad, (room_width, room_height))
    matched_texture = match_masked_luminance(room_rgb, warped_texture, mask)
    shading_map = build_shading_map(room_rgb, mask, shading_strength)
    shaded_texture = apply_shading(matched_texture, shading_map)
    alpha_mask = feather_mask(mask, feather_radius)
    final_rgb = composite_floor(room_rgb, shaded_texture, alpha_mask)
    preserve_alpha = None
    if preserve_mask is not None and np.any(preserve_mask):
        preserve_alpha = feather_mask(preserve_mask, max(1, feather_radius // 3))
        final_rgb = composite_floor(final_rgb, room_rgb, preserve_alpha)

    debug_images = {
        "mask": _mask_preview(mask),
        "quad_overlay": _draw_quad_overlay(room_rgb, resolved_quad),
        "tiled_texture": tiled_texture,
        "warped_texture": warped_texture,
        "matched_texture": matched_texture,
        "shaded_texture": shaded_texture,
        "alpha_mask": np.clip(alpha_mask * 255.0, 0.0, 255.0).astype(np.uint8),
    }
    if preserve_mask is not None:
        debug_images["preserve_mask"] = _mask_preview(preserve_mask)
    if preserve_alpha is not None:
        debug_images["preserve_alpha"] = np.clip(preserve_alpha * 255.0, 0.0, 255.0).astype(np.uint8)

    return FloorReplacementArtifacts(
        final_rgb=final_rgb,
        quad=resolved_quad,
        homography=homography,
        debug_images=debug_images,
    )


def run_manual_visualiser(config: ManualVisualiserConfig) -> ManualVisualiserResult:
    room_rgb = load_rgb_image(config.room_image)
    flooring_rgb = load_rgb_image(config.flooring_image)
    room_height, room_width = room_rgb.shape[:2]

    output_dir = _resolve_output_dir(config.output_dir)
    mask, quad = _build_mask_and_quad(
        (room_width, room_height),
        mask_image=config.mask_image,
        polygon_file=config.polygon_file,
        quad_file=config.quad_file,
    )
    artifacts = build_floor_replacement_artifacts(
        room_rgb,
        flooring_rgb,
        mask,
        quad=quad,
        preserve_mask=None,
        texture_scale=config.texture_scale,
        texture_rotation_degrees=config.texture_rotation_degrees,
        feather_radius=config.feather_radius,
        shading_strength=config.shading_strength,
    )

    final_image_path = output_dir / "final.png"
    metadata_path = output_dir / "run.json"
    save_image(final_image_path, artifacts.final_rgb)

    debug_paths: dict[str, Path] = {}
    if config.save_debug_images:
        for name, image in artifacts.debug_images.items():
            path = output_dir / f"{name}.png"
            save_image(path, image)
            debug_paths[name] = path

    save_json_file(
        metadata_path,
        {
            "room_image": str(config.room_image),
            "flooring_image": str(config.flooring_image),
            "mask_image": str(config.mask_image) if config.mask_image else None,
            "polygon_file": str(config.polygon_file) if config.polygon_file else None,
            "quad_file": str(config.quad_file) if config.quad_file else None,
            "texture_scale": config.texture_scale,
            "texture_rotation_degrees": config.texture_rotation_degrees,
            "feather_radius": config.feather_radius,
            "shading_strength": config.shading_strength,
            "used_quad": artifacts.quad.round(3).tolist(),
            "homography": artifacts.homography.round(6).tolist(),
            "final_image": str(final_image_path),
            "debug_images": {name: str(path) for name, path in debug_paths.items()},
        },
    )

    return ManualVisualiserResult(
        output_dir=output_dir,
        final_image_path=final_image_path,
        metadata_path=metadata_path,
        debug_paths=debug_paths,
    )
