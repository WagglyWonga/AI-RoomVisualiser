"""Phase 2 automatic floor segmentation and visualisation flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import AppConfig
from .io_utils import ensure_directory, load_rgb_image, save_image, save_json_file
from .pipeline import build_floor_replacement_artifacts
from .segmenter.floor_segmenter import AutomaticFloorSegmenter, AutomaticFloorSegmenterConfig


@dataclass(slots=True)
class AutoVisualiserConfig:
    room_image: Path
    output_dir: Path
    flooring_image: Path | None = None
    device: str = "auto"
    segmentation_max_side: int = 960
    depth_input_size: int = 518
    sam_points_per_side: int = 16
    sam_points_per_batch: int = 32
    sam_pred_iou_thresh: float = 0.7
    sam_stability_score_thresh: float = 0.9
    sam_box_nms_thresh: float = 0.7
    sam_crop_n_layers: int = 0
    min_mask_region_area: int = 400
    top_candidate_count: int = 5
    phase27_max_gap_ratio: float = 0.08
    phase27_min_component_area_ratio: float = 0.00015
    phase27_min_score: float = 0.58
    phase27_min_depth_score: float = 0.5
    phase27_min_chroma_score: float = 0.45
    phase27_min_proximity_score: float = 0.25
    phase27_min_row_alignment: float = 0.35
    phase28_enabled: bool = True
    phase28_max_touch_distance_ratio: float = 0.03
    phase28_max_object_lift_ratio: float = 0.14
    phase28_min_object_area_ratio: float = 0.00015
    phase28_max_object_area_ratio: float = 0.14
    phase28_inpaint_radius: int = 5
    texture_scale: float = 1.0
    texture_rotation_degrees: float = 0.0
    feather_radius: int = 10
    shading_strength: float = 0.7
    save_debug_images: bool = True


@dataclass(slots=True)
class AutoVisualiserResult:
    output_dir: Path
    final_image_path: Path
    metadata_path: Path
    debug_paths: dict[str, Path]


def _resolve_output_dir(output_dir: Path, prefix: str) -> Path:
    if output_dir.name:
        return ensure_directory(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ensure_directory(output_dir / f"{prefix}-{timestamp}")


def _build_segmenter_config(config: AutoVisualiserConfig) -> AutomaticFloorSegmenterConfig:
    return AutomaticFloorSegmenterConfig(
        room_image=config.room_image,
        output_dir=config.output_dir,
        device=config.device,
        segmentation_max_side=config.segmentation_max_side,
        depth_input_size=config.depth_input_size,
        sam_points_per_side=config.sam_points_per_side,
        sam_points_per_batch=config.sam_points_per_batch,
        sam_pred_iou_thresh=config.sam_pred_iou_thresh,
        sam_stability_score_thresh=config.sam_stability_score_thresh,
        sam_box_nms_thresh=config.sam_box_nms_thresh,
        sam_crop_n_layers=config.sam_crop_n_layers,
        min_mask_region_area=config.min_mask_region_area,
        top_candidate_count=config.top_candidate_count,
        phase27_max_gap_ratio=config.phase27_max_gap_ratio,
        phase27_min_component_area_ratio=config.phase27_min_component_area_ratio,
        phase27_min_score=config.phase27_min_score,
        phase27_min_depth_score=config.phase27_min_depth_score,
        phase27_min_chroma_score=config.phase27_min_chroma_score,
        phase27_min_proximity_score=config.phase27_min_proximity_score,
        phase27_min_row_alignment=config.phase27_min_row_alignment,
        phase28_enabled=config.phase28_enabled,
        phase28_max_touch_distance_ratio=config.phase28_max_touch_distance_ratio,
        phase28_max_object_lift_ratio=config.phase28_max_object_lift_ratio,
        phase28_min_object_area_ratio=config.phase28_min_object_area_ratio,
        phase28_max_object_area_ratio=config.phase28_max_object_area_ratio,
        phase28_inpaint_radius=config.phase28_inpaint_radius,
    )


def save_segmentation_bundle(
    output_dir: Path,
    segmentation_result,
) -> tuple[Path, Path, dict[str, Path]]:
    mask_path = output_dir / "selected_mask.png"
    metadata_path = output_dir / "segment_floor.json"
    save_image(mask_path, np.where(segmentation_result.selected_mask, 255, 0).astype(np.uint8))

    debug_paths: dict[str, Path] = {}
    for name, image in segmentation_result.debug_images.items():
        path = output_dir / f"{name}.png"
        save_image(path, image)
        debug_paths[name] = path

    save_json_file(
        metadata_path,
        {
            "selected_quad": segmentation_result.selected_quad.round(3).tolist(),
            "candidate_summaries": segmentation_result.candidate_summaries,
            "recovery_summary": segmentation_result.recovery_summary,
            "phase28_summary": segmentation_result.phase28_summary,
            "mask_image": str(mask_path),
            "debug_images": {name: str(path) for name, path in debug_paths.items()},
        },
    )
    return mask_path, metadata_path, debug_paths


def run_segment_floor(app_config: AppConfig, config: AutoVisualiserConfig) -> AutoVisualiserResult:
    room_rgb = load_rgb_image(config.room_image)
    output_dir = _resolve_output_dir(config.output_dir, "segment-floor")
    segmenter = AutomaticFloorSegmenter(app_config, _build_segmenter_config(config))
    segmentation_result = segmenter.segment(room_rgb)
    mask_path, metadata_path, debug_paths = save_segmentation_bundle(output_dir, segmentation_result)

    debug_paths = {"selected_mask": mask_path, **debug_paths}
    return AutoVisualiserResult(
        output_dir=output_dir,
        final_image_path=mask_path,
        metadata_path=metadata_path,
        debug_paths=debug_paths,
    )


def run_auto_visualiser(app_config: AppConfig, config: AutoVisualiserConfig) -> AutoVisualiserResult:
    if config.flooring_image is None:
        raise ValueError("A flooring image is required for automatic visualisation.")

    room_rgb = load_rgb_image(config.room_image)
    flooring_rgb = load_rgb_image(config.flooring_image)
    output_dir = _resolve_output_dir(config.output_dir, "auto-visualise")

    segmenter = AutomaticFloorSegmenter(app_config, _build_segmenter_config(config))
    segmentation_result = segmenter.segment(room_rgb)

    artifacts = build_floor_replacement_artifacts(
        room_rgb,
        flooring_rgb,
        segmentation_result.selected_mask,
        quad=segmentation_result.selected_quad,
        preserve_mask=segmentation_result.occluder_mask,
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
        combined_debug_images = {
            "selected_mask": np.where(segmentation_result.selected_mask, 255, 0).astype(np.uint8),
            **{f"seg_{name}": image for name, image in segmentation_result.debug_images.items()},
            **artifacts.debug_images,
        }
        for name, image in combined_debug_images.items():
            path = output_dir / f"{name}.png"
            save_image(path, image)
            debug_paths[name] = path

    save_json_file(
        metadata_path,
        {
            "room_image": str(config.room_image),
            "flooring_image": str(config.flooring_image),
            "final_image": str(final_image_path),
            "selected_quad": segmentation_result.selected_quad.round(3).tolist(),
            "homography": artifacts.homography.round(6).tolist(),
            "texture_scale": config.texture_scale,
            "texture_rotation_degrees": config.texture_rotation_degrees,
            "feather_radius": config.feather_radius,
            "shading_strength": config.shading_strength,
            "candidate_summaries": segmentation_result.candidate_summaries,
            "recovery_summary": segmentation_result.recovery_summary,
            "phase28_summary": segmentation_result.phase28_summary,
            "debug_images": {name: str(path) for name, path in debug_paths.items()},
        },
    )

    return AutoVisualiserResult(
        output_dir=output_dir,
        final_image_path=final_image_path,
        metadata_path=metadata_path,
        debug_paths=debug_paths,
    )
