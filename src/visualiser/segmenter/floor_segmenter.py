"""Automatic floor segmentation using SAM2 candidates plus depth-guided ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig
from ..depth.depth_anything import DepthAnythingConfig, DepthAnythingRunner, resolve_torch_device
from ..geometry.homography import infer_quad_from_mask
from ..segmenter.sam2_segmenter import Sam2AutomaticMaskConfig, Sam2AutomaticMaskRunner


@dataclass(slots=True)
class AutomaticFloorSegmenterConfig:
    room_image: Path
    output_dir: Path
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


@dataclass(slots=True)
class FloorSegmentationResult:
    selected_mask: np.ndarray
    selected_quad: np.ndarray
    visible_mask: np.ndarray
    occluder_mask: np.ndarray
    depth_map: np.ndarray
    candidate_summaries: list[dict[str, Any]]
    recovery_summary: dict[str, Any]
    phase28_summary: dict[str, Any]
    debug_images: dict[str, np.ndarray]


def resize_image_to_max_side(image_rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if max_side <= 0:
        raise ValueError("segmentation_max_side must be greater than zero.")

    height, width = image_rgb.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image_rgb.copy(), 1.0

    scale = max_side / longest_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def resize_mask(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())
    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth_map, dtype=np.float32)
    return ((depth_map - depth_min) / (depth_max - depth_min)).astype(np.float32)


def create_depth_preview(depth_map: np.ndarray) -> np.ndarray:
    normalized = np.clip(normalize_depth_map(depth_map) * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    label: str | None = None,
) -> np.ndarray:
    overlay = image_rgb.copy()
    color_vector = np.array(color, dtype=np.float32)
    alpha = 0.45
    overlay[mask] = np.clip(
        (1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color_vector,
        0.0,
        255.0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(np.where(mask, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    if label:
        cv2.putText(
            overlay,
            label,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def postprocess_floor_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    kernel = max(3, int(round(min(mask.shape) * 0.015)))
    if kernel % 2 == 0:
        kernel += 1
    kernel_matrix = np.ones((kernel, kernel), dtype=np.uint8)

    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_matrix)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if component_count <= 1:
        return mask

    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    minimum_keep_area = max(
        64,
        int(round(largest_area * 0.035)),
        int(round(mask.size * 0.0006)),
    )

    result = np.zeros_like(opened)
    height, width = mask.shape
    lower_half_start = height // 2
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_keep_area:
            continue

        component_mask = labels == label
        touches_bottom = bool(component_mask[-1, :].any())
        touches_lower_side = bool(component_mask[lower_half_start:, 0].any() or component_mask[lower_half_start:, width - 1].any())
        centroid_y = float(np.where(component_mask)[0].mean() / max(height - 1, 1))
        if not (touches_bottom or touches_lower_side or centroid_y >= 0.62):
            continue

        result[component_mask] = 255

    if not np.any(result):
        return mask

    return result > 0


def smooth_mask_edges(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()

    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    smoothed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    return smoothed > 0


def build_floor_sample_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()

    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    original_area = int(mask.sum())
    minimum_core_area = max(48, int(round(original_area * 0.1)))
    tested_kernel_sizes: set[int] = set()

    for ratio in (0.03, 0.02, 0.01):
        kernel_size = max(3, int(round(min(mask.shape) * ratio)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size in tested_kernel_sizes:
            continue
        tested_kernel_sizes.add(kernel_size)

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1) > 0
        if int(eroded.sum()) >= minimum_core_area:
            return eroded

    return mask.copy()


def build_floor_sample_profile(
    room_rgb: np.ndarray,
    depth_map: np.ndarray,
    sample_mask: np.ndarray,
) -> dict[str, Any]:
    if not np.any(sample_mask):
        raise ValueError("Cannot build a floor sample profile from an empty mask.")

    lab_image = cv2.cvtColor(room_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_pixels = lab_image[sample_mask]
    l_pixels = lab_pixels[:, 0]
    ab_pixels = lab_pixels[:, 1:3]

    ab_std = ab_pixels.std(axis=0)
    l_std = float(l_pixels.std())

    return {
        "lab_image": lab_image,
        "sample_pixels": int(sample_mask.sum()),
        "l_mean": float(l_pixels.mean()),
        "ab_mean": ab_pixels.mean(axis=0).astype(np.float32),
        "chroma_tolerance": float(max(12.0, (float(ab_std.mean()) * 3.0) + 6.0)),
        "luminance_tolerance": float(max(20.0, (l_std * 3.5) + 8.0)),
        "row_depth_tolerance": 0.12,
        "global_depth_range": float(max(np.ptp(depth_map), 1e-6)),
    }


def compute_floor_top_profile(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    valid_columns = np.where(mask.any(axis=0))[0]
    if valid_columns.size == 0:
        return np.full(width, height - 1, dtype=np.int32)

    top_values = np.array(
        [int(np.where(mask[:, column])[0][0]) for column in valid_columns],
        dtype=np.int32,
    )
    if valid_columns.size == 1:
        return np.full(width, int(top_values[0]), dtype=np.int32)

    interpolated = np.interp(
        np.arange(width, dtype=np.float32),
        valid_columns.astype(np.float32),
        top_values.astype(np.float32),
    )
    return np.clip(np.round(interpolated).astype(np.int32), 0, height - 1)


def build_floor_support_region(
    mask: np.ndarray,
    *,
    max_object_lift_px: int,
) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    top_profile = compute_floor_top_profile(mask)
    y_coordinates = np.arange(mask.shape[0], dtype=np.int32)[:, None]
    return y_coordinates >= np.clip(
        top_profile[None, :] - max(1, int(max_object_lift_px)),
        0,
        mask.shape[0] - 1,
    )


def build_floor_contact_region(
    mask: np.ndarray,
    *,
    max_touch_distance_px: int,
    max_object_lift_px: int,
) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    kernel_radius = max(1, int(max_touch_distance_px))
    kernel_size = (kernel_radius * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(np.where(mask, 255, 0).astype(np.uint8), kernel) > 0
    return dilated_mask & build_floor_support_region(mask, max_object_lift_px=max_object_lift_px)


def compute_row_depth_reference(mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    row_depths = np.full(mask.shape[0], np.nan, dtype=np.float32)
    rows = np.where(mask.any(axis=1))[0]
    for row in rows:
        row_depths[row] = float(np.median(depth_map[row, mask[row]].astype(np.float32)))
    return row_depths


def score_floor_recovery_component(
    component_mask: np.ndarray,
    *,
    room_lab: np.ndarray,
    depth_map: np.ndarray,
    floor_profile: dict[str, Any],
    row_depth_reference: np.ndarray,
    distance_to_selected: np.ndarray,
    max_gap_px: int,
) -> tuple[float, dict[str, float]]:
    area = int(component_mask.sum())
    if area == 0:
        return -1.0, {"area": 0.0}

    ys, _ = np.nonzero(component_mask)
    centroid_y = float(ys.mean() / max(component_mask.shape[0] - 1, 1))

    lab_pixels = room_lab[component_mask]
    l_mean = float(lab_pixels[:, 0].mean())
    ab_mean = lab_pixels[:, 1:3].mean(axis=0)
    chroma_distance = float(np.linalg.norm(ab_mean - floor_profile["ab_mean"]))
    luminance_distance = abs(l_mean - float(floor_profile["l_mean"]))

    chroma_score = float(
        np.clip(1.0 - (chroma_distance / max(float(floor_profile["chroma_tolerance"]), 1e-6)), 0.0, 1.0)
    )
    luminance_score = float(
        np.clip(1.0 - (luminance_distance / max(float(floor_profile["luminance_tolerance"]), 1e-6)), 0.0, 1.0)
    )

    component_rows = np.where(component_mask.any(axis=1))[0]
    row_depth_diffs: list[float] = []
    aligned_rows = 0
    for row in component_rows:
        reference_depth = row_depth_reference[row]
        if not np.isfinite(reference_depth):
            continue
        row_pixels = depth_map[row, component_mask[row]].astype(np.float32)
        if row_pixels.size == 0:
            continue
        row_depth_diffs.append(abs(float(np.median(row_pixels)) - float(reference_depth)))
        aligned_rows += 1

    if row_depth_diffs:
        normalized_depth_diff = float(
            np.mean(row_depth_diffs) / max(float(floor_profile["global_depth_range"]), 1e-6)
        )
    else:
        normalized_depth_diff = 1.0
    row_alignment = float(aligned_rows / max(len(component_rows), 1))
    depth_score = float(
        np.clip(
            1.0 - (normalized_depth_diff / max(float(floor_profile["row_depth_tolerance"]), 1e-6)),
            0.0,
            1.0,
        )
    )

    min_distance_px = float(distance_to_selected[component_mask].min())
    proximity_score = float(np.clip(1.0 - (min_distance_px / max(max_gap_px, 1)), 0.0, 1.0))
    lower_image_score = float(np.clip((centroid_y - 0.42) / 0.36, 0.0, 1.0))

    score = (
        0.34 * depth_score
        + 0.24 * chroma_score
        + 0.14 * luminance_score
        + 0.18 * proximity_score
        + 0.06 * row_alignment
        + 0.04 * lower_image_score
    )

    metrics = {
        "score": float(score),
        "area": float(area),
        "centroid_y": centroid_y,
        "chroma_distance": chroma_distance,
        "luminance_distance": float(luminance_distance),
        "normalized_depth_diff": normalized_depth_diff,
        "min_distance_px": min_distance_px,
        "chroma_score": chroma_score,
        "luminance_score": luminance_score,
        "depth_score": depth_score,
        "proximity_score": proximity_score,
        "row_alignment": row_alignment,
        "lower_image_score": lower_image_score,
    }
    return float(score), metrics


def recover_floor_islands(
    room_rgb: np.ndarray,
    depth_map: np.ndarray,
    selected_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    max_gap_ratio: float,
    min_component_area_ratio: float,
    min_score: float,
    min_depth_score: float,
    min_chroma_score: float,
    min_proximity_score: float,
    min_row_alignment: float,
) -> dict[str, Any]:
    sample_mask = build_floor_sample_mask(selected_mask)
    empty_mask = np.zeros_like(selected_mask, dtype=bool)

    if not np.any(selected_mask) or not candidates:
        return {
            "recovered_mask": empty_mask,
            "sample_mask": sample_mask,
            "accepted_pixels_by_candidate": {},
            "summary": {
                "sample_pixels": int(sample_mask.sum()),
                "max_gap_px": 0,
                "min_component_area": 0,
                "considered_components": 0,
                "accepted_component_count": 0,
                "recovered_area": 0,
                "accepted_components": [],
            },
        }

    floor_profile = build_floor_sample_profile(room_rgb, depth_map, sample_mask)
    row_depth_reference = compute_row_depth_reference(selected_mask, depth_map)
    room_lab = floor_profile["lab_image"]
    max_gap_px = max(6, int(round(min(selected_mask.shape) * max_gap_ratio)))
    min_component_area = max(24, int(round(selected_mask.size * min_component_area_ratio)))
    distance_to_selected = cv2.distanceTransform((~selected_mask).astype(np.uint8), cv2.DIST_L2, 3)

    recovered_mask = np.zeros_like(selected_mask, dtype=bool)
    accepted_components: list[dict[str, Any]] = []
    accepted_pixels_by_candidate: dict[int, int] = {}
    considered_components = 0

    for candidate in candidates:
        candidate_mask = candidate.get("mask")
        if not isinstance(candidate_mask, np.ndarray):
            continue

        novel_pixels = candidate_mask.astype(bool) & ~selected_mask & ~recovered_mask
        if int(novel_pixels.sum()) < min_component_area:
            continue

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(novel_pixels, 255, 0).astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue

            component_mask = labels == label
            considered_components += 1
            recovery_score, metrics = score_floor_recovery_component(
                component_mask,
                room_lab=room_lab,
                depth_map=depth_map,
                floor_profile=floor_profile,
                row_depth_reference=row_depth_reference,
                distance_to_selected=distance_to_selected,
                max_gap_px=max_gap_px,
            )

            accepted = (
                recovery_score >= min_score
                and metrics["depth_score"] >= min_depth_score
                and metrics["chroma_score"] >= min_chroma_score
                and metrics["proximity_score"] >= min_proximity_score
                and metrics["row_alignment"] >= min_row_alignment
            )
            if not accepted:
                continue

            recovered_mask |= component_mask
            candidate_index = int(candidate.get("index", -1))
            accepted_pixels_by_candidate[candidate_index] = accepted_pixels_by_candidate.get(candidate_index, 0) + area
            accepted_components.append(
                {
                    "candidate_index": candidate_index,
                    "source": str(candidate.get("source", "sam2")),
                    "recovered_area": area,
                    "score": round(recovery_score, 6),
                    "metrics": {name: round(value, 6) for name, value in metrics.items()},
                }
            )

    recovered_mask = smooth_mask_edges(recovered_mask, kernel_size=3) & ~selected_mask

    return {
        "recovered_mask": recovered_mask,
        "sample_mask": sample_mask,
        "accepted_pixels_by_candidate": accepted_pixels_by_candidate,
        "summary": {
            "sample_pixels": int(sample_mask.sum()),
            "max_gap_px": max_gap_px,
            "min_component_area": min_component_area,
            "considered_components": considered_components,
            "accepted_component_count": len(accepted_components),
            "recovered_area": int(recovered_mask.sum()),
            "profile": {
                "l_mean": round(float(floor_profile["l_mean"]), 6),
                "ab_mean": [round(float(value), 6) for value in floor_profile["ab_mean"]],
                "chroma_tolerance": round(float(floor_profile["chroma_tolerance"]), 6),
                "luminance_tolerance": round(float(floor_profile["luminance_tolerance"]), 6),
                "row_depth_tolerance": round(float(floor_profile["row_depth_tolerance"]), 6),
            },
            "thresholds": {
                "max_gap_ratio": round(float(max_gap_ratio), 6),
                "min_component_area_ratio": round(float(min_component_area_ratio), 8),
                "min_score": round(float(min_score), 6),
                "min_depth_score": round(float(min_depth_score), 6),
                "min_chroma_score": round(float(min_chroma_score), 6),
                "min_proximity_score": round(float(min_proximity_score), 6),
                "min_row_alignment": round(float(min_row_alignment), 6),
            },
            "accepted_components": accepted_components,
        },
    }


def detect_floor_touching_occluders(
    room_rgb: np.ndarray,
    depth_map: np.ndarray,
    selected_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    selected_candidate_index: int,
    max_touch_distance_ratio: float,
    max_object_lift_ratio: float,
    min_object_area_ratio: float,
    max_object_area_ratio: float,
    floor_match_min_score: float,
    floor_match_min_depth_score: float,
    floor_match_min_chroma_score: float,
    floor_match_min_row_alignment: float,
) -> dict[str, Any]:
    sample_mask = build_floor_sample_mask(selected_mask)
    empty_mask = np.zeros_like(selected_mask, dtype=bool)

    if not np.any(selected_mask) or not candidates:
        return {
            "occluder_mask": empty_mask,
            "accepted_pixels_by_candidate": {},
            "summary": {
                "sample_pixels": int(sample_mask.sum()),
                "max_touch_distance_px": 0,
                "max_object_lift_px": 0,
                "min_object_area": 0,
                "max_object_area": 0,
                "considered_components": 0,
                "floor_like_rejected_count": 0,
                "accepted_component_count": 0,
                "occluder_area": 0,
                "accepted_components": [],
            },
        }

    floor_profile = build_floor_sample_profile(room_rgb, depth_map, sample_mask)
    room_lab = floor_profile["lab_image"]
    row_depth_reference = compute_row_depth_reference(selected_mask, depth_map)
    max_touch_distance_px = max(2, int(round(min(selected_mask.shape) * max_touch_distance_ratio)))
    max_object_lift_px = max(4, int(round(selected_mask.shape[0] * max_object_lift_ratio)))
    min_object_area = max(18, int(round(selected_mask.size * min_object_area_ratio)))
    max_object_area = max(min_object_area, int(round(selected_mask.size * max_object_area_ratio)))
    distance_to_selected = cv2.distanceTransform((~selected_mask).astype(np.uint8), cv2.DIST_L2, 3)
    support_region = build_floor_support_region(
        selected_mask,
        max_object_lift_px=max_object_lift_px,
    )
    contact_region = build_floor_contact_region(
        selected_mask,
        max_touch_distance_px=max_touch_distance_px,
        max_object_lift_px=max_object_lift_px,
    )

    occluder_mask = np.zeros_like(selected_mask, dtype=bool)
    accepted_components: list[dict[str, Any]] = []
    accepted_pixels_by_candidate: dict[int, int] = {}
    considered_components = 0
    floor_like_rejected_count = 0

    for candidate_index, candidate in enumerate(candidates):
        if candidate_index == selected_candidate_index:
            continue

        segmentation = candidate.get("segmentation")
        if not isinstance(segmentation, np.ndarray):
            continue

        support_pixels = segmentation.astype(bool) & support_region & ~occluder_mask
        if int(support_pixels.sum()) < min_object_area:
            continue

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(support_pixels, 255, 0).astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_object_area or area > max_object_area:
                continue

            component_mask = labels == label
            contact_area = int((component_mask & contact_region).sum())
            if contact_area < max(8, min_object_area // 4):
                continue
            min_distance_px = float(distance_to_selected[component_mask].min())
            if min_distance_px > max_touch_distance_px:
                continue

            considered_components += 1
            floor_match_score, floor_match_metrics = score_floor_recovery_component(
                component_mask,
                room_lab=room_lab,
                depth_map=depth_map,
                floor_profile=floor_profile,
                row_depth_reference=row_depth_reference,
                distance_to_selected=distance_to_selected,
                max_gap_px=max_touch_distance_px,
            )

            floor_like = (
                floor_match_score >= floor_match_min_score
                and floor_match_metrics["depth_score"] >= floor_match_min_depth_score
                and floor_match_metrics["chroma_score"] >= floor_match_min_chroma_score
                and floor_match_metrics["row_alignment"] >= floor_match_min_row_alignment
            )
            if floor_like:
                floor_like_rejected_count += 1
                continue

            occluder_mask |= component_mask
            accepted_pixels_by_candidate[candidate_index] = (
                accepted_pixels_by_candidate.get(candidate_index, 0) + area
            )
            accepted_components.append(
                {
                    "candidate_index": candidate_index,
                    "source": str(candidate.get("source", "sam2")),
                    "occluder_area": area,
                    "min_distance_px": round(min_distance_px, 6),
                    "floor_match_score": round(float(floor_match_score), 6),
                    "floor_match_metrics": {
                        name: round(float(value), 6) for name, value in floor_match_metrics.items()
                    },
                }
            )

    occluder_mask = smooth_mask_edges(occluder_mask, kernel_size=3)

    return {
        "occluder_mask": occluder_mask,
        "accepted_pixels_by_candidate": accepted_pixels_by_candidate,
        "summary": {
            "sample_pixels": int(sample_mask.sum()),
            "max_touch_distance_px": max_touch_distance_px,
            "max_object_lift_px": max_object_lift_px,
            "min_object_area": min_object_area,
            "max_object_area": max_object_area,
            "considered_components": considered_components,
            "floor_like_rejected_count": floor_like_rejected_count,
            "accepted_component_count": len(accepted_components),
            "occluder_area": int(occluder_mask.sum()),
            "thresholds": {
                "max_touch_distance_ratio": round(float(max_touch_distance_ratio), 6),
                "max_object_lift_ratio": round(float(max_object_lift_ratio), 6),
                "min_object_area_ratio": round(float(min_object_area_ratio), 8),
                "max_object_area_ratio": round(float(max_object_area_ratio), 6),
                "floor_match_min_score": round(float(floor_match_min_score), 6),
                "floor_match_min_depth_score": round(float(floor_match_min_depth_score), 6),
                "floor_match_min_chroma_score": round(float(floor_match_min_chroma_score), 6),
                "floor_match_min_row_alignment": round(float(floor_match_min_row_alignment), 6),
            },
            "accepted_components": accepted_components,
        },
    }


def inpaint_room_region(room_rgb: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if not np.any(mask):
        return room_rgb.copy()

    inpaint_radius = max(1, int(radius))
    inpaint_mask = np.where(mask, 255, 0).astype(np.uint8)
    room_bgr = cv2.cvtColor(room_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(room_bgr, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def expand_floor_from_inpainted_candidates(
    selected_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    min_base_coverage: float = 0.55,
    min_candidate_coverage: float = 0.45,
    min_score: float = 0.45,
) -> dict[str, Any]:
    empty_mask = np.zeros_like(selected_mask, dtype=bool)
    selected_area = int(selected_mask.sum())
    if selected_area == 0 or not candidates:
        return {
            "expanded_mask": empty_mask,
            "summary": {
                "accepted_candidate_count": 0,
                "expanded_area": 0,
                "accepted_candidates": [],
            },
        }

    expanded_mask = np.zeros_like(selected_mask, dtype=bool)
    accepted_candidates: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_mask = candidate.get("mask")
        if not isinstance(candidate_mask, np.ndarray):
            continue

        overlap_area = int((candidate_mask & selected_mask).sum())
        if overlap_area == 0:
            continue

        candidate_area = int(candidate_mask.sum())
        base_coverage = float(overlap_area / max(selected_area, 1))
        candidate_coverage = float(overlap_area / max(candidate_area, 1))
        candidate_score = float(candidate.get("score", 0.0))
        if (
            base_coverage < min_base_coverage
            or candidate_coverage < min_candidate_coverage
            or candidate_score < min_score
        ):
            continue

        novel_pixels = candidate_mask & ~selected_mask & ~expanded_mask
        if not np.any(novel_pixels):
            continue

        expanded_mask |= novel_pixels
        accepted_candidates.append(
            {
                "candidate_index": int(candidate.get("index", -1)),
                "score": round(candidate_score, 6),
                "expanded_area": int(novel_pixels.sum()),
                "base_coverage": round(base_coverage, 6),
                "candidate_coverage": round(candidate_coverage, 6),
            }
        )

    expanded_mask = smooth_mask_edges(expanded_mask, kernel_size=3) & ~selected_mask
    return {
        "expanded_mask": expanded_mask,
        "summary": {
            "accepted_candidate_count": len(accepted_candidates),
            "expanded_area": int(expanded_mask.sum()),
            "thresholds": {
                "min_base_coverage": round(float(min_base_coverage), 6),
                "min_candidate_coverage": round(float(min_candidate_coverage), 6),
                "min_score": round(float(min_score), 6),
            },
            "accepted_candidates": accepted_candidates,
        },
    }


def compute_bottom_zone_coverages(mask: np.ndarray, bottom_band: int) -> tuple[float, float, float]:
    bottom_occupancy = mask[mask.shape[0] - bottom_band :, :].any(axis=0).astype(np.float32)
    left_zone, center_zone, right_zone = np.array_split(bottom_occupancy, 3)
    return float(left_zone.mean()), float(center_zone.mean()), float(right_zone.mean())


def compute_depth_planarity(mask: np.ndarray, depth_map: np.ndarray) -> tuple[float, float]:
    depth_range = float(max(np.ptp(depth_map), 1e-6))
    row_depth_stds: list[float] = []

    rows = np.where(mask.any(axis=1))[0]
    for row in rows:
        row_mask = mask[row]
        if int(row_mask.sum()) < 5:
            continue
        row_depth_stds.append(float(depth_map[row, row_mask].std()))

    if not row_depth_stds:
        return 0.0, 1.0

    normalized_row_depth_std = float(np.mean(row_depth_stds) / depth_range)
    depth_planarity_score = float(np.clip(1.0 - (normalized_row_depth_std / 0.08), 0.0, 1.0))
    return depth_planarity_score, normalized_row_depth_std


def compute_lower_side_coverages(mask: np.ndarray) -> tuple[float, float]:
    lower_half_start = mask.shape[0] // 2
    left_coverage = float(mask[lower_half_start:, 0].mean())
    right_coverage = float(mask[lower_half_start:, -1].mean())
    return left_coverage, right_coverage


def score_floor_candidate(
    mask: np.ndarray,
    depth_map: np.ndarray,
    ann: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    height, width = mask.shape
    image_area = height * width
    area = int(mask.sum())
    if area == 0:
        return -1.0, {"area_ratio": 0.0}

    ys, xs = np.nonzero(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    bottom_band = max(1, int(round(height * 0.08)))
    top_band = max(1, int(round(height * 0.08)))

    horizontal_coverage = float(mask.any(axis=0).mean())
    bottom_width_coverage = float(mask[height - bottom_band :, :].any(axis=0).mean())
    top_width_coverage = float(mask[:top_band, :].any(axis=0).mean())
    bottom_left_coverage, bottom_center_coverage, bottom_right_coverage = compute_bottom_zone_coverages(mask, bottom_band)
    bottom_corner_coverage = max(bottom_left_coverage, bottom_right_coverage)
    lower_left_side_coverage, lower_right_side_coverage = compute_lower_side_coverages(mask)
    side_contact_score = max(lower_left_side_coverage, lower_right_side_coverage)
    lower_half_ratio = float(mask[height // 2 :, :].sum() / area)
    centroid_y = float(ys.mean() / max(height - 1, 1))
    upper_reach_score = float(1.0 - (y_min / max(height - 1, 1)))
    area_ratio = float(area / image_area)
    aspect_score = float(np.clip((bbox_width / max(bbox_height, 1)) / 2.5, 0.0, 1.0))
    y_max_norm = float(y_max / max(height - 1, 1))
    footprint_ratio = float(bottom_width_coverage / max(horizontal_coverage, 1e-6))
    floating_penalty = 0.0
    if bottom_width_coverage < 0.01 and bottom_corner_coverage < 0.05:
        floating_penalty = float(np.clip((0.9 - y_max_norm) / 0.35, 0.0, 1.0))

    rows = np.where(mask.any(axis=1))[0]
    row_spans: list[float] = []
    row_depths: list[float] = []
    row_positions: list[float] = []
    for row in rows:
        row_mask = mask[row]
        row_pixels = np.where(row_mask)[0]
        if row_pixels.size == 0:
            continue
        row_spans.append(float(row_pixels[-1] - row_pixels[0] + 1))
        row_depths.append(float(depth_map[row, row_mask].mean()))
        row_positions.append(float(row) / max(height - 1, 1))

    if len(row_spans) >= 4:
        quartile = max(1, len(row_spans) // 4)
        top_span_mean = float(np.mean(row_spans[:quartile]))
        bottom_span_mean = float(np.mean(row_spans[-quartile:]))
        trapezoid_gain = bottom_span_mean / max(top_span_mean, 1.0)
    else:
        trapezoid_gain = 1.0
    trapezoid_score = float(np.clip((trapezoid_gain - 1.0) / 1.0, 0.0, 1.0))

    if len(row_depths) >= 5 and np.std(row_depths) > 1e-6:
        corr = float(np.corrcoef(row_positions, row_depths)[0, 1])
        depth_row_correlation = abs(corr) if np.isfinite(corr) else 0.0
        quartile = max(1, len(row_depths) // 4)
        depth_range = float(max(np.ptp(depth_map), 1e-6))
        depth_delta = abs(float(np.mean(row_depths[-quartile:]) - np.mean(row_depths[:quartile]))) / depth_range
    else:
        depth_row_correlation = 0.0
        depth_delta = 0.0
    depth_delta_score = float(np.clip(depth_delta, 0.0, 1.0))
    depth_planarity_score, normalized_row_depth_std = compute_depth_planarity(mask, depth_map)
    stilt_penalty = float(np.clip(0.72 - footprint_ratio, 0.0, 0.72) / 0.72) * (1.0 - depth_planarity_score)

    tiny_penalty = float(np.clip((0.02 - area_ratio) / 0.02, 0.0, 1.0)) if area_ratio < 0.02 else 0.0
    oversize_penalty = float(np.clip((area_ratio - 0.85) / 0.15, 0.0, 1.0)) if area_ratio > 0.85 else 0.0

    # Favor masks that extend back into the room and stay depth-planar across rows.
    score = (
        0.18 * float(ann.get("predicted_iou", 0.0))
        + 0.14 * float(ann.get("stability_score", 0.0))
        + 0.14 * bottom_width_coverage
        + 0.10 * horizontal_coverage
        + 0.10 * lower_half_ratio
        + 0.08 * upper_reach_score
        + 0.05 * centroid_y
        + 0.05 * aspect_score
        + 0.05 * bottom_corner_coverage
        + 0.04 * side_contact_score
        + 0.06 * footprint_ratio
        + 0.04 * depth_row_correlation
        + 0.05 * depth_delta_score
        + 0.12 * depth_planarity_score
        + 0.02 * trapezoid_score
        - 0.10 * top_width_coverage
        - 0.08 * oversize_penalty
        - 0.04 * tiny_penalty
        - 0.22 * floating_penalty
        - 0.12 * stilt_penalty
    )

    metrics = {
        "score": float(score),
        "area_ratio": area_ratio,
        "horizontal_coverage": horizontal_coverage,
        "bottom_width_coverage": bottom_width_coverage,
        "top_width_coverage": top_width_coverage,
        "bottom_left_coverage": bottom_left_coverage,
        "bottom_center_coverage": bottom_center_coverage,
        "bottom_right_coverage": bottom_right_coverage,
        "bottom_corner_coverage": bottom_corner_coverage,
        "lower_left_side_coverage": lower_left_side_coverage,
        "lower_right_side_coverage": lower_right_side_coverage,
        "side_contact_score": side_contact_score,
        "lower_half_ratio": lower_half_ratio,
        "centroid_y": centroid_y,
        "upper_reach_score": upper_reach_score,
        "aspect_score": aspect_score,
        "y_max_norm": y_max_norm,
        "footprint_ratio": footprint_ratio,
        "floating_penalty": floating_penalty,
        "stilt_penalty": stilt_penalty,
        "trapezoid_gain": float(trapezoid_gain),
        "trapezoid_score": trapezoid_score,
        "depth_row_correlation": depth_row_correlation,
        "depth_delta_score": depth_delta_score,
        "depth_planarity_score": depth_planarity_score,
        "normalized_row_depth_std": normalized_row_depth_std,
        "predicted_iou": float(ann.get("predicted_iou", 0.0)),
        "stability_score": float(ann.get("stability_score", 0.0)),
    }
    return float(score), metrics


def rank_floor_candidates(
    candidates: list[dict[str, Any]],
    depth_map: np.ndarray,
    *,
    top_candidate_count: int,
) -> list[dict[str, Any]]:
    ranked = evaluate_floor_candidates(candidates, depth_map)
    return ranked[:top_candidate_count]


def evaluate_floor_candidates(
    candidates: list[dict[str, Any]],
    depth_map: np.ndarray,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, ann in enumerate(candidates):
        segmentation = ann.get("segmentation")
        if not isinstance(segmentation, np.ndarray):
            continue

        mask = segmentation.astype(bool)
        area_ratio = float(mask.sum() / mask.size)
        if area_ratio < 0.005 or area_ratio > 0.95:
            continue

        score, metrics = score_floor_candidate(mask, depth_map, ann)
        ranked.append(
            {
                "index": index,
                "mask": mask,
                "score": score,
                "bbox": list(ann.get("bbox", [])),
                "metrics": metrics,
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
                "area": int(ann.get("area", int(mask.sum()))),
                "source": str(ann.get("source", "sam2")),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


class AutomaticFloorSegmenter:
    def __init__(self, app_config: AppConfig, config: AutomaticFloorSegmenterConfig) -> None:
        self.app_config = app_config
        self.config = config
        device = resolve_torch_device(config.device)
        self.depth_runner = DepthAnythingRunner(
            DepthAnythingConfig(
                repo_dir=app_config.depth_anything_repo_dir,
                checkpoint=app_config.depth_anything_checkpoint,
                encoder=app_config.depth_anything_encoder,
                input_size=config.depth_input_size,
                device=device,
            )
        )
        self.sam2_runner = Sam2AutomaticMaskRunner(
            Sam2AutomaticMaskConfig(
                checkpoint=app_config.sam2_checkpoint,
                model_config=app_config.sam2_model_config,
                device=device,
                points_per_side=config.sam_points_per_side,
                points_per_batch=config.sam_points_per_batch,
                pred_iou_thresh=config.sam_pred_iou_thresh,
                stability_score_thresh=config.sam_stability_score_thresh,
                box_nms_thresh=config.sam_box_nms_thresh,
                crop_n_layers=config.sam_crop_n_layers,
                min_mask_region_area=config.min_mask_region_area,
            )
        )

    def segment(self, room_rgb: np.ndarray) -> FloorSegmentationResult:
        original_height, original_width = room_rgb.shape[:2]
        resized_room, _ = resize_image_to_max_side(room_rgb, self.config.segmentation_max_side)
        depth_map_small = self.depth_runner.infer_depth(resized_room)
        candidates = self.sam2_runner.generate_masks(resized_room)
        evaluated = evaluate_floor_candidates(candidates, depth_map_small)
        ranked = evaluated[: self.config.top_candidate_count]
        if not evaluated:
            raise RuntimeError("Automatic floor segmentation did not return any viable floor candidates.")

        selected_candidate = ranked[0]
        anchor_mask_small = postprocess_floor_mask(selected_candidate["mask"])
        recovery = recover_floor_islands(
            resized_room,
            depth_map_small,
            anchor_mask_small,
            evaluated,
            max_gap_ratio=self.config.phase27_max_gap_ratio,
            min_component_area_ratio=self.config.phase27_min_component_area_ratio,
            min_score=self.config.phase27_min_score,
            min_depth_score=self.config.phase27_min_depth_score,
            min_chroma_score=self.config.phase27_min_chroma_score,
            min_proximity_score=self.config.phase27_min_proximity_score,
            min_row_alignment=self.config.phase27_min_row_alignment,
        )
        phase27_recovered_mask_small = recovery["recovered_mask"]
        phase27_visible_mask_small = smooth_mask_edges(
            anchor_mask_small | phase27_recovered_mask_small,
            kernel_size=3,
        )

        occluder_mask_small = np.zeros_like(phase27_visible_mask_small, dtype=bool)
        phase28_recovered_mask_small = np.zeros_like(phase27_visible_mask_small, dtype=bool)
        visible_mask_small = phase27_visible_mask_small.copy()
        selected_mask_small = phase27_visible_mask_small.copy()
        phase28_summary: dict[str, Any] = {
            "enabled": bool(self.config.phase28_enabled),
            "ran_second_pass": False,
            "occluder_area": 0,
            "recovered_area": 0,
            "final_visible_area": int(visible_mask_small.sum()),
            "final_selected_area": int(selected_mask_small.sum()),
            "detection": {},
        }
        phase28_debug_images: dict[str, np.ndarray] = {}
        phase28_occluder_pixels_by_candidate: dict[int, int] = {}

        if self.config.phase28_enabled:
            occluder_detection = detect_floor_touching_occluders(
                resized_room,
                depth_map_small,
                phase27_visible_mask_small,
                candidates,
                selected_candidate_index=int(selected_candidate["index"]),
                max_touch_distance_ratio=self.config.phase28_max_touch_distance_ratio,
                max_object_lift_ratio=self.config.phase28_max_object_lift_ratio,
                min_object_area_ratio=self.config.phase28_min_object_area_ratio,
                max_object_area_ratio=self.config.phase28_max_object_area_ratio,
                floor_match_min_score=self.config.phase27_min_score,
                floor_match_min_depth_score=self.config.phase27_min_depth_score,
                floor_match_min_chroma_score=self.config.phase27_min_chroma_score,
                floor_match_min_row_alignment=self.config.phase27_min_row_alignment,
            )
            occluder_mask_small = occluder_detection["occluder_mask"]
            phase28_occluder_pixels_by_candidate = occluder_detection["accepted_pixels_by_candidate"]
            phase28_summary = {
                "enabled": True,
                "ran_second_pass": False,
                "occluder_area": int(occluder_mask_small.sum()),
                "recovered_area": 0,
                "final_visible_area": int(visible_mask_small.sum()),
                "final_selected_area": int(selected_mask_small.sum()),
                "detection": occluder_detection["summary"],
            }

            if np.any(occluder_mask_small):
                inpainted_room_small = inpaint_room_region(
                    resized_room,
                    occluder_mask_small,
                    self.config.phase28_inpaint_radius,
                )
                phase28_depth_small = self.depth_runner.infer_depth(inpainted_room_small)
                phase28_candidates = self.sam2_runner.generate_masks(inpainted_room_small)
                phase28_evaluated = evaluate_floor_candidates(phase28_candidates, phase28_depth_small)
                phase28_candidate_expansion = expand_floor_from_inpainted_candidates(
                    phase27_visible_mask_small,
                    phase28_evaluated,
                )
                phase28_recovery = recover_floor_islands(
                    inpainted_room_small,
                    phase28_depth_small,
                    phase27_visible_mask_small,
                    phase28_evaluated,
                    max_gap_ratio=self.config.phase27_max_gap_ratio,
                    min_component_area_ratio=self.config.phase27_min_component_area_ratio,
                    min_score=self.config.phase27_min_score,
                    min_depth_score=self.config.phase27_min_depth_score,
                    min_chroma_score=self.config.phase27_min_chroma_score,
                    min_proximity_score=self.config.phase27_min_proximity_score,
                    min_row_alignment=self.config.phase27_min_row_alignment,
                )
                phase28_recovered_mask_small = smooth_mask_edges(
                    phase28_candidate_expansion["expanded_mask"] | phase28_recovery["recovered_mask"],
                    kernel_size=3,
                ) & ~phase27_visible_mask_small
                visible_mask_small = smooth_mask_edges(
                    (phase27_visible_mask_small | phase28_recovered_mask_small) & ~occluder_mask_small,
                    kernel_size=3,
                )
                selected_mask_small = smooth_mask_edges(
                    visible_mask_small | occluder_mask_small,
                    kernel_size=3,
                )
                phase28_summary = {
                    "enabled": True,
                    "ran_second_pass": True,
                    "inpaint_radius": int(max(1, self.config.phase28_inpaint_radius)),
                    "reranked_candidate_count": len(phase28_evaluated),
                    "occluder_area": int(occluder_mask_small.sum()),
                    "recovered_area": int(phase28_recovered_mask_small.sum()),
                    "final_visible_area": int(visible_mask_small.sum()),
                    "final_selected_area": int(selected_mask_small.sum()),
                    "detection": occluder_detection["summary"],
                    "candidate_expansion": phase28_candidate_expansion["summary"],
                    "recovery": phase28_recovery["summary"],
                }
                phase28_debug_images = {
                    "phase28_inpainted_room": cv2.resize(
                        inpainted_room_small,
                        (original_width, original_height),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    "phase28_recovered_mask": np.where(
                        resize_mask(phase28_recovered_mask_small, (original_width, original_height)),
                        255,
                        0,
                    ).astype(np.uint8),
                }

        total_recovered_mask_small = visible_mask_small & ~anchor_mask_small

        anchor_mask = resize_mask(anchor_mask_small, (original_width, original_height))
        anchor_mask = postprocess_floor_mask(anchor_mask)
        recovered_mask = resize_mask(total_recovered_mask_small, (original_width, original_height))
        phase27_recovered_mask = resize_mask(phase27_recovered_mask_small, (original_width, original_height))
        phase28_recovered_mask = resize_mask(phase28_recovered_mask_small, (original_width, original_height))
        occluder_mask = resize_mask(occluder_mask_small, (original_width, original_height))
        occluder_mask = smooth_mask_edges(occluder_mask, kernel_size=3)
        visible_mask = resize_mask(visible_mask_small, (original_width, original_height))
        visible_mask = smooth_mask_edges(visible_mask, kernel_size=3) & ~occluder_mask
        selected_mask = smooth_mask_edges(visible_mask | occluder_mask, kernel_size=3)
        selected_quad = infer_quad_from_mask(anchor_mask)

        depth_map = cv2.resize(
            depth_map_small.astype(np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

        recovery_summary = {
            **recovery["summary"],
            "anchor_area": int(anchor_mask_small.sum()),
            "selected_candidate_index": selected_candidate["index"],
            "selected_candidate_score": round(selected_candidate["score"], 6),
            "final_area": int(selected_mask_small.sum()),
        }

        debug_images: dict[str, np.ndarray] = {
            "anchor_mask": np.where(anchor_mask, 255, 0).astype(np.uint8),
            "selected_mask": np.where(selected_mask, 255, 0).astype(np.uint8),
            "visible_mask": np.where(visible_mask, 255, 0).astype(np.uint8),
            "recovered_mask": np.where(recovered_mask, 255, 0).astype(np.uint8),
            "phase27_recovered_mask": np.where(phase27_recovered_mask, 255, 0).astype(np.uint8),
            "phase28_recovered_mask": np.where(phase28_recovered_mask, 255, 0).astype(np.uint8),
            "sample_core_mask": np.where(resize_mask(recovery["sample_mask"], (original_width, original_height)), 255, 0).astype(np.uint8),
            "occluder_mask": np.where(occluder_mask, 255, 0).astype(np.uint8),
            "selected_overlay": overlay_mask(room_rgb, anchor_mask, color=(64, 220, 64), label=f"anchor score={selected_candidate['score']:.3f}"),
            "visible_overlay": overlay_mask(room_rgb, visible_mask, color=(64, 180, 255), label="phase 2.8 visible floor"),
            "recovered_overlay": overlay_mask(room_rgb, recovered_mask, color=(255, 210, 96), label="recovered floor"),
            "occluder_overlay": overlay_mask(room_rgb, occluder_mask, color=(255, 96, 96), label="phase 2.8 touching objects"),
            "depth_preview": create_depth_preview(depth_map),
        }
        debug_images.update(phase28_debug_images)

        for display_rank, candidate in enumerate(ranked, start=1):
            candidate_mask = resize_mask(candidate["mask"], (original_width, original_height))
            label = f"rank {display_rank} score={candidate['score']:.3f}"
            debug_images[f"candidate_{display_rank:02d}"] = overlay_mask(
                room_rgb,
                candidate_mask,
                color=(255, 180, 64) if display_rank > 1 else (64, 220, 64),
                label=label,
            )

        candidate_summaries = []
        recovered_pixels_by_candidate = recovery["accepted_pixels_by_candidate"]
        for display_rank, candidate in enumerate(ranked, start=1):
            candidate_summaries.append(
                {
                    "rank": display_rank,
                    "index": candidate["index"],
                    "source": candidate["source"],
                    "score": round(candidate["score"], 6),
                    "bbox": candidate["bbox"],
                    "area": candidate["area"],
                    "predicted_iou": round(candidate["predicted_iou"], 6),
                    "stability_score": round(candidate["stability_score"], 6),
                    "phase27_recovered_pixels": int(recovered_pixels_by_candidate.get(candidate["index"], 0)),
                    "phase28_occluder_pixels": int(
                        phase28_occluder_pixels_by_candidate.get(candidate["index"], 0)
                    ),
                    "metrics": {name: round(value, 6) for name, value in candidate["metrics"].items()},
                }
            )

        return FloorSegmentationResult(
            selected_mask=selected_mask,
            selected_quad=selected_quad,
            visible_mask=visible_mask,
            occluder_mask=occluder_mask,
            depth_map=depth_map,
            candidate_summaries=candidate_summaries,
            recovery_summary=recovery_summary,
            phase28_summary=phase28_summary,
            debug_images=debug_images,
        )
