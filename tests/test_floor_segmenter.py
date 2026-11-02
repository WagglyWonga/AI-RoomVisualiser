from __future__ import annotations

from pathlib import Path

import numpy as np

from visualiser.config import AppConfig
from visualiser.segmenter.floor_segmenter import (
    AutomaticFloorSegmenter,
    AutomaticFloorSegmenterConfig,
    postprocess_floor_mask,
    rank_floor_candidates,
    resize_image_to_max_side,
)


def test_resize_image_to_max_side_preserves_aspect_ratio() -> None:
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    resized, scale = resize_image_to_max_side(image, 200)

    assert resized.shape == (100, 200, 3)
    assert round(scale, 3) == 0.25


def test_rank_floor_candidates_prefers_bottom_wide_mask() -> None:
    height, width = 180, 240
    depth_map = np.tile(np.linspace(0.1, 1.0, height, dtype=np.float32).reshape(height, 1), (1, width))

    floor_mask = np.zeros((height, width), dtype=bool)
    floor_mask[90:, :] = True
    wall_mask = np.zeros((height, width), dtype=bool)
    wall_mask[20:170, 150:205] = True
    ceiling_mask = np.zeros((height, width), dtype=bool)
    ceiling_mask[:70, :] = True

    candidates = [
        {"segmentation": wall_mask, "predicted_iou": 0.94, "stability_score": 0.96, "bbox": [150, 20, 55, 150], "area": int(wall_mask.sum())},
        {"segmentation": floor_mask, "predicted_iou": 0.86, "stability_score": 0.91, "bbox": [0, 90, 240, 90], "area": int(floor_mask.sum())},
        {"segmentation": ceiling_mask, "predicted_iou": 0.95, "stability_score": 0.97, "bbox": [0, 0, 240, 70], "area": int(ceiling_mask.sum())},
    ]

    ranked = rank_floor_candidates(candidates, depth_map, top_candidate_count=3)

    assert ranked
    assert ranked[0]["mask"].sum() == floor_mask.sum()
    assert ranked[0]["metrics"]["bottom_width_coverage"] > ranked[0]["metrics"]["top_width_coverage"]


def test_postprocess_floor_mask_keeps_significant_floor_islands_and_removes_noise() -> None:
    mask = np.zeros((120, 160), dtype=bool)
    mask[70:119, 10:85] = True
    mask[82:119, 108:159] = True
    mask[10:16, 10:16] = True
    mask[35:38, 100:103] = True

    cleaned = postprocess_floor_mask(mask)

    assert cleaned[90, 40]
    assert cleaned[96, 132]
    assert not cleaned[12, 12]
    assert not cleaned[36, 101]


def test_rank_floor_candidates_prefers_occluded_floor_over_foreground_sofa() -> None:
    height, width = 180, 240
    depth_map = np.tile(np.linspace(0.2, 1.0, height, dtype=np.float32).reshape(height, 1), (1, width))

    floor_mask = np.zeros((height, width), dtype=bool)
    floor_mask[90:, :] = True
    floor_mask[125:, 70:180] = False

    sofa_mask = np.zeros((height, width), dtype=bool)
    sofa_polygon = np.array([[70, 118], [165, 118], [205, 179], [58, 179]], dtype=np.int32)
    import cv2

    sofa_filled = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(sofa_filled, [sofa_polygon], 255)
    sofa_mask = sofa_filled > 0

    x_curve = np.abs(np.linspace(-1.0, 1.0, width, dtype=np.float32))[None, :]
    depth_map = depth_map + (0.25 * np.tile(x_curve, (height, 1)) * sofa_mask.astype(np.float32))

    candidates = [
        {
            "segmentation": sofa_mask,
            "predicted_iou": 0.97,
            "stability_score": 0.96,
            "bbox": [58, 118, 148, 62],
            "area": int(sofa_mask.sum()),
        },
        {
            "segmentation": floor_mask,
            "predicted_iou": 0.93,
            "stability_score": 0.92,
            "bbox": [0, 90, 240, 90],
            "area": int(floor_mask.sum()),
        },
    ]

    ranked = rank_floor_candidates(candidates, depth_map, top_candidate_count=2)

    assert ranked
    assert ranked[0]["mask"].sum() == floor_mask.sum()
    assert ranked[0]["metrics"]["depth_planarity_score"] > ranked[1]["metrics"]["depth_planarity_score"]
    assert ranked[0]["metrics"]["upper_reach_score"] > ranked[1]["metrics"]["upper_reach_score"]


def test_rank_floor_candidates_penalizes_floating_tabletop() -> None:
    height, width = 180, 240
    depth_map = np.tile(np.linspace(0.15, 1.0, height, dtype=np.float32).reshape(height, 1), (1, width))

    floor_mask = np.zeros((height, width), dtype=bool)
    floor_mask[108:, 10:230] = True

    tabletop_mask = np.zeros((height, width), dtype=bool)
    tabletop_mask[84:112, 58:186] = True

    candidates = [
        {
            "segmentation": tabletop_mask,
            "predicted_iou": 0.96,
            "stability_score": 0.97,
            "bbox": [58, 84, 128, 28],
            "area": int(tabletop_mask.sum()),
        },
        {
            "segmentation": floor_mask,
            "predicted_iou": 0.87,
            "stability_score": 0.92,
            "bbox": [10, 108, 220, 72],
            "area": int(floor_mask.sum()),
        },
    ]

    ranked = rank_floor_candidates(candidates, depth_map, top_candidate_count=2)

    assert ranked
    assert ranked[0]["mask"].sum() == floor_mask.sum()
    assert ranked[1]["metrics"]["floating_penalty"] > 0.0


def test_segment_recovers_nearby_floor_island_and_rejects_rug(monkeypatch) -> None:
    height, width = 120, 160
    room_rgb = np.full((height, width, 3), 232, dtype=np.uint8)

    floor_color = np.array([152, 116, 76], dtype=np.uint8)
    rug_color = np.array([46, 78, 182], dtype=np.uint8)
    room_rgb[84:, :] = floor_color
    room_rgb[88:106, 58:82] = floor_color
    room_rgb[88:106, 104:128] = rug_color

    depth_map = np.tile(np.linspace(0.2, 1.0, height, dtype=np.float32).reshape(height, 1), (1, width))

    main_floor_mask = np.zeros((height, width), dtype=bool)
    main_floor_mask[84:, :] = True
    main_floor_mask[88:106, 58:82] = False
    main_floor_mask[88:106, 104:128] = False

    island_mask = np.zeros((height, width), dtype=bool)
    island_mask[88:106, 58:82] = True

    rug_mask = np.zeros((height, width), dtype=bool)
    rug_mask[88:106, 104:128] = True

    candidates = [
        {
            "segmentation": rug_mask,
            "predicted_iou": 0.96,
            "stability_score": 0.96,
            "bbox": [104, 88, 24, 18],
            "area": int(rug_mask.sum()),
        },
        {
            "segmentation": main_floor_mask,
            "predicted_iou": 0.9,
            "stability_score": 0.93,
            "bbox": [0, 84, width, height - 84],
            "area": int(main_floor_mask.sum()),
        },
        {
            "segmentation": island_mask,
            "predicted_iou": 0.84,
            "stability_score": 0.9,
            "bbox": [58, 88, 24, 18],
            "area": int(island_mask.sum()),
        },
    ]

    class FakeDepthRunner:
        def __init__(self, config) -> None:
            self.config = config

        def infer_depth(self, image: np.ndarray) -> np.ndarray:
            return depth_map

    class FakeSamRunner:
        def __init__(self, config) -> None:
            self.config = config

        def generate_masks(self, image: np.ndarray) -> list[dict]:
            return candidates

    monkeypatch.setattr("visualiser.segmenter.floor_segmenter.DepthAnythingRunner", FakeDepthRunner)
    monkeypatch.setattr("visualiser.segmenter.floor_segmenter.Sam2AutomaticMaskRunner", FakeSamRunner)

    app_config = AppConfig(
        project_root=Path.cwd(),
        checkpoints_dir=Path.cwd() / "checkpoints",
        data_dir=Path.cwd() / "data",
        outputs_dir=Path.cwd() / "outputs",
        sam2_repo_dir=None,
        sam2_checkpoint=None,
        sam2_model_config=None,
        depth_backend="repo",
        depth_anything_repo_dir=None,
        depth_anything_checkpoint=None,
        depth_anything_encoder="vits",
        sample_room_image=None,
    )

    segmenter = AutomaticFloorSegmenter(
        app_config,
        AutomaticFloorSegmenterConfig(
            room_image=Path("room.png"),
            output_dir=Path("outputs"),
            segmentation_max_side=width,
        ),
    )

    result = segmenter.segment(room_rgb)

    assert result.selected_mask[94, 70]
    assert not result.visible_mask[94, 114]
    assert result.occluder_mask[94, 114]
    assert result.debug_images["recovered_mask"][94, 70] == 255
    assert result.debug_images["visible_mask"][94, 114] == 0
    assert result.recovery_summary["accepted_component_count"] >= 1
    assert any(
        summary["index"] == 2 and summary["phase27_recovered_pixels"] > 0
        for summary in result.candidate_summaries
    )


def test_segment_phase28_inpaints_touching_object_and_restores_it(monkeypatch) -> None:
    height, width = 120, 160
    room_rgb = np.full((height, width, 3), 228, dtype=np.uint8)

    floor_color = np.array([150, 114, 78], dtype=np.uint8)
    object_color = np.array([36, 64, 144], dtype=np.uint8)
    room_rgb[84:, :] = floor_color
    room_rgb[88:106, 60:96] = object_color

    depth_map = np.tile(np.linspace(0.2, 1.0, height, dtype=np.float32).reshape(height, 1), (1, width))

    main_floor_mask = np.zeros((height, width), dtype=bool)
    main_floor_mask[84:, :] = True
    main_floor_mask[88:106, 60:118] = False

    touching_object_mask = np.zeros((height, width), dtype=bool)
    touching_object_mask[88:106, 60:96] = True

    expanded_floor_mask = np.zeros((height, width), dtype=bool)
    expanded_floor_mask[84:, :] = True

    first_pass_candidates = [
        {
            "segmentation": main_floor_mask,
            "predicted_iou": 0.91,
            "stability_score": 0.94,
            "bbox": [0, 84, width, height - 84],
            "area": int(main_floor_mask.sum()),
        },
        {
            "segmentation": touching_object_mask,
            "predicted_iou": 0.95,
            "stability_score": 0.95,
            "bbox": [60, 88, 36, 18],
            "area": int(touching_object_mask.sum()),
        },
    ]
    second_pass_candidates = [
        {
            "segmentation": expanded_floor_mask,
            "predicted_iou": 0.89,
            "stability_score": 0.92,
            "bbox": [0, 84, width, height - 84],
            "area": int(expanded_floor_mask.sum()),
        }
    ]

    class FakeDepthRunner:
        def __init__(self, config) -> None:
            self.config = config

        def infer_depth(self, image: np.ndarray) -> np.ndarray:
            return depth_map

    class FakeSamRunner:
        def __init__(self, config) -> None:
            self.config = config
            self.call_count = 0

        def generate_masks(self, image: np.ndarray) -> list[dict]:
            self.call_count += 1
            if self.call_count == 1:
                return first_pass_candidates
            return second_pass_candidates

    monkeypatch.setattr("visualiser.segmenter.floor_segmenter.DepthAnythingRunner", FakeDepthRunner)
    monkeypatch.setattr("visualiser.segmenter.floor_segmenter.Sam2AutomaticMaskRunner", FakeSamRunner)

    app_config = AppConfig(
        project_root=Path.cwd(),
        checkpoints_dir=Path.cwd() / "checkpoints",
        data_dir=Path.cwd() / "data",
        outputs_dir=Path.cwd() / "outputs",
        sam2_repo_dir=None,
        sam2_checkpoint=None,
        sam2_model_config=None,
        depth_backend="repo",
        depth_anything_repo_dir=None,
        depth_anything_checkpoint=None,
        depth_anything_encoder="vits",
        sample_room_image=None,
    )

    segmenter = AutomaticFloorSegmenter(
        app_config,
        AutomaticFloorSegmenterConfig(
            room_image=Path("room.png"),
            output_dir=Path("outputs"),
            segmentation_max_side=width,
        ),
    )

    result = segmenter.segment(room_rgb)

    assert result.occluder_mask[94, 72]
    assert result.selected_mask[94, 72]
    assert not result.visible_mask[94, 72]
    assert result.selected_mask[94, 108]
    assert result.visible_mask[94, 108]
    assert result.phase28_summary["ran_second_pass"] is True
    assert result.phase28_summary["occluder_area"] > 0
    assert result.phase28_summary["recovered_area"] > 0
    assert "phase28_inpainted_room" in result.debug_images
    assert any(
        summary["index"] == 1 and summary["phase28_occluder_pixels"] > 0
        for summary in result.candidate_summaries
    )
