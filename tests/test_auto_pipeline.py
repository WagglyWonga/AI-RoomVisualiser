from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from visualiser.auto_pipeline import AutoVisualiserConfig, run_auto_visualiser
from visualiser.config import AppConfig
from visualiser.io_utils import ensure_directory, save_image
from visualiser.pipeline import FloorReplacementArtifacts
from visualiser.segmenter.floor_segmenter import FloorSegmentationResult


def test_run_auto_visualiser_passes_occluder_mask_to_preserve_mask(monkeypatch) -> None:
    tmp_path = ensure_directory(Path.cwd() / ".tmp" / f"test-auto-pipeline-{uuid4().hex}")
    room_rgb = np.full((16, 16, 3), 180, dtype=np.uint8)
    flooring_rgb = np.full((16, 16, 3), 120, dtype=np.uint8)

    room_path = tmp_path / "room.png"
    flooring_path = tmp_path / "flooring.png"
    output_dir = tmp_path / "outputs"
    save_image(room_path, room_rgb)
    save_image(flooring_path, flooring_rgb)

    selected_mask = np.zeros((16, 16), dtype=bool)
    selected_mask[8:, :] = True
    occluder_mask = np.zeros((16, 16), dtype=bool)
    occluder_mask[9:13, 5:9] = True

    fake_segmentation_result = FloorSegmentationResult(
        selected_mask=selected_mask,
        selected_quad=np.array([[0, 8], [15, 8], [15, 15], [0, 15]], dtype=np.float32),
        visible_mask=selected_mask & ~occluder_mask,
        occluder_mask=occluder_mask,
        depth_map=np.zeros((16, 16), dtype=np.float32),
        candidate_summaries=[],
        recovery_summary={},
        phase28_summary={"enabled": True, "ran_second_pass": True},
        debug_images={},
    )

    class FakeSegmenter:
        def __init__(self, app_config, config) -> None:
            self.app_config = app_config
            self.config = config

        def segment(self, room_rgb: np.ndarray) -> FloorSegmentationResult:
            return fake_segmentation_result

    captured: dict[str, np.ndarray] = {}

    def fake_build_floor_replacement_artifacts(
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
        captured["mask"] = mask.copy()
        captured["preserve_mask"] = preserve_mask.copy() if preserve_mask is not None else None
        return FloorReplacementArtifacts(
            final_rgb=room_rgb.copy(),
            quad=quad if quad is not None else np.zeros((4, 2), dtype=np.float32),
            homography=np.eye(3, dtype=np.float32),
            debug_images={},
        )

    monkeypatch.setattr("visualiser.auto_pipeline.AutomaticFloorSegmenter", FakeSegmenter)
    monkeypatch.setattr(
        "visualiser.auto_pipeline.build_floor_replacement_artifacts",
        fake_build_floor_replacement_artifacts,
    )

    app_config = AppConfig(
        project_root=tmp_path,
        checkpoints_dir=tmp_path / "checkpoints",
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs-root",
        sam2_repo_dir=None,
        sam2_checkpoint=None,
        sam2_model_config=None,
        depth_backend="repo",
        depth_anything_repo_dir=None,
        depth_anything_checkpoint=None,
        depth_anything_encoder="vits",
        sample_room_image=None,
    )

    result = run_auto_visualiser(
        app_config,
        AutoVisualiserConfig(
            room_image=room_path,
            flooring_image=flooring_path,
            output_dir=output_dir,
            save_debug_images=False,
        ),
    )

    assert result.final_image_path.exists()
    assert captured["mask"].sum() == int(selected_mask.sum())
    assert captured["preserve_mask"] is not None
    assert np.array_equal(captured["preserve_mask"], occluder_mask)
