from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from visualiser.geometry.homography import infer_quad_from_mask, order_quad_points
from visualiser.pipeline import ManualVisualiserConfig, run_manual_visualiser


def _save_rgb_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def _save_mask_image(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def test_order_quad_points_returns_expected_clockwise_order() -> None:
    points = np.array(
        [
            [150.0, 90.0],
            [20.0, 40.0],
            [180.0, 220.0],
            [10.0, 200.0],
        ],
        dtype=np.float32,
    )

    ordered = order_quad_points(points)

    np.testing.assert_allclose(
        ordered,
        np.array(
            [
                [20.0, 40.0],
                [150.0, 90.0],
                [180.0, 220.0],
                [10.0, 200.0],
            ],
            dtype=np.float32,
        ),
    )


def test_infer_quad_from_mask_returns_four_points() -> None:
    mask = np.zeros((180, 240), dtype=bool)
    polygon = np.array([[60, 70], [180, 60], [220, 170], [20, 175]], dtype=np.int32)
    import cv2

    cv2.fillPoly(mask.astype(np.uint8), [polygon], 1)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, [polygon], 255)

    quad = infer_quad_from_mask(filled > 0)

    assert quad.shape == (4, 2)
    assert np.all(quad[:, 0] >= 0)
    assert np.all(quad[:, 1] >= 0)


def test_infer_quad_from_mask_prefers_trapezoid_for_floor_like_shape() -> None:
    import cv2

    mask = np.zeros((416, 650), dtype=np.uint8)
    floor_hull = np.array(
        [[0, 337], [72, 310], [271, 283], [405, 286], [649, 334], [649, 415], [0, 415]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [floor_hull], 255)

    quad = infer_quad_from_mask(mask > 0)
    top_width = float(np.linalg.norm(quad[1] - quad[0]))
    bottom_width = float(np.linalg.norm(quad[2] - quad[3]))

    assert quad.shape == (4, 2)
    assert top_width < bottom_width
    assert quad[0, 1] < quad[3, 1]
    assert quad[1, 1] < quad[2, 1]


def test_manual_visualiser_writes_final_and_debug_outputs() -> None:
    workspace_tmp = Path.cwd() / ".tmp" / "pytest-artifacts" / uuid4().hex
    workspace_tmp.mkdir(parents=True, exist_ok=True)

    room = np.full((180, 240, 3), 230, dtype=np.uint8)
    room[90:, :] = np.array([160, 150, 140], dtype=np.uint8)
    room[120:, :] = np.array([135, 125, 115], dtype=np.uint8)

    flooring = np.zeros((32, 32, 3), dtype=np.uint8)
    flooring[:, :16] = np.array([185, 130, 85], dtype=np.uint8)
    flooring[:, 16:] = np.array([165, 115, 70], dtype=np.uint8)
    flooring[::4, :] = np.array([120, 80, 45], dtype=np.uint8)

    mask = np.zeros((180, 240), dtype=bool)
    mask[90:, 10:230] = True

    polygon_path = workspace_tmp / "floor_polygon.json"
    polygon_path.write_text(
        json.dumps([[40, 95], [200, 90], [230, 175], [10, 178]]),
        encoding="utf-8",
    )

    room_path = workspace_tmp / "room.png"
    flooring_path = workspace_tmp / "flooring.png"
    mask_path = workspace_tmp / "mask.png"
    output_dir = workspace_tmp / "run"

    _save_rgb_image(room_path, room)
    _save_rgb_image(flooring_path, flooring)
    _save_mask_image(mask_path, mask)

    result = run_manual_visualiser(
        ManualVisualiserConfig(
            room_image=room_path,
            flooring_image=flooring_path,
            mask_image=mask_path,
            polygon_file=polygon_path,
            output_dir=output_dir,
        )
    )

    assert result.final_image_path.exists()
    assert result.metadata_path.exists()
    assert "warped_texture" in result.debug_paths
    assert result.debug_paths["warped_texture"].exists()

    final_image = np.array(Image.open(result.final_image_path).convert("RGB"))
    assert final_image.shape == room.shape
    assert not np.array_equal(final_image[150, 120], room[150, 120])
