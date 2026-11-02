"""Geometry helpers for manual floor warping."""

from __future__ import annotations

import math

import cv2
import numpy as np


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected four 2D points, got shape {pts.shape}.")

    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    topmost_index = np.lexsort((ordered[:, 0], ordered[:, 1]))[0]
    return np.roll(ordered, -topmost_index, axis=0)


def rasterize_polygon(image_size: tuple[int, int], points: np.ndarray) -> np.ndarray:
    width, height = image_size
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        raise ValueError("A polygon needs at least three 2D points.")

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask > 0


def _pick_extreme_point(points: np.ndarray, *, prefer_top: bool, prefer_left: bool) -> np.ndarray:
    primary = points[:, 1] if prefer_top else -points[:, 1]
    secondary = points[:, 0] if prefer_left else -points[:, 0]
    order = np.lexsort((secondary, primary))
    return points[order[0]]


def _fit_quad_from_convex_hull(hull_points: np.ndarray) -> np.ndarray | None:
    if hull_points.shape[0] < 4:
        return None

    center_x = float(hull_points[:, 0].min() + hull_points[:, 0].max()) / 2.0
    left_points = hull_points[hull_points[:, 0] <= center_x]
    right_points = hull_points[hull_points[:, 0] >= center_x]
    if left_points.shape[0] < 2 or right_points.shape[0] < 2:
        return None

    top_left = _pick_extreme_point(left_points, prefer_top=True, prefer_left=True)
    top_right = _pick_extreme_point(right_points, prefer_top=True, prefer_left=False)
    bottom_right = _pick_extreme_point(right_points, prefer_top=False, prefer_left=False)
    bottom_left = _pick_extreme_point(left_points, prefer_top=False, prefer_left=True)

    quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    ordered = order_quad_points(quad)

    top_width = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom_width = float(np.linalg.norm(ordered[2] - ordered[3]))
    left_height = float(np.linalg.norm(ordered[3] - ordered[0]))
    right_height = float(np.linalg.norm(ordered[2] - ordered[1]))

    if top_width < 2.0 or bottom_width < 2.0 or left_height < 2.0 or right_height < 2.0:
        return None
    if ordered[0, 0] >= ordered[1, 0] or ordered[3, 0] >= ordered[2, 0]:
        return None
    if ordered[0, 1] >= ordered[3, 1] or ordered[1, 1] >= ordered[2, 1]:
        return None
    return ordered


def infer_quad_from_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find a contour in the supplied mask.")

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) == 4:
        quad = approx[:, 0, :]
    else:
        hull = cv2.convexHull(largest)
        hull_epsilon = 0.01 * cv2.arcLength(hull, True)
        hull_approx = cv2.approxPolyDP(hull, hull_epsilon, True)
        hull_points = hull_approx[:, 0, :].astype(np.float32)
        fitted = _fit_quad_from_convex_hull(hull_points)
        if fitted is not None:
            return fitted

        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect)

    return order_quad_points(quad)


def estimate_texture_canvas_size(room_shape: tuple[int, int], quad: np.ndarray) -> tuple[int, int]:
    room_height, room_width = room_shape
    top_width = np.linalg.norm(quad[1] - quad[0])
    bottom_width = np.linalg.norm(quad[2] - quad[3])
    left_height = np.linalg.norm(quad[3] - quad[0])
    right_height = np.linalg.norm(quad[2] - quad[1])

    canvas_width = int(max(room_width * 2.0, math.ceil(max(top_width, bottom_width) * 2.5), 512))
    canvas_height = int(max(room_height * 2.0, math.ceil(max(left_height, right_height) * 3.0), 512))
    return canvas_width, canvas_height


def warp_texture_to_room(
    texture_canvas: np.ndarray,
    target_quad: np.ndarray,
    room_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    room_width, room_height = room_size
    canvas_height, canvas_width = texture_canvas.shape[:2]
    source_quad = np.array(
        [
            [0.0, 0.0],
            [canvas_width - 1.0, 0.0],
            [canvas_width - 1.0, canvas_height - 1.0],
            [0.0, canvas_height - 1.0],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(source_quad, target_quad.astype(np.float32))
    warped = cv2.warpPerspective(
        texture_canvas,
        transform,
        (room_width, room_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return warped, transform
