"""Blend helpers for the manual floor replacement MVP."""

from __future__ import annotations

import cv2
import numpy as np


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_float = mask.astype(np.float32)
    if radius <= 0:
        return mask_float

    sigma = max(radius / 2.0, 0.1)
    blurred = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def match_masked_luminance(
    room_rgb: np.ndarray,
    warped_texture_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if not np.any(mask):
        return warped_texture_rgb

    room_gray = cv2.cvtColor(room_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    warped_gray = cv2.cvtColor(warped_texture_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    target_mean = float(room_gray[mask].mean())
    texture_mean = float(max(warped_gray[mask].mean(), 1.0))
    gain = target_mean / texture_mean
    adjusted = np.clip(warped_texture_rgb.astype(np.float32) * gain, 0.0, 255.0)
    return adjusted.astype(np.uint8)


def build_shading_map(room_rgb: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    gray = cv2.cvtColor(room_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    soft_lighting = cv2.GaussianBlur(gray, (0, 0), sigmaX=31.0, sigmaY=31.0)
    reference = float(max(soft_lighting[mask].mean() if np.any(mask) else soft_lighting.mean(), 1e-3))
    normalized = soft_lighting / reference
    shading = 1.0 + strength * (normalized - 1.0)
    return np.clip(shading, 0.55, 1.45)


def apply_shading(texture_rgb: np.ndarray, shading_map: np.ndarray) -> np.ndarray:
    shaded = texture_rgb.astype(np.float32) * shading_map[..., None]
    return np.clip(shaded, 0.0, 255.0).astype(np.uint8)


def composite_floor(
    room_rgb: np.ndarray,
    floor_rgb: np.ndarray,
    alpha_mask: np.ndarray,
) -> np.ndarray:
    alpha = np.clip(alpha_mask.astype(np.float32), 0.0, 1.0)[..., None]
    composite = room_rgb.astype(np.float32) * (1.0 - alpha) + floor_rgb.astype(np.float32) * alpha
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)
