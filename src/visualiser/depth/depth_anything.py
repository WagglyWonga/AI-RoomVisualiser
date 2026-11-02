"""Depth Anything V2 runtime wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


@contextmanager
def _prepend_sys_path(path: Path | None):
    inserted = False
    path_str = str(path) if path is not None else None
    if path_str and path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True

    try:
        yield
    finally:
        if inserted and path_str in sys.path:
            sys.path.remove(path_str)


def resolve_torch_device(device_preference: str) -> str:
    choice = device_preference.lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but no MPS device is available.")
    if choice not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device preference: {device_preference}")
    return choice


@dataclass(slots=True)
class DepthAnythingConfig:
    repo_dir: Path
    checkpoint: Path
    encoder: str = "vits"
    input_size: int = 518
    device: str = "auto"


class DepthAnythingRunner:
    def __init__(self, config: DepthAnythingConfig) -> None:
        self.config = config
        self.device = resolve_torch_device(config.device)
        self._model = None
        self._transform = None

    def _load(self) -> None:
        if self.config.encoder not in DEPTH_MODEL_CONFIGS:
            raise ValueError(f"Unsupported Depth Anything encoder: {self.config.encoder}")
        if not self.config.checkpoint.exists():
            raise FileNotFoundError(f"Depth checkpoint not found: {self.config.checkpoint}")

        with _prepend_sys_path(self.config.repo_dir):
            from depth_anything_v2.dpt import DepthAnythingV2
            from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

        self._transform = Compose(
            [
                Resize(
                    width=self.config.input_size,
                    height=self.config.input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        model = DepthAnythingV2(**DEPTH_MODEL_CONFIGS[self.config.encoder])
        state_dict = torch.load(str(self.config.checkpoint), map_location="cpu")
        model.load_state_dict(state_dict)
        self._model = model.to(self.device).eval()

    def infer_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        if self._model is None or self._transform is None:
            self._load()

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Depth inference expects an RGB image with shape HxWx3.")

        height, width = image_rgb.shape[:2]
        image = image_rgb.astype(np.float32) / 255.0
        transformed = self._transform({"image": image})["image"]
        tensor = torch.from_numpy(transformed).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            depth = self._model.forward(tensor)
            depth = F.interpolate(depth[:, None], (height, width), mode="bilinear", align_corners=True)[0, 0]

        return depth.detach().cpu().numpy()
