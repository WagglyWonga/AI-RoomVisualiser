"""Preflight and smoke-test helpers for the project."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import platform
import sys

from .config import AppConfig


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    message: str
    recommendation: str | None = None


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


def _module_available(module_name: str, extra_sys_path: Path | None = None) -> bool:
    with _prepend_sys_path(extra_sys_path):
        return find_spec(module_name) is not None


def _status_prefix(status: str) -> str:
    return {
        "ok": "[OK]",
        "warn": "[WARN]",
        "fail": "[FAIL]",
    }.get(status, "[INFO]")


def _path_result(name: str, path: Path | None, missing_message: str) -> CheckResult:
    if path is None:
        return CheckResult(
            name=name,
            status="warn",
            message=missing_message,
        )

    if path.exists():
        return CheckResult(name=name, status="ok", message=f"Found at {path}")

    return CheckResult(name=name, status="warn", message=f"Configured path does not exist: {path}")


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _sam2_smoke_test(config: AppConfig, image_path: Path | None) -> CheckResult:
    if not config.sam2_checkpoint or not config.sam2_checkpoint.exists():
        return CheckResult(
            name="smoke:sam2",
            status="warn",
            message="SAM2 checkpoint is not available for model loading.",
        )

    if not config.sam2_model_config:
        return CheckResult(
            name="smoke:sam2",
            status="warn",
            message="SAM2 model config is not configured.",
        )

    try:
        import numpy as np
        from PIL import Image
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(
            build_sam2(
                config.sam2_model_config,
                str(config.sam2_checkpoint),
                device="cpu",
            )
        )

        if image_path is not None:
            image = np.array(Image.open(image_path).convert("RGB"))
            predictor.set_image(image)
            point_coords = np.array(
                [[image.shape[1] / 2.0, image.shape[0] / 2.0]],
                dtype=np.float32,
            )
            point_labels = np.array([1], dtype=np.int32)
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            return CheckResult(
                name="smoke:sam2",
                status="ok",
                message=f"Loaded checkpoint and produced mask output with shape {tuple(masks.shape)} on cpu.",
            )

        return CheckResult(
            name="smoke:sam2",
            status="ok",
            message="Loaded SAM2 image predictor on cpu.",
        )
    except Exception as exc:
        return CheckResult(
            name="smoke:sam2",
            status="fail",
            message=_format_exception(exc),
        )


def _depth_anything_smoke_test(config: AppConfig, image_path: Path | None) -> CheckResult:
    if not config.depth_anything_checkpoint or not config.depth_anything_checkpoint.exists():
        return CheckResult(
            name="smoke:depth-anything",
            status="warn",
            message="Depth Anything V2 checkpoint is not available for model loading.",
        )

    if not config.depth_anything_repo_dir or not config.depth_anything_repo_dir.exists():
        return CheckResult(
            name="smoke:depth-anything",
            status="warn",
            message="Depth Anything V2 repo directory is not available for import.",
        )

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    if config.depth_anything_encoder not in model_configs:
        return CheckResult(
            name="smoke:depth-anything",
            status="fail",
            message=f"Unsupported Depth Anything encoder: {config.depth_anything_encoder}",
        )

    try:
        import cv2
        import torch

        with _prepend_sys_path(config.depth_anything_repo_dir):
            from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(**model_configs[config.depth_anything_encoder])
        state_dict = torch.load(str(config.depth_anything_checkpoint), map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to("cpu").eval()

        if image_path is not None:
            raw_image = cv2.imread(str(image_path))
            if raw_image is None:
                return CheckResult(
                    name="smoke:depth-anything",
                    status="fail",
                    message=f"OpenCV could not read the sample image: {image_path}",
                )

            with torch.inference_mode():
                depth = model.infer_image(raw_image)

            return CheckResult(
                name="smoke:depth-anything",
                status="ok",
                message=f"Loaded checkpoint and produced depth output with shape {tuple(depth.shape)} on cpu.",
            )

        return CheckResult(
            name="smoke:depth-anything",
            status="ok",
            message="Loaded Depth Anything V2 model on cpu.",
        )
    except Exception as exc:
        return CheckResult(
            name="smoke:depth-anything",
            status="fail",
            message=_format_exception(exc),
        )


def collect_preflight_report(
    config: AppConfig,
    *,
    depth_backend: str | None = None,
) -> list[CheckResult]:
    report: list[CheckResult] = []
    backend = depth_backend or config.depth_backend
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if (sys.version_info.major, sys.version_info.minor) == (3, 12):
        report.append(
            CheckResult(
                name="python-runtime",
                status="ok",
                message=f"Detected Python {python_version}",
            )
        )
    elif sys.version_info.major == 3 and sys.version_info.minor >= 10:
        report.append(
            CheckResult(
                name="python-runtime",
                status="warn",
                message=f"Detected Python {python_version}",
                recommendation="Use Python 3.12 to match the project baseline for SAM2 and PyTorch compatibility.",
            )
        )
    else:
        report.append(
            CheckResult(
                name="python-runtime",
                status="fail",
                message=f"Detected Python {python_version}",
                recommendation="Install Python 3.11 before continuing with model integration.",
            )
        )

    if platform.system() == "Windows":
        report.append(
            CheckResult(
                name="platform",
                status="warn",
                message="Running directly on Windows.",
                recommendation="Use WSL2 + Ubuntu for SAM2 installation and GPU inference.",
            )
        )
    else:
        report.append(
            CheckResult(
                name="platform",
                status="ok",
                message=f"Running on {platform.system()}",
            )
        )

    report.append(_path_result("project-root", config.project_root, "Project root is not configured."))
    report.append(_path_result("checkpoints-dir", config.checkpoints_dir, "Checkpoints directory is not configured."))
    report.append(_path_result("data-dir", config.data_dir, "Data directory is not configured."))
    report.append(_path_result("outputs-dir", config.outputs_dir, "Outputs directory is not configured."))

    for module_name in ("numpy", "cv2", "PIL"):
        if _module_available(module_name):
            report.append(CheckResult(name=f"module:{module_name}", status="ok", message="Importable"))
        else:
            report.append(
                CheckResult(
                    name=f"module:{module_name}",
                    status="warn",
                    message="Not installed in the current environment.",
                )
            )

    if _module_available("torch"):
        report.append(CheckResult(name="module:torch", status="ok", message="Importable"))
    else:
        report.append(
            CheckResult(
                name="module:torch",
                status="warn",
                message="PyTorch is not installed in the current environment.",
                recommendation="Install a matching torch/torchvision build before model inference.",
            )
        )

    if _module_available("torchvision"):
        report.append(CheckResult(name="module:torchvision", status="ok", message="Importable"))
    else:
        report.append(
            CheckResult(
                name="module:torchvision",
                status="warn",
                message="torchvision is not installed in the current environment.",
                recommendation="Install torchvision alongside torch before model inference.",
            )
        )

    if _module_available("sam2"):
        report.append(CheckResult(name="module:sam2", status="ok", message="Importable"))
    else:
        report.append(
            CheckResult(
                name="module:sam2",
                status="warn",
                message="SAM2 is not importable yet.",
                recommendation="Clone and install the official SAM2 repo with pip install -e third_party/sam2.",
            )
        )

    report.append(
        _path_result(
            "sam2-repo-dir",
            config.sam2_repo_dir,
            "SAM2_REPO_DIR is not configured.",
        )
    )
    report.append(
        _path_result(
            "sam2-checkpoint",
            config.sam2_checkpoint,
            "SAM2_CHECKPOINT is not configured yet.",
        )
    )

    if config.sam2_model_config:
        report.append(
            CheckResult(
                name="sam2-model-config",
                status="ok",
                message=f"Configured as {config.sam2_model_config}",
            )
        )
    else:
        report.append(
            CheckResult(
                name="sam2-model-config",
                status="warn",
                message="SAM2_MODEL_CONFIG is not configured yet.",
            )
        )

    if backend not in {"repo", "transformers"}:
        report.append(
            CheckResult(
                name="depth-backend",
                status="fail",
                message=f"Unsupported depth backend: {backend}",
                recommendation="Use either 'repo' or 'transformers'.",
            )
        )
        return report

    report.append(
        CheckResult(
            name="depth-backend",
            status="ok",
            message=f"Using depth backend '{backend}'",
        )
    )

    if backend == "repo":
        if _module_available("depth_anything_v2", config.depth_anything_repo_dir):
            depth_message = "Importable"
            if config.depth_anything_repo_dir and config.depth_anything_repo_dir.exists():
                depth_message = f"Importable via configured repo dir {config.depth_anything_repo_dir}"
            report.append(CheckResult(name="module:depth_anything_v2", status="ok", message=depth_message))
        else:
            report.append(
                CheckResult(
                    name="module:depth_anything_v2",
                    status="warn",
                    message="Depth Anything V2 repo package is not importable yet.",
                    recommendation="Install the repo dependencies or add the repo to the environment path.",
                )
            )

        report.append(
            _path_result(
                "depth-anything-repo-dir",
                config.depth_anything_repo_dir,
                "DEPTH_ANYTHING_REPO_DIR is not configured.",
            )
        )
        report.append(
            _path_result(
                "depth-anything-checkpoint",
                config.depth_anything_checkpoint,
                "DEPTH_ANYTHING_CHECKPOINT is not configured yet.",
            )
        )
    else:
        if _module_available("transformers"):
            report.append(CheckResult(name="module:transformers", status="ok", message="Importable"))
        else:
            report.append(
                CheckResult(
                    name="module:transformers",
                    status="warn",
                    message="Transformers is not installed in the current environment.",
                    recommendation="Install the hf extra with pip install -e \".[hf]\" if you want the Hugging Face depth backend.",
                )
            )

    report.append(
        CheckResult(
            name="depth-encoder",
            status="ok",
            message=f"Configured encoder: {config.depth_anything_encoder}",
        )
    )

    return report


def collect_smoke_test_report(
    config: AppConfig,
    *,
    depth_backend: str | None = None,
    image_path: Path | None = None,
) -> list[CheckResult]:
    report = collect_preflight_report(config, depth_backend=depth_backend)
    target_image = image_path or config.sample_room_image
    image_available = target_image is not None and target_image.exists()

    if target_image is None:
        report.append(
            CheckResult(
                name="sample-image",
                status="warn",
                message="No sample image provided for input validation.",
                recommendation="Pass --image or set SAMPLE_ROOM_IMAGE once you collect sample data.",
            )
        )
    elif not target_image.exists():
        report.append(
            CheckResult(
                name="sample-image",
                status="fail",
                message=f"Configured image does not exist: {target_image}",
            )
        )
    else:
        report.append(CheckResult(name="sample-image", status="ok", message=f"Found at {target_image}"))

        suffix = target_image.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            report.append(
                CheckResult(
                    name="sample-image-format",
                    status="warn",
                    message=f"Unusual image extension: {suffix or '<none>'}",
                    recommendation="Use a common raster image format for initial experiments.",
                )
            )
        else:
            report.append(CheckResult(name="sample-image-format", status="ok", message=f"Detected format {suffix}"))

    if _module_available("torch"):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        report.append(CheckResult(name="torch-device", status="ok", message=f"Resolved runtime device: {device}"))
    else:
        report.append(
            CheckResult(
                name="torch-device",
                status="warn",
                message="Torch is not importable, so runtime device could not be determined.",
            )
        )

    if _module_available("sam2") and config.sam2_checkpoint and config.sam2_model_config:
        report.append(_sam2_smoke_test(config, target_image if image_available else None))

    if (
        (depth_backend or config.depth_backend) == "repo"
        and _module_available("depth_anything_v2", config.depth_anything_repo_dir)
        and config.depth_anything_checkpoint
    ):
        report.append(_depth_anything_smoke_test(config, target_image if image_available else None))

    return report


def render_report(report: list[CheckResult]) -> str:
    lines = []
    for item in report:
        lines.append(f"{_status_prefix(item.status)} {item.name}: {item.message}")
        if item.recommendation:
            lines.append(f"        next: {item.recommendation}")
    return "\n".join(lines)
