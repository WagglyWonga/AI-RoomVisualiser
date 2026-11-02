"""Configuration loading for the room visualiser project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def _resolve_path(project_root: Path, raw_value: str | None) -> Path | None:
    if not raw_value:
        return None

    path = Path(raw_value)
    if path.is_absolute():
        return path

    return (project_root / path).resolve()


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    checkpoints_dir: Path
    data_dir: Path
    outputs_dir: Path
    sam2_repo_dir: Path | None
    sam2_checkpoint: Path | None
    sam2_model_config: str | None
    depth_backend: str
    depth_anything_repo_dir: Path | None
    depth_anything_checkpoint: Path | None
    depth_anything_encoder: str
    sample_room_image: Path | None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "AppConfig":
        defaults = {
            "PROJECT_ROOT": ".",
            "CHECKPOINTS_DIR": "checkpoints",
            "DATA_DIR": "data",
            "OUTPUTS_DIR": "outputs",
            "SAM2_REPO_DIR": "third_party/sam2",
            "SAM2_CHECKPOINT": "",
            "SAM2_MODEL_CONFIG": "",
            "DEPTH_BACKEND": "repo",
            "DEPTH_ANYTHING_REPO_DIR": "third_party/Depth-Anything-V2",
            "DEPTH_ANYTHING_CHECKPOINT": "",
            "DEPTH_ANYTHING_ENCODER": "vits",
            "SAMPLE_ROOM_IMAGE": "",
        }

        root_hint = _default_project_root()
        chosen_env_file = env_file or (root_hint / ".env")
        file_values = _parse_env_file(chosen_env_file)

        def get_value(name: str) -> str:
            return os.environ.get(name, file_values.get(name, defaults[name]))

        project_root = _resolve_path(root_hint, get_value("PROJECT_ROOT")) or root_hint

        return cls(
            project_root=project_root,
            checkpoints_dir=_resolve_path(project_root, get_value("CHECKPOINTS_DIR")) or (project_root / "checkpoints"),
            data_dir=_resolve_path(project_root, get_value("DATA_DIR")) or (project_root / "data"),
            outputs_dir=_resolve_path(project_root, get_value("OUTPUTS_DIR")) or (project_root / "outputs"),
            sam2_repo_dir=_resolve_path(project_root, get_value("SAM2_REPO_DIR")),
            sam2_checkpoint=_resolve_path(project_root, get_value("SAM2_CHECKPOINT")),
            sam2_model_config=get_value("SAM2_MODEL_CONFIG") or None,
            depth_backend=get_value("DEPTH_BACKEND") or "repo",
            depth_anything_repo_dir=_resolve_path(project_root, get_value("DEPTH_ANYTHING_REPO_DIR")),
            depth_anything_checkpoint=_resolve_path(project_root, get_value("DEPTH_ANYTHING_CHECKPOINT")),
            depth_anything_encoder=get_value("DEPTH_ANYTHING_ENCODER") or "vits",
            sample_room_image=_resolve_path(project_root, get_value("SAMPLE_ROOM_IMAGE")),
        )
