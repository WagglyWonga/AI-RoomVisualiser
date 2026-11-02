from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from visualiser.batch_runner import (
    BatchAutoVisualiserConfig,
    BatchRunResult,
    build_html_gallery,
    build_output_dir_name,
    find_image_files,
    run_batch_auto_visualiser,
    save_batch_summary_csv,
)
from visualiser.config import AppConfig


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def test_find_image_files_returns_sorted_supported_images() -> None:
    workspace_tmp = Path.cwd() / ".tmp" / "pytest-artifacts" / uuid4().hex
    images_dir = workspace_tmp / "images"
    _touch(images_dir / "zeta.png")
    _touch(images_dir / "alpha.jpg")
    _touch(images_dir / "notes.txt")

    files = find_image_files(images_dir)

    assert [path.name for path in files] == ["alpha.jpg", "zeta.png"]


def test_build_output_dir_name_uses_room_and_floor_stems() -> None:
    name = build_output_dir_name(Path("inputs/rooms/lounge.photo.jpg"), Path("inputs/floors/oak-plank.png"))

    assert name == "lounge.photo-oak-plank"


def test_save_batch_summary_csv_and_html_gallery_include_expected_links() -> None:
    workspace_tmp = Path.cwd() / ".tmp" / "pytest-artifacts" / uuid4().hex
    output_dir = workspace_tmp / "outputs"
    combo_dir = output_dir / "room-a-oak"
    combo_dir.mkdir(parents=True, exist_ok=True)

    result = BatchRunResult(
        room_image=workspace_tmp / "inputs" / "rooms" / "room-a.jpg",
        flooring_image=workspace_tmp / "inputs" / "floors" / "oak.jpg",
        output_dir=combo_dir,
        status="success",
        final_image_path=combo_dir / "final.png",
        metadata_path=combo_dir / "run.json",
    )

    csv_path = save_batch_summary_csv(output_dir, [result])
    html = build_html_gallery(output_dir, [result])

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "room_image,flooring_image,output_dir,status,final_image_path,metadata_path,error" in csv_text
    assert "room-a.jpg" in csv_text
    assert "oak.jpg" in csv_text

    assert "room-a-oak" in html
    assert "room-a-oak/final.png" in html
    assert "../inputs/rooms/room-a.jpg" in html
    assert "../inputs/floors/oak.jpg" in html


def test_run_batch_auto_visualiser_creates_one_output_folder_per_combination(monkeypatch) -> None:
    workspace_tmp = Path.cwd() / ".tmp" / "pytest-artifacts" / uuid4().hex
    rooms_dir = workspace_tmp / "inputs" / "rooms"
    floors_dir = workspace_tmp / "inputs" / "floors"
    output_dir = workspace_tmp / "outputs"

    room_a = rooms_dir / "room-a.jpg"
    room_b = rooms_dir / "room-b.png"
    floor_a = floors_dir / "oak.jpg"
    floor_b = floors_dir / "tile.png"
    for path in (room_a, room_b, floor_a, floor_b):
        _touch(path)

    dummy_app_config = AppConfig(
        project_root=workspace_tmp,
        checkpoints_dir=workspace_tmp / "checkpoints",
        data_dir=workspace_tmp / "data",
        outputs_dir=output_dir,
        sam2_repo_dir=None,
        sam2_checkpoint=None,
        sam2_model_config=None,
        depth_backend="repo",
        depth_anything_repo_dir=None,
        depth_anything_checkpoint=None,
        depth_anything_encoder="vits",
        sample_room_image=None,
    )

    monkeypatch.setattr("visualiser.batch_runner.AppConfig.from_env", lambda env_file=None: dummy_app_config)

    calls: list[tuple[str, str, str]] = []

    def fake_run_auto_visualiser(app_config, config):
        calls.append((config.room_image.name, config.flooring_image.name, config.output_dir.name))
        return type(
            "FakeResult",
            (),
            {
                "final_image_path": config.output_dir / "final.png",
                "metadata_path": config.output_dir / "run.json",
            },
        )()

    monkeypatch.setattr("visualiser.batch_runner.run_auto_visualiser", fake_run_auto_visualiser)

    results = run_batch_auto_visualiser(
        BatchAutoVisualiserConfig(
            rooms_dir=rooms_dir,
            floors_dir=floors_dir,
            output_dir=output_dir,
        )
    )

    assert len(results) == 4
    assert all(isinstance(result, BatchRunResult) for result in results)
    assert [result.output_dir.name for result in results] == [
        "room-a-oak",
        "room-b-oak",
        "room-a-tile",
        "room-b-tile",
    ]
    assert calls == [
        ("room-a.jpg", "oak.jpg", "room-a-oak"),
        ("room-b.png", "oak.jpg", "room-b-oak"),
        ("room-a.jpg", "tile.png", "room-a-tile"),
        ("room-b.png", "tile.png", "room-b-tile"),
    ]
    assert (output_dir / "batch-auto-visualise-summary.json").exists()
    assert (output_dir / "batch-auto-visualise-summary.csv").exists()
    assert (output_dir / "batch-auto-visualise-gallery.html").exists()
