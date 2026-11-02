"""Batch runner for testing multiple room and flooring combinations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import os
from pathlib import Path
from typing import Sequence

from .auto_pipeline import AutoVisualiserConfig, run_auto_visualiser
from .config import AppConfig
from .io_utils import ensure_directory, save_json_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class BatchAutoVisualiserConfig:
    rooms_dir: Path
    floors_dir: Path
    output_dir: Path
    env_file: Path | None = None
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
    texture_scale: float = 1.0
    texture_rotation_degrees: float = 0.0
    feather_radius: int = 10
    shading_strength: float = 0.7
    save_debug_images: bool = True
    max_runs: int | None = None


@dataclass(slots=True)
class BatchRunResult:
    room_image: Path
    flooring_image: Path
    output_dir: Path
    status: str
    final_image_path: Path | None = None
    metadata_path: Path | None = None
    error: str | None = None


def find_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory of images: {directory}")

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.name.lower())


def build_output_dir_name(room_image: Path, flooring_image: Path) -> str:
    return f"{room_image.stem}-{flooring_image.stem}"


def build_run_pairs(rooms: list[Path], floors: list[Path]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for flooring_image in floors:
        for room_image in rooms:
            pairs.append((room_image, flooring_image))
    return pairs


def _serialize_result(result: BatchRunResult) -> dict[str, str | None]:
    return {
        "room_image": str(result.room_image),
        "flooring_image": str(result.flooring_image),
        "output_dir": str(result.output_dir),
        "status": result.status,
        "final_image_path": str(result.final_image_path) if result.final_image_path else None,
        "metadata_path": str(result.metadata_path) if result.metadata_path else None,
        "error": result.error,
    }


def _relative_href(root: Path, target: Path | None) -> str | None:
    if target is None:
        return None
    return Path(os.path.relpath(target, root)).as_posix()


def save_batch_summary_csv(output_root: Path, results: list[BatchRunResult]) -> Path:
    csv_path = output_root / "batch-auto-visualise-summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "room_image",
                "flooring_image",
                "output_dir",
                "status",
                "final_image_path",
                "metadata_path",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(_serialize_result(result))
    return csv_path


def build_html_gallery(output_root: Path, results: list[BatchRunResult]) -> str:
    successful_runs = sum(result.status == "success" for result in results)
    failed_runs = sum(result.status == "failed" for result in results)
    cards: list[str] = []

    for result in results:
        room_name = escape(result.room_image.name)
        floor_name = escape(result.flooring_image.name)
        combo_name = escape(result.output_dir.name)
        status_class = "success" if result.status == "success" else "failed"
        status_label = escape(result.status.upper())
        output_href = _relative_href(output_root, result.output_dir)
        final_href = _relative_href(output_root, result.final_image_path)
        metadata_href = _relative_href(output_root, result.metadata_path)
        room_href = _relative_href(output_root, result.room_image)
        floor_href = _relative_href(output_root, result.flooring_image)
        error_html = (
            f"<p class=\"error\">{escape(result.error)}</p>"
            if result.error
            else ""
        )

        if final_href:
            preview_html = (
                f"<a class=\"preview-link\" href=\"{escape(final_href)}\">"
                f"<img src=\"{escape(final_href)}\" alt=\"Preview for {combo_name}\">"
                "</a>"
            )
        else:
            preview_html = "<div class=\"preview missing\">Run failed</div>"

        links: list[str] = []
        if output_href:
            links.append(f"<a href=\"{escape(output_href)}\">output folder</a>")
        if final_href:
            links.append(f"<a href=\"{escape(final_href)}\">final image</a>")
        if metadata_href:
            links.append(f"<a href=\"{escape(metadata_href)}\">run.json</a>")
        if room_href:
            links.append(f"<a href=\"{escape(room_href)}\">room image</a>")
        if floor_href:
            links.append(f"<a href=\"{escape(floor_href)}\">floor image</a>")

        cards.append(
            "\n".join(
                [
                    f"<article class=\"card {status_class}\">",
                    preview_html,
                    "<div class=\"card-body\">",
                    f"<div class=\"status {status_class}\">{status_label}</div>",
                    f"<h2>{combo_name}</h2>",
                    f"<p><strong>Room:</strong> {room_name}</p>",
                    f"<p><strong>Floor:</strong> {floor_name}</p>",
                    f"<p><strong>Folder:</strong> {combo_name}</p>",
                    error_html,
                    f"<p class=\"links\">{' | '.join(links)}</p>",
                    "</div>",
                    "</article>",
                ]
            )
        )

    cards_markup = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Batch Auto Visualise Summary</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe8;
      --panel: #fffdf8;
      --ink: #2f261f;
      --muted: #6f6256;
      --line: #ded1c2;
      --success: #2d8a52;
      --success-bg: #e8f5ec;
      --failed: #b2452f;
      --failed-bg: #fbeae6;
      --shadow: 0 10px 24px rgba(68, 48, 28, 0.08);
      --radius: 18px;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f8f3ec 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      margin-bottom: 28px;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: radial-gradient(circle at top left, #fff7ee 0%, var(--panel) 55%);
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0 0 18px;
      color: var(--muted);
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .stat {{
      min-width: 140px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
    }}
    .stat strong {{
      display: block;
      font-size: 1.35rem;
      margin-bottom: 4px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 18px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .preview-link,
    .preview {{
      display: block;
      aspect-ratio: 4 / 3;
      background: #efe5d8;
    }}
    .preview img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .preview.missing {{
      display: grid;
      place-items: center;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .card-body {{
      padding: 16px 18px 18px;
    }}
    .card-body h2 {{
      margin: 0 0 12px;
      font-size: 1.15rem;
      line-height: 1.2;
    }}
    .card-body p {{
      margin: 0 0 8px;
      color: var(--muted);
    }}
    .status {{
      display: inline-block;
      margin-bottom: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .status.success {{
      color: var(--success);
      background: var(--success-bg);
    }}
    .status.failed {{
      color: var(--failed);
      background: var(--failed-bg);
    }}
    .error {{
      color: var(--failed);
      font-weight: 600;
    }}
    .links {{
      margin-top: 14px;
      font-size: 0.95rem;
    }}
    .links a {{
      color: #624f3b;
    }}
    @media (max-width: 640px) {{
      main {{
        padding-left: 14px;
        padding-right: 14px;
      }}
      .hero {{
        padding: 20px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Batch Auto Visualise Summary</h1>
      <p>Review every room and flooring combination from the latest batch run.</p>
      <div class="stats">
        <div class="stat"><strong>{len(results)}</strong><span>Total runs</span></div>
        <div class="stat"><strong>{successful_runs}</strong><span>Successful</span></div>
        <div class="stat"><strong>{failed_runs}</strong><span>Failed</span></div>
      </div>
    </section>
    <section class="gallery">
      {cards_markup}
    </section>
  </main>
</body>
</html>
"""


def save_batch_summary_gallery(output_root: Path, results: list[BatchRunResult]) -> Path:
    html_path = output_root / "batch-auto-visualise-gallery.html"
    html_path.write_text(build_html_gallery(output_root, results), encoding="utf-8")
    return html_path


def run_batch_auto_visualiser(config: BatchAutoVisualiserConfig) -> list[BatchRunResult]:
    app_config = AppConfig.from_env(env_file=config.env_file)
    rooms = find_image_files(config.rooms_dir)
    floors = find_image_files(config.floors_dir)
    if not rooms:
        raise RuntimeError(f"No room images found in {config.rooms_dir}")
    if not floors:
        raise RuntimeError(f"No flooring images found in {config.floors_dir}")

    output_root = ensure_directory(config.output_dir)
    run_pairs = build_run_pairs(rooms, floors)
    if config.max_runs is not None:
        run_pairs = run_pairs[: config.max_runs]

    results: list[BatchRunResult] = []
    total_runs = len(run_pairs)

    for run_index, (room_image, flooring_image) in enumerate(run_pairs, start=1):
        combo_output_dir = output_root / build_output_dir_name(room_image, flooring_image)
        ensure_directory(combo_output_dir)
        print(
            f"[{run_index}/{total_runs}] "
            f"{room_image.name} x {flooring_image.name} -> {combo_output_dir.name}"
        )

        try:
            run_result = run_auto_visualiser(
                app_config,
                AutoVisualiserConfig(
                    room_image=room_image.resolve(),
                    flooring_image=flooring_image.resolve(),
                    output_dir=combo_output_dir.resolve(),
                    device=config.device,
                    segmentation_max_side=config.segmentation_max_side,
                    depth_input_size=config.depth_input_size,
                    sam_points_per_side=config.sam_points_per_side,
                    sam_points_per_batch=config.sam_points_per_batch,
                    sam_pred_iou_thresh=config.sam_pred_iou_thresh,
                    sam_stability_score_thresh=config.sam_stability_score_thresh,
                    sam_box_nms_thresh=config.sam_box_nms_thresh,
                    sam_crop_n_layers=config.sam_crop_n_layers,
                    min_mask_region_area=config.min_mask_region_area,
                    top_candidate_count=config.top_candidate_count,
                    phase27_max_gap_ratio=config.phase27_max_gap_ratio,
                    phase27_min_component_area_ratio=config.phase27_min_component_area_ratio,
                    phase27_min_score=config.phase27_min_score,
                    phase27_min_depth_score=config.phase27_min_depth_score,
                    phase27_min_chroma_score=config.phase27_min_chroma_score,
                    phase27_min_proximity_score=config.phase27_min_proximity_score,
                    phase27_min_row_alignment=config.phase27_min_row_alignment,
                    texture_scale=config.texture_scale,
                    texture_rotation_degrees=config.texture_rotation_degrees,
                    feather_radius=config.feather_radius,
                    shading_strength=config.shading_strength,
                    save_debug_images=config.save_debug_images,
                ),
            )
            results.append(
                BatchRunResult(
                    room_image=room_image,
                    flooring_image=flooring_image,
                    output_dir=combo_output_dir,
                    status="success",
                    final_image_path=run_result.final_image_path,
                    metadata_path=run_result.metadata_path,
                )
            )
        except Exception as exc:
            results.append(
                BatchRunResult(
                    room_image=room_image,
                    flooring_image=flooring_image,
                    output_dir=combo_output_dir,
                    status="failed",
                    error=str(exc),
                )
            )
            print(f"  failed: {exc}")

    summary_path = output_root / "batch-auto-visualise-summary.json"
    save_json_file(
        summary_path,
        {
            "rooms_dir": str(config.rooms_dir),
            "floors_dir": str(config.floors_dir),
            "output_dir": str(output_root),
            "total_runs": total_runs,
            "successful_runs": sum(result.status == "success" for result in results),
            "failed_runs": sum(result.status == "failed" for result in results),
            "results": [_serialize_result(result) for result in results],
        },
    )
    csv_path = save_batch_summary_csv(output_root, results)
    html_path = save_batch_summary_gallery(output_root, results)
    print(f"Saved batch summary: {summary_path}")
    print(f"Saved batch CSV: {csv_path}")
    print(f"Saved batch gallery: {html_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch-auto-visualise",
        description="Run the automatic room visualiser across every room and flooring combination.",
    )
    parser.add_argument("--env-file", type=Path, default=None, help="Optional path to a .env file")
    parser.add_argument("--rooms-dir", type=Path, default=Path("inputs/rooms"), help="Directory of room images")
    parser.add_argument("--floors-dir", type=Path, default=Path("inputs/floors"), help="Directory of flooring images")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to write per-combination outputs")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Select the runtime device for SAM2 and Depth Anything.",
    )
    parser.add_argument(
        "--segmentation-max-side",
        type=int,
        default=960,
        help="Resize the room image so its longest side is at most this size before segmentation.",
    )
    parser.add_argument(
        "--depth-input-size",
        type=int,
        default=518,
        help="Depth Anything input size before upsampling back to the room image size.",
    )
    parser.add_argument(
        "--sam-points-per-side",
        type=int,
        default=16,
        help="SAM2 automatic mask grid size. Lower is faster, higher may be more detailed.",
    )
    parser.add_argument(
        "--sam-points-per-batch",
        type=int,
        default=32,
        help="How many SAM2 prompt points to process at once.",
    )
    parser.add_argument(
        "--sam-pred-iou-thresh",
        type=float,
        default=0.7,
        help="SAM2 predicted-IoU threshold for keeping candidate masks.",
    )
    parser.add_argument(
        "--sam-stability-thresh",
        type=float,
        default=0.9,
        help="SAM2 stability threshold for keeping candidate masks.",
    )
    parser.add_argument(
        "--sam-box-nms-thresh",
        type=float,
        default=0.7,
        help="SAM2 NMS threshold for duplicate mask suppression.",
    )
    parser.add_argument(
        "--sam-crop-n-layers",
        type=int,
        default=0,
        help="Extra SAM2 crop layers. Higher may improve detail but costs more time.",
    )
    parser.add_argument(
        "--min-mask-region-area",
        type=int,
        default=400,
        help="Minimum region area used during SAM2 mask post-processing.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=5,
        help="How many ranked segmentation candidates to save in debug output.",
    )
    parser.add_argument(
        "--phase27-max-gap-ratio",
        type=float,
        default=0.08,
        help="Maximum gap, as a fraction of image size, for recovering disconnected floor islands.",
    )
    parser.add_argument(
        "--phase27-min-component-area-ratio",
        type=float,
        default=0.00015,
        help="Minimum component area ratio for considering a disconnected floor island.",
    )
    parser.add_argument(
        "--phase27-min-score",
        type=float,
        default=0.58,
        help="Minimum combined Phase 2.7 recovery score required to accept a floor island.",
    )
    parser.add_argument(
        "--phase27-min-depth-score",
        type=float,
        default=0.5,
        help="Minimum depth-consistency score required to accept a recovered floor island.",
    )
    parser.add_argument(
        "--phase27-min-chroma-score",
        type=float,
        default=0.45,
        help="Minimum color-consistency score required to accept a recovered floor island.",
    )
    parser.add_argument(
        "--phase27-min-proximity-score",
        type=float,
        default=0.25,
        help="Minimum proximity score required to accept a recovered floor island.",
    )
    parser.add_argument(
        "--phase27-min-row-alignment",
        type=float,
        default=0.35,
        help="Minimum row-depth alignment score required to accept a recovered floor island.",
    )
    parser.add_argument(
        "--texture-scale",
        type=float,
        default=1.0,
        help="Scales the flooring texture before tiling. Values above 1 make planks or tiles larger.",
    )
    parser.add_argument(
        "--texture-rotation",
        type=float,
        default=0.0,
        help="Rotates the tiled flooring texture in degrees before perspective warping.",
    )
    parser.add_argument(
        "--feather-radius",
        type=int,
        default=10,
        help="Soft edge radius for blending the floor back into the room image.",
    )
    parser.add_argument(
        "--shading-strength",
        type=float,
        default=0.7,
        help="How strongly to preserve room lighting when blending the flooring texture.",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Skip saving intermediate debug images.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of room/floor combinations to run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results = run_batch_auto_visualiser(
        BatchAutoVisualiserConfig(
            rooms_dir=args.rooms_dir.resolve(),
            floors_dir=args.floors_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            env_file=args.env_file.resolve() if args.env_file else None,
            device=args.device,
            segmentation_max_side=args.segmentation_max_side,
            depth_input_size=args.depth_input_size,
            sam_points_per_side=args.sam_points_per_side,
            sam_points_per_batch=args.sam_points_per_batch,
            sam_pred_iou_thresh=args.sam_pred_iou_thresh,
            sam_stability_score_thresh=args.sam_stability_thresh,
            sam_box_nms_thresh=args.sam_box_nms_thresh,
            sam_crop_n_layers=args.sam_crop_n_layers,
            min_mask_region_area=args.min_mask_region_area,
            top_candidate_count=args.top_candidates,
            phase27_max_gap_ratio=args.phase27_max_gap_ratio,
            phase27_min_component_area_ratio=args.phase27_min_component_area_ratio,
            phase27_min_score=args.phase27_min_score,
            phase27_min_depth_score=args.phase27_min_depth_score,
            phase27_min_chroma_score=args.phase27_min_chroma_score,
            phase27_min_proximity_score=args.phase27_min_proximity_score,
            phase27_min_row_alignment=args.phase27_min_row_alignment,
            texture_scale=args.texture_scale,
            texture_rotation_degrees=args.texture_rotation,
            feather_radius=args.feather_radius,
            shading_strength=args.shading_strength,
            save_debug_images=not args.no_debug_images,
            max_runs=args.max_runs,
        )
    )
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
