"""Command line entrypoints for project tooling and the Phase 1/2 MVPs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .auto_pipeline import AutoVisualiserConfig, run_auto_visualiser, run_segment_floor
from .config import AppConfig
from .pipeline import ManualVisualiserConfig, run_manual_visualiser
from .preflight import collect_preflight_report, collect_smoke_test_report, render_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visualiser", description="Room Visualiser tooling")
    subparsers = parser.add_subparsers(dest="command")

    env_parser = subparsers.add_parser("env-check", help="Run environment preflight checks")
    env_parser.add_argument("--env-file", type=Path, default=None, help="Optional path to a .env file")
    env_parser.add_argument(
        "--depth-backend",
        choices=("repo", "transformers"),
        default=None,
        help="Override the configured depth backend",
    )

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run non-destructive model readiness checks",
    )
    smoke_parser.add_argument("--env-file", type=Path, default=None, help="Optional path to a .env file")
    smoke_parser.add_argument(
        "--depth-backend",
        choices=("repo", "transformers"),
        default=None,
        help="Override the configured depth backend",
    )
    smoke_parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image to validate as a future inference input",
    )

    manual_parser = subparsers.add_parser(
        "manual-visualise",
        help="Run the Phase 1 manual-mask floor replacement MVP",
    )
    manual_parser.add_argument("--env-file", type=Path, default=None, help="Optional path to a .env file")
    manual_parser.add_argument("--room", type=Path, required=True, help="Path to the room image")
    manual_parser.add_argument("--flooring", type=Path, required=True, help="Path to the flooring texture image")
    manual_parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional grayscale floor mask image aligned to the room photo",
    )
    manual_parser.add_argument(
        "--polygon",
        type=Path,
        default=None,
        help="Optional JSON polygon file shaped like [[x, y], ...] for the floor region",
    )
    manual_parser.add_argument(
        "--quad",
        type=Path,
        default=None,
        help="Optional JSON file with four floor corner points for perspective warping",
    )
    manual_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a timestamped folder under outputs.",
    )
    manual_parser.add_argument(
        "--texture-scale",
        type=float,
        default=1.0,
        help="Scales the flooring texture before tiling. Values above 1 make planks or tiles larger.",
    )
    manual_parser.add_argument(
        "--texture-rotation",
        type=float,
        default=0.0,
        help="Rotates the tiled flooring texture in degrees before perspective warping.",
    )
    manual_parser.add_argument(
        "--feather-radius",
        type=int,
        default=10,
        help="Soft edge radius for blending the floor back into the room image.",
    )
    manual_parser.add_argument(
        "--shading-strength",
        type=float,
        default=0.7,
        help="How strongly to preserve room lighting when blending the flooring texture.",
    )
    manual_parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Skip saving intermediate debug images.",
    )

    segment_parser = subparsers.add_parser(
        "segment-floor",
        help="Run Phase 2 automatic floor segmentation and save mask debug outputs",
    )
    segment_parser.add_argument("--env-file", type=Path, default=None, help="Optional path to a .env file")
    segment_parser.add_argument("--room", type=Path, required=True, help="Path to the room image")
    segment_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a timestamped folder under outputs.",
    )
    _add_common_auto_args(segment_parser)

    auto_parser = subparsers.add_parser(
        "auto-visualise",
        help="Run Phase 2 automatic floor segmentation and generate a visualised floor replacement",
    )
    auto_parser.add_argument("--env-file", type=Path, default=None, help="Optional path to a .env file")
    auto_parser.add_argument("--room", type=Path, required=True, help="Path to the room image")
    auto_parser.add_argument("--flooring", type=Path, required=True, help="Path to the flooring texture image")
    auto_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a timestamped folder under outputs.",
    )
    _add_common_auto_args(auto_parser)
    auto_parser.add_argument(
        "--texture-scale",
        type=float,
        default=1.0,
        help="Scales the flooring texture before tiling. Values above 1 make planks or tiles larger.",
    )
    auto_parser.add_argument(
        "--texture-rotation",
        type=float,
        default=0.0,
        help="Rotates the tiled flooring texture in degrees before perspective warping.",
    )
    auto_parser.add_argument(
        "--feather-radius",
        type=int,
        default=10,
        help="Soft edge radius for blending the floor back into the room image.",
    )
    auto_parser.add_argument(
        "--shading-strength",
        type=float,
        default=0.7,
        help="How strongly to preserve room lighting when blending the flooring texture.",
    )
    auto_parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Skip saving intermediate debug images.",
    )

    return parser


def _add_common_auto_args(parser: argparse.ArgumentParser) -> None:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    config = AppConfig.from_env(env_file=args.env_file)

    if args.command == "env-check":
        report = collect_preflight_report(config, depth_backend=args.depth_backend)
    elif args.command == "smoke-test":
        report = collect_smoke_test_report(
            config,
            depth_backend=args.depth_backend,
            image_path=args.image,
        )
        print(render_report(report))
        return 1 if any(item.status == "fail" for item in report) else 0
    elif args.command == "manual-visualise":
        default_output_dir = config.outputs_dir / f"manual-visualise-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_config = ManualVisualiserConfig(
            room_image=args.room.resolve(),
            flooring_image=args.flooring.resolve(),
            output_dir=(args.output_dir.resolve() if args.output_dir else default_output_dir),
            mask_image=args.mask.resolve() if args.mask else None,
            polygon_file=args.polygon.resolve() if args.polygon else None,
            quad_file=args.quad.resolve() if args.quad else None,
            texture_scale=args.texture_scale,
            texture_rotation_degrees=args.texture_rotation,
            feather_radius=args.feather_radius,
            shading_strength=args.shading_strength,
            save_debug_images=not args.no_debug_images,
        )
        result = run_manual_visualiser(run_config)
        print(f"Saved final image: {result.final_image_path}")
        print(f"Saved run metadata: {result.metadata_path}")
        if result.debug_paths:
            print("Saved debug images:")
            for name, path in sorted(result.debug_paths.items()):
                print(f"  - {name}: {path}")
        return 0
    elif args.command == "segment-floor":
        default_output_dir = config.outputs_dir / f"segment-floor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_config = AutoVisualiserConfig(
            room_image=args.room.resolve(),
            output_dir=(args.output_dir.resolve() if args.output_dir else default_output_dir),
            flooring_image=None,
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
        )
        result = run_segment_floor(config, run_config)
        print(f"Saved selected mask: {result.final_image_path}")
        print(f"Saved segmentation metadata: {result.metadata_path}")
        if result.debug_paths:
            print("Saved debug images:")
            for name, path in sorted(result.debug_paths.items()):
                print(f"  - {name}: {path}")
        return 0
    elif args.command == "auto-visualise":
        default_output_dir = config.outputs_dir / f"auto-visualise-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_config = AutoVisualiserConfig(
            room_image=args.room.resolve(),
            flooring_image=args.flooring.resolve(),
            output_dir=(args.output_dir.resolve() if args.output_dir else default_output_dir),
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
        )
        result = run_auto_visualiser(config, run_config)
        print(f"Saved final image: {result.final_image_path}")
        print(f"Saved run metadata: {result.metadata_path}")
        if result.debug_paths:
            print("Saved debug images:")
            for name, path in sorted(result.debug_paths.items()):
                print(f"  - {name}: {path}")
        return 0
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    print(render_report(report))
    return 1 if any(item.status == "fail" for item in report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
