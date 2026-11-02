"""SAM2 runtime wrapper for automatic mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Sam2AutomaticMaskConfig:
    checkpoint: Path
    model_config: str
    device: str = "auto"
    points_per_side: int = 16
    points_per_batch: int = 32
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.9
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_overlap_ratio: float = 512 / 1500
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 400
    output_mode: str = "binary_mask"
    multimask_output: bool = True


class Sam2AutomaticMaskRunner:
    def __init__(self, config: Sam2AutomaticMaskConfig) -> None:
        self.config = config
        self._generator = None

    def _load(self) -> None:
        if not self.config.checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.config.checkpoint}")

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        model = build_sam2(
            self.config.model_config,
            str(self.config.checkpoint),
            device=self.config.device,
        )
        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            box_nms_thresh=self.config.box_nms_thresh,
            crop_n_layers=self.config.crop_n_layers,
            crop_overlap_ratio=self.config.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
            min_mask_region_area=self.config.min_mask_region_area,
            output_mode=self.config.output_mode,
            multimask_output=self.config.multimask_output,
        )

    def generate_masks(self, image_rgb) -> list[dict]:
        if self._generator is None:
            self._load()

        return self._generator.generate(image_rgb)
