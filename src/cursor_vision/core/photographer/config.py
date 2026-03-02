from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterable


_TUPLE_FIELDS = frozenset(
    {
        "area_classes",
        "box_classes",
        "person_classes",
        "hand_classes",
        "empty_classes",
        "occlusion_item_classes",
    }
)


def _coerce_path(path_value: Path | str) -> Path:
    if isinstance(path_value, Path):
        return path_value
    return Path(path_value)


def _coerce_optional_path(path_value: Path | str | None) -> Path | None:
    if path_value is None:
        return None
    return _coerce_path(path_value)


def _coerce_tuple(values: Iterable[object] | object | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        return (values,)
    try:
        return tuple(str(value) for value in values)  # type: ignore[arg-type]
    except TypeError:
        return (str(values),)


@dataclass(frozen=True)
class PhotographerConfig:
    outdir: Path
    area_classes: tuple[str, ...]
    box_classes: tuple[str, ...]
    person_classes: tuple[str, ...]
    hand_classes: tuple[str, ...]
    empty_classes: tuple[str, ...]
    depth_dir: Path | None = None
    product_kf_model: Path | None = None
    product_kf_conf: float = 0.1
    area_min_conf: float = 0.1
    area_warmup_frames: int = 10
    area_stable_frames: int = 1
    area_hold_frames: int = 30
    area_expand_ratio: float = 0.2
    area_refit_enabled: bool = True
    area_refit_frames: int = 12
    area_refit_center_dist_min_px: float = 10.0
    area_refit_center_dist_max_px: float = 30.0
    capture_all: bool = False
    clear_events: bool = True
    person_dist_px: float = 5.0
    person_near_enabled: bool = True
    hand_dist_px: float = 500.0
    stable_area_frames: int = 5
    stable_area_movement_max: float = 0.01
    stable_area_require_count_stability: bool = True
    movement_by_area: bool = False
    stable_reconfirm_frames: int = 5
    stable_empty_frames: int = 5
    occlusion_start_ratio: float = 0.2
    occlusion_end_ratio: float = 0.1
    occlusion_start_frames: int = 2
    occlusion_end_frames: int = 2
    occlusion_pre_offset_frames: int = 0
    occlusion_post_offset_frames: int = 0
    occlusion_item_classes: tuple[str, ...] = ("producto", "cajas", "folio", "manga", "saco")
    occlusion_change_count_min: int = 1
    occlusion_change_coverage_min: float = 0.05
    occlusion_change_confirm_frames: int = 2
    occlusion_change_cooldown_frames: int = 10
    product_start_frames: int = 1
    product_end_frames: int = 1
    product_pre_offset_frames: int = 0
    product_post_offset_frames: int = 0
    product_save_prepost: bool = True

    @classmethod
    def from_keyframe_settings(
        cls,
        *,
        outdir: Path | str,
        keyframe: object,
        area_classes: Iterable[str],
        depth_dir: Path | str | None = None,
        product_kf_model: Path | str | None = None,
    ) -> "PhotographerConfig":
        field_values: dict[str, object] = {}
        excluded = {"outdir", "area_classes", "depth_dir", "product_kf_model"}
        for field_info in fields(cls):
            name = field_info.name
            if name in excluded:
                continue
            if hasattr(keyframe, name):
                field_values[name] = getattr(keyframe, name)

        for field_name in _TUPLE_FIELDS:
            if field_name in field_values:
                field_values[field_name] = _coerce_tuple(field_values[field_name])

        return cls(
            outdir=_coerce_path(outdir),
            area_classes=_coerce_tuple(area_classes),
            depth_dir=_coerce_optional_path(depth_dir),
            product_kf_model=_coerce_optional_path(product_kf_model),
            **field_values,
        )

    def for_outdir(self, outdir: Path | str) -> "PhotographerConfig":
        return replace(self, outdir=_coerce_path(outdir))
