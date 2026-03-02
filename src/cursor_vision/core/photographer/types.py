from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def bbox_xyxy_to_cxcywh(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return cx, cy, w, h


@dataclass(frozen=True)
class DetectionData:
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float
    extra_data: dict[str, object] = field(default_factory=dict)
    tracking_id: int | None = None


@dataclass
class FrameMetrics:
    frame_index: int
    image_w: int
    image_h: int
    area_bbox: tuple[float, float, float, float] | None
    area_bbox_raw: tuple[float, float, float, float] | None
    area_class_name: str | None
    area_confidence: float | None
    area_stable_frames: int
    count_in_area: int
    coverage_ratio: float
    movement_score: float
    occlusion_ratio: float
    has_person_near: bool
    class_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "image_w": self.image_w,
            "image_h": self.image_h,
            "area_bbox": (
                list(bbox_xyxy_to_cxcywh(self.area_bbox)) if self.area_bbox else None
            ),
            "area_bbox_raw": (
                list(bbox_xyxy_to_cxcywh(self.area_bbox_raw))
                if self.area_bbox_raw
                else None
            ),
            "area_class_name": self.area_class_name,
            "area_confidence": self.area_confidence,
            "area_stable_frames": self.area_stable_frames,
            "count_in_area": self.count_in_area,
            "coverage_ratio": self.coverage_ratio,
            "movement_score": self.movement_score,
            "occlusion_ratio": self.occlusion_ratio,
            "has_person_near": self.has_person_near,
            "class_counts": dict(self.class_counts),
        }


@dataclass
class KeyframeEvent:
    event_type: str
    frame_index: int
    image_path: str
    metrics: FrameMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "frame_index": self.frame_index,
            "image_path": self.image_path,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class KeyframeSaveRequest:
    event_type: str
    frame_index: int
    image: object
    metrics: FrameMetrics
    detections: Sequence[DetectionData]
    filename_tag: str | None = None
    folder_tag: str | None = None
    filename_override: str | None = None
    original_image_path: str | Path | None = None
    original_image: object | None = None
    original_depth: object | None = None
    kf_test_skip: bool = False

    def resolved_filename(self) -> str:
        if self.filename_override:
            filename = self.filename_override
            if not filename.lower().endswith(".jpg"):
                filename = f"{filename}.jpg"
            return filename
        tag = self.filename_tag or self.event_type
        return f"{tag}_{self.frame_index:06d}.jpg"

    def resolved_folder_name(self) -> str:
        return self.folder_tag or self.event_type

    def event_group_stage(self) -> tuple[str | None, str | None]:
        if not self.filename_override:
            return None, None
        filename_stem = Path(self.resolved_filename()).stem
        if "_" not in filename_stem:
            return None, None
        parts = filename_stem.split("_")
        if len(parts) < 3:
            return None, None
        return parts[0], parts[1]


@dataclass(frozen=True)
class KeyframeSignal:
    request: KeyframeSaveRequest
    event_group: str | None
    event_stage: str | None
    is_debug_event: bool
    is_kfs_final: bool
    kfs_overwrite: bool
    is_kf_test_candidate: bool
    kf_test_overwrite: bool
    kf_test_requires_count_change: bool
