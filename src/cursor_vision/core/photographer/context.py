from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .area import WorkAreaState
from .types import DetectionData, FrameMetrics, KeyframeEvent


@dataclass
class FrameContext:
    frame_index: int
    image: object
    detections: list[DetectionData]
    metrics: FrameMetrics
    area_state: WorkAreaState | None
    area_bbox: tuple[float, float, float, float] | None
    area_ready: bool
    event_image: object
    person_dets: list[DetectionData]
    hand_dets: list[DetectionData]
    hand_near_dets: list[DetectionData]
    box_dets: list[DetectionData]
    render_labels: Callable[[Sequence[DetectionData] | None], object]
    emit_event: Callable[..., KeyframeEvent]
