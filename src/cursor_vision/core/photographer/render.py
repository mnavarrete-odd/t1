from __future__ import annotations

from typing import Sequence

from core.visualizer import VisualizerWrapper
from core.visualizer.detection import Detection as VizDetection

from .types import DetectionData


class LabelRenderer:
    def __init__(self) -> None:
        self._visualizer = VisualizerWrapper()

    def render(
        self,
        image,
        area_bbox: tuple[float, float, float, float] | None,
        area_class_name: str | None,
        extra_detections: Sequence[DetectionData] | None = None,
    ):
        labels: list[VizDetection] = []
        if area_bbox:
            label = area_class_name or "work_area"
            labels.append(
                VizDetection(
                    class_id=0,
                    class_name=label,
                    bbox=area_bbox,
                    confidence=1.0,
                    bbox_format="xyxy",
                )
            )
        if extra_detections:
            for d in extra_detections:
                labels.append(
                    VizDetection(
                        class_id=d.class_id,
                        class_name=d.class_name,
                        bbox=d.bbox,
                        confidence=d.confidence,
                        bbox_format="xyxy",
                        extra_data=dict(d.extra_data) if d.extra_data else {},
                        tracking_id=d.tracking_id,
                    )
                )
        if not labels:
            return image
        return self._visualizer.render(image, labels)
