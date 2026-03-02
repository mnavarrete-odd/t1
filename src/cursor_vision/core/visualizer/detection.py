from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Literal

BBoxFormat = Literal["xyxy", "xywh"]


@dataclass
class Detection:
    """Generic detection label structure.

    Attributes:
        class_id: Numeric class id.
        class_name: Human-readable class name.
        bbox: Bounding box in the format defined by bbox_format.
        confidence: Confidence score (0-1).
        bbox_format: "xyxy" (x1, y1, x2, y2) or "xywh" (x, y, w, h) using top-left origin.
        tracking_id: Optional tracking id.
        extra_data: Optional metadata dict.
    """

    class_id: int
    class_name: str | None
    bbox: Sequence[float]
    confidence: float
    bbox_format: BBoxFormat = "xyxy"
    tracking_id: int | None = None
    extra_data: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.class_id = int(self.class_id)
        if not self.class_name:
            self.class_name = str(self.class_id)
        self.confidence = float(self.confidence)
        if self.tracking_id is not None:
            self.tracking_id = int(self.tracking_id)
        if self.bbox_format not in ("xyxy", "xywh"):
            raise ValueError(f"Unsupported bbox_format: {self.bbox_format}")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain 4 values")
        self.bbox = tuple(float(v) for v in self.bbox)
        if self.extra_data is None:
            self.extra_data = {}

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return bbox as (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = self.bbox
        if self.bbox_format == "xywh":
            x2 = x1 + x2
            y2 = y1 + y2
        return float(x1), float(y1), float(x2), float(y2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return bbox as (x, y, w, h)."""
        x1, y1, x2, y2 = self.to_xyxy()
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, bbox_format: BBoxFormat | None = None) -> "Detection":
        """Create a Detection from a dict.

        Accepted keys:
            class_id, class_name, conf/confidence, track_id/tracking_id,
            bbox/bbox_xyxy/bbox_xywh, bbox_format, extra_data.
        """
        if "class_id" not in data:
            raise KeyError("class_id is required")

        if "bbox" in data:
            bbox = data["bbox"]
            fmt = bbox_format or data.get("bbox_format", "xyxy")
        elif "bbox_xyxy" in data:
            bbox = data["bbox_xyxy"]
            fmt = "xyxy"
        elif "bbox_xywh" in data:
            bbox = data["bbox_xywh"]
            fmt = "xywh"
        else:
            raise KeyError("bbox is required (bbox, bbox_xyxy, or bbox_xywh)")

        extra = data.get("extra_data")
        extra_data = dict(extra) if isinstance(extra, Mapping) else {}

        known_keys = {
            "class_id",
            "class_name",
            "conf",
            "confidence",
            "track_id",
            "tracking_id",
            "bbox",
            "bbox_xyxy",
            "bbox_xywh",
            "bbox_format",
            "extra_data",
        }
        for key, value in data.items():
            if key in known_keys:
                continue
            extra_data[key] = value

        return cls(
            class_id=int(data["class_id"]),
            class_name=data.get("class_name"),
            bbox=bbox,
            confidence=float(data.get("confidence", data.get("conf", 0.0))),
            bbox_format=fmt,
            tracking_id=data.get("tracking_id", data.get("track_id")),
            extra_data=extra_data,
        )


def coerce_detection(item: Detection | Mapping[str, Any], *, bbox_format: BBoxFormat = "xyxy") -> Detection:
    if isinstance(item, Detection):
        return item
    if isinstance(item, Mapping):
        return Detection.from_dict(item, bbox_format=bbox_format)
    raise TypeError("Detection or mapping expected")


def coerce_detections(
    items: Iterable[Detection | Mapping[str, Any]], *, bbox_format: BBoxFormat = "xyxy"
) -> list[Detection]:
    return [coerce_detection(item, bbox_format=bbox_format) for item in items]
