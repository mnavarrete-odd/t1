from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import math

from .types import DetectionData


def bbox_area(b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def expand_bbox(
    bbox: tuple[float, float, float, float],
    ratio: float,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float]:
    if ratio <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    cx, cy = bbox_center(bbox)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    scale = 1.0 + ratio
    new_w = bw * scale
    new_h = bh * scale
    nx1 = cx - new_w * 0.5
    ny1 = cy - new_h * 0.5
    nx2 = cx + new_w * 0.5
    ny2 = cy + new_h * 0.5
    nx1 = max(0.0, min(nx1, image_w - 1))
    ny1 = max(0.0, min(ny1, image_h - 1))
    nx2 = max(0.0, min(nx2, image_w - 1))
    ny2 = max(0.0, min(ny2, image_h - 1))
    return nx1, ny1, nx2, ny2


def coerce_detection(item: DetectionData | Mapping[str, object]) -> DetectionData:
    if isinstance(item, DetectionData):
        return item
    return DetectionData(
        class_id=int(item.get("class_id", -1)),
        class_name=str(item.get("class_name", "")),
        bbox=tuple(item.get("bbox", (0, 0, 0, 0))),
        confidence=float(item.get("confidence", 0.0)),
        extra_data=dict(item.get("extra_data", {}) or {}),
        tracking_id=item.get("tracking_id"),
    )


def boxes_in_area(
    dets: Sequence[DetectionData],
    area_bbox: tuple[float, float, float, float] | None,
    box_classes: Sequence[str],
) -> list[DetectionData]:
    if not area_bbox:
        return []
    ax1, ay1, ax2, ay2 = area_bbox
    boxes = []
    for d in dets:
        if d.class_name not in box_classes:
            continue
        cx, cy = bbox_center(d.bbox)
        if ax1 <= cx <= ax2 and ay1 <= cy <= ay2:
            boxes.append(d)
    return boxes


def coverage_ratio(
    area_bbox: tuple[float, float, float, float] | None,
    boxes_in_area_list: Sequence[DetectionData],
) -> float:
    if not area_bbox:
        return 0.0
    area_area = bbox_area(area_bbox)
    if area_area <= 0:
        return 0.0
    total = 0.0
    for d in boxes_in_area_list:
        total += bbox_intersection(area_bbox, d.bbox)
    return min(1.0, total / area_area)


def occlusion_ratio_by_body(
    area_bbox: tuple[float, float, float, float] | None,
    occluder_boxes: Sequence[DetectionData],
) -> float:
    if not area_bbox:
        return 0.0
    max_ratio = 0.0
    for d in occluder_boxes:
        occ_area = bbox_area(d.bbox)
        if occ_area <= 0:
            continue
        inter = bbox_intersection(area_bbox, d.bbox)
        ratio = inter / occ_area
        if ratio > max_ratio:
            max_ratio = ratio
    return min(1.0, max_ratio)


def person_near(
    area_bbox: tuple[float, float, float, float] | None,
    person_boxes: Sequence[DetectionData],
    dist_px: float,
) -> bool:
    if not area_bbox:
        return False
    ax, ay = bbox_center(area_bbox)
    for d in person_boxes:
        px, py = bbox_center(d.bbox)
        if math.hypot(px - ax, py - ay) <= dist_px:
            return True
    return False


def movement_score(
    curr: Sequence[DetectionData],
    prev: Sequence[DetectionData],
    w: int,
    h: int,
) -> float:
    if not curr or not prev:
        return 0.0
    diag = math.hypot(w, h) or 1.0
    used: set[int] = set()
    total = 0.0
    count = 0
    for c in curr:
        cx, cy = bbox_center(c.bbox)
        best_idx = None
        best_dist = None
        for i, p in enumerate(prev):
            if i in used:
                continue
            if p.class_name != c.class_name:
                continue
            px, py = bbox_center(p.bbox)
            dist = math.hypot(cx - px, cy - py)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist is not None:
            used.add(best_idx)
            total += best_dist / diag
            count += 1
    return total / count if count else 0.0
