from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from .containment import ContainmentConfig, filter_contained_detections
from .types import DetectionRaw
from .workarea_mask import (
    WorkAreaMaskConfig,
    compute_working_area_mask,
    detection_in_work_area,
)


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _variants(name: str) -> Set[str]:
    n = normalize_class_name(name)
    out = {n}
    if n.endswith("s"):
        out.add(n[:-1])
    else:
        out.add(f"{n}s")
    return out


def filter_detections_by_class(
    detections: List[DetectionRaw],
    allowed_class_names: Iterable[str],
) -> List[DetectionRaw]:
    allowed = set()
    for cls in allowed_class_names:
        allowed.update(_variants(cls))

    filtered = []
    for det in detections:
        cls = normalize_class_name(det.class_name)
        if cls in allowed:
            filtered.append(det)
    return filtered


def filter_detections_by_work_area(
    detections: List[DetectionRaw],
    area_bbox_cxcywh,
) -> List[DetectionRaw]:
    if area_bbox_cxcywh is None:
        return list(detections)

    try:
        area_cx = float(area_bbox_cxcywh[0])
        area_cy = float(area_bbox_cxcywh[1])
        area_w = float(area_bbox_cxcywh[2])
        area_h = float(area_bbox_cxcywh[3])
    except Exception:
        return list(detections)

    if area_w <= 0.0 or area_h <= 0.0:
        return list(detections)

    ax1 = area_cx - (area_w * 0.5)
    ay1 = area_cy - (area_h * 0.5)
    ax2 = area_cx + (area_w * 0.5)
    ay2 = area_cy + (area_h * 0.5)

    filtered: List[DetectionRaw] = []
    for det in detections:
        dcx = float(det.bbox_cxcywh[0])
        dcy = float(det.bbox_cxcywh[1])
        if ax1 <= dcx <= ax2 and ay1 <= dcy <= ay2:
            filtered.append(det)
    return filtered


def filter_detections_by_depth_work_area(
    detections: List[DetectionRaw],
    area_bbox_cxcywh: Optional[np.ndarray],
    depth_map: Optional[np.ndarray],
    *,
    workarea_cfg: WorkAreaMaskConfig,
    containment_cfg: Optional[ContainmentConfig] = None,
    rotation_matrix: Optional[np.ndarray] = None,
) -> Tuple[List[DetectionRaw], Optional[np.ndarray]]:
    if len(detections) == 0:
        return [], None

    mask = compute_working_area_mask(
        depth_map,
        cfg=workarea_cfg,
        rotation_matrix=rotation_matrix,
    )

    # Preserve baseline behavior: if we cannot infer any work area cue, keep detections.
    if mask is None and area_bbox_cxcywh is None:
        filtered = list(detections)
    else:
        filtered = []
        for det in detections:
            if detection_in_work_area(
                det.bbox_cxcywh,
                area_bbox_cxcywh,
                mask,
                cfg=workarea_cfg,
                depth_img=depth_map,
                is_area_detection=False,
            ):
                filtered.append(det)

    if containment_cfg is not None:
        filtered = filter_contained_detections(filtered, containment_cfg)

    return filtered, mask
