from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .types import DetectionRaw


@dataclass
class ContainmentConfig:
    enabled: bool = True
    ioa_threshold: float = 0.70
    # contained_class_id -> (container_class_id, factor)
    rules: Dict[int, Tuple[int, float]] | None = None


def calculate_ioa(inner_bbox_cxcywh: np.ndarray, outer_bbox_cxcywh: np.ndarray) -> float:
    x1_min = float(inner_bbox_cxcywh[0] - inner_bbox_cxcywh[2] * 0.5)
    y1_min = float(inner_bbox_cxcywh[1] - inner_bbox_cxcywh[3] * 0.5)
    x1_max = float(inner_bbox_cxcywh[0] + inner_bbox_cxcywh[2] * 0.5)
    y1_max = float(inner_bbox_cxcywh[1] + inner_bbox_cxcywh[3] * 0.5)

    x2_min = float(outer_bbox_cxcywh[0] - outer_bbox_cxcywh[2] * 0.5)
    y2_min = float(outer_bbox_cxcywh[1] - outer_bbox_cxcywh[3] * 0.5)
    x2_max = float(outer_bbox_cxcywh[0] + outer_bbox_cxcywh[2] * 0.5)
    y2_max = float(outer_bbox_cxcywh[1] + outer_bbox_cxcywh[3] * 0.5)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    inner_area = max(float(inner_bbox_cxcywh[2] * inner_bbox_cxcywh[3]), 1e-8)
    return float(inter_area / inner_area)


def filter_contained_detections(
    detections: List[DetectionRaw],
    cfg: ContainmentConfig,
) -> List[DetectionRaw]:
    if not cfg.enabled or len(detections) <= 1:
        return list(detections)

    rules = cfg.rules or {}
    keep = [True] * len(detections)

    for i, det_i in enumerate(detections):
        if not keep[i]:
            continue

        for j, det_j in enumerate(detections):
            if i == j or not keep[j]:
                continue

            ioa = calculate_ioa(det_i.bbox_cxcywh, det_j.bbox_cxcywh)
            if ioa <= cfg.ioa_threshold:
                continue

            contained_id = int(det_i.class_id)
            container_id = int(det_j.class_id)
            is_special = contained_id in rules and int(rules[contained_id][0]) == container_id

            # Mimics Andina: special containment and general containment both drop inner box.
            if is_special or ioa > cfg.ioa_threshold:
                keep[i] = False
                break

    return [det for idx, det in enumerate(detections) if keep[idx]]
