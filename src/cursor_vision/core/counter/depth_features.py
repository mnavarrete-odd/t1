from typing import Tuple

import numpy as np


def depth_at_bbox_center(
    depth_map: np.ndarray,
    bbox_cxcywh: np.ndarray,
    patch_radius: int = 2,
    depth_scale: float = 1.0,
) -> float:
    if depth_map is None:
        return float("nan")

    h, w = depth_map.shape[:2]
    cx, cy, _, _ = bbox_cxcywh
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    x1 = max(0, cx_i - patch_radius)
    x2 = min(w, cx_i + patch_radius + 1)
    y1 = max(0, cy_i - patch_radius)
    y2 = min(h, cy_i + patch_radius + 1)

    if x2 <= x1 or y2 <= y1:
        return float("nan")

    patch = depth_map[y1:y2, x1:x2].astype(np.float32)
    if patch.size == 0:
        return float("nan")

    valid = patch[np.isfinite(patch)]
    valid = valid[valid > 0]
    if valid.size == 0:
        return float("nan")

    return float(np.median(valid) * depth_scale)


def bbox_cxcywh_to_xyxy_clamped(
    bbox_cxcywh: np.ndarray,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    cx, cy, bw, bh = bbox_cxcywh
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return x1, y1, x2, y2

