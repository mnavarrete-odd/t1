from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class WorkAreaMaskConfig:
    enabled: bool = True
    max_depth_mm: float = 2800.0
    bbox_mask_threshold: float = 0.40
    area_bbox_mask_threshold: float = 0.20
    depth_scale_to_mm: float = 1.0
    downsample_scale: int = 4
    grid_resolution_mm: float = 50.0
    center_tolerance_px: int = 2
    fx: float = 1125.636
    fy: float = 1125.133
    cx: float = 952.45
    cy: float = 550.62


def _bbox_center_inside_area(
    bbox_cxcywh: np.ndarray,
    area_bbox_cxcywh: Optional[np.ndarray],
) -> bool:
    if area_bbox_cxcywh is None:
        return False
    cx, cy = float(bbox_cxcywh[0]), float(bbox_cxcywh[1])
    acx, acy, aw, ah = [float(v) for v in area_bbox_cxcywh]
    if aw <= 0.0 or ah <= 0.0:
        return False
    ax1 = acx - (aw * 0.5)
    ay1 = acy - (ah * 0.5)
    ax2 = acx + (aw * 0.5)
    ay2 = acy + (ah * 0.5)
    return ax1 <= cx <= ax2 and ay1 <= cy <= ay2


def bbox_ratio_in_mask(
    bbox_cxcywh: np.ndarray,
    mask: np.ndarray,
    depth_img: Optional[np.ndarray] = None,
) -> float:
    x1 = int(max(0, bbox_cxcywh[0] - bbox_cxcywh[2] * 0.5))
    x2 = int(min(mask.shape[1], bbox_cxcywh[0] + bbox_cxcywh[2] * 0.5))
    y1 = int(max(0, bbox_cxcywh[1] - bbox_cxcywh[3] * 0.5))
    y2 = int(min(mask.shape[0], bbox_cxcywh[1] + bbox_cxcywh[3] * 0.5))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    mask_crop = mask[y1:y2, x1:x2]
    positives = float(np.count_nonzero(mask_crop))

    if depth_img is None:
        area = float((y2 - y1) * (x2 - x1))
    else:
        depth_crop = depth_img[y1:y2, x1:x2]
        area = float(np.count_nonzero(np.isfinite(depth_crop) & (depth_crop > 0)))

    if area <= 0.0:
        return 0.0
    return positives / area


def detection_in_work_area(
    bbox_cxcywh: np.ndarray,
    area_bbox_cxcywh: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    cfg: WorkAreaMaskConfig,
    depth_img: Optional[np.ndarray] = None,
    *,
    is_area_detection: bool = False,
) -> bool:
    center_inside = _bbox_center_inside_area(bbox_cxcywh, area_bbox_cxcywh)
    if mask is None:
        return center_inside

    ratio = bbox_ratio_in_mask(bbox_cxcywh, mask, depth_img=depth_img)
    threshold = cfg.area_bbox_mask_threshold if is_area_detection else cfg.bbox_mask_threshold
    return center_inside or (ratio >= threshold)


def compute_working_area_mask(
    depth_img: Optional[np.ndarray],
    cfg: WorkAreaMaskConfig,
    rotation_matrix: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if depth_img is None or not cfg.enabled:
        return None

    if depth_img.ndim != 2:
        return None

    h_orig, w_orig = depth_img.shape
    if h_orig <= 0 or w_orig <= 0:
        return None

    depth_mm = depth_img.astype(np.float32) * float(cfg.depth_scale_to_mm)

    scale = max(1, int(cfg.downsample_scale))
    depth_small = cv2.resize(
        depth_mm,
        None,
        fx=1.0 / scale,
        fy=1.0 / scale,
        interpolation=cv2.INTER_NEAREST,
    )

    valid_mask = np.isfinite(depth_small) & (depth_small > 0) & (depth_small < cfg.max_depth_mm)
    v_coords, u_coords = np.nonzero(valid_mask)
    if v_coords.size == 0:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    z = depth_small[v_coords, u_coords]
    fx = float(cfg.fx) / scale
    fy = float(cfg.fy) / scale
    cx = float(cfg.cx) / scale
    cy = float(cfg.cy) / scale
    x = (u_coords.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y = (v_coords.astype(np.float32) - cy) * z / max(fy, 1e-6)

    if rotation_matrix is None:
        rot = np.eye(3, dtype=np.float32)
    else:
        rot = np.asarray(rotation_matrix, dtype=np.float32)
        if rot.shape != (3, 3):
            rot = np.eye(3, dtype=np.float32)

    points = np.stack([x, y, z], axis=1)
    points_rot = points @ rot.T

    valid_rot = points_rot[:, 2] <= float(cfg.max_depth_mm)
    if not np.any(valid_rot):
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    points_rot = points_rot[valid_rot]
    u_coords = u_coords[valid_rot]
    v_coords = v_coords[valid_rot]

    xy = points_rot[:, :2]
    xy_min = np.min(xy, axis=0)
    xy_max = np.max(xy, axis=0)
    res = max(float(cfg.grid_resolution_mm), 1.0)

    grid_coords = ((xy - xy_min) / res).astype(np.int32)
    grid_w = int(np.max(grid_coords[:, 0])) + 1
    grid_h = int(np.max(grid_coords[:, 1])) + 1
    if grid_w <= 0 or grid_h <= 0:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    grid_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
    grid_mask[grid_coords[:, 1], grid_coords[:, 0]] = 255

    num_labels, labels = cv2.connectedComponents(grid_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    ref_u = int((w_orig * 0.5) / scale)
    tol = max(0, int(cfg.center_tolerance_px))
    center_hits = np.abs(u_coords - ref_u) <= tol
    if not np.any(center_hits):
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    point_labels = labels[grid_coords[:, 1], grid_coords[:, 0]]
    center_labels = point_labels[center_hits]
    center_labels = center_labels[center_labels > 0]
    if center_labels.size == 0:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    selected_label = None
    selected_count = -1
    for lbl in np.unique(center_labels):
        cnt = int(np.count_nonzero(point_labels == lbl))
        if cnt > selected_count:
            selected_count = cnt
            selected_label = int(lbl)

    if selected_label is None:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    keep = point_labels == selected_label
    result_small = np.zeros_like(depth_small, dtype=np.uint8)
    result_small[v_coords[keep], u_coords[keep]] = 1
    result = cv2.resize(
        result_small,
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST,
    )
    return (result > 0).astype(np.uint8)
