from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class DepthTopdownConfig:
    depth_scale: float = 0.001
    max_depth_m: float = 3.0
    resolution_cm: float = 1.0
    histogram_bins: int = 300
    histogram_top_n: int = 10
    inset_ratio: float = 0.5
    center_crop_ratio: float = 0.5


def box_convert_cxcywh_to_xyxy(boxes_cxcywh: np.ndarray) -> np.ndarray:
    if boxes_cxcywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = boxes_cxcywh.astype(np.float32)
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    return np.stack([cx - (w * 0.5), cy - (h * 0.5), cx + (w * 0.5), cy + (h * 0.5)], axis=1)


def _rotation_or_identity(rotation_matrix: Optional[np.ndarray]) -> np.ndarray:
    if rotation_matrix is None:
        return np.eye(3, dtype=np.float32)
    rot = np.asarray(rotation_matrix, dtype=np.float32)
    if rot.shape != (3, 3):
        return np.eye(3, dtype=np.float32)
    return rot


def _to_depth_meters(depth_img: np.ndarray, depth_scale: float) -> np.ndarray:
    return depth_img.astype(np.float32) * float(depth_scale)


def _mode_histogram(values: np.ndarray, bins: int, top_n: int) -> float:
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return float("nan")

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    if vmax <= vmin:
        return vmin

    hist, edges = np.histogram(valid, bins=max(1, int(bins)), range=(vmin, vmax))
    nonzero = np.where(hist > 0)[0]
    if nonzero.size == 0:
        return float("nan")

    candidates = nonzero[: max(1, int(top_n))]
    best_bin = int(candidates[np.argmax(hist[candidates])])
    return float((edges[best_bin] + edges[best_bin + 1]) * 0.5)


def estimate_bbox_depths(
    depth_image_m: np.ndarray,
    bboxes_xyxy: np.ndarray,
    *,
    histogram_bins: int = 300,
    histogram_top_n: int = 10,
    center_crop_ratio: float = 0.5,
    rotation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = depth_image_m.shape[:2]
    ratio = float(np.clip(center_crop_ratio, 0.1, 1.0))
    out = np.full((len(bboxes_xyxy),), np.nan, dtype=np.float32)

    for idx, bb in enumerate(bboxes_xyxy):
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        bw = x2 - x1
        bh = y2 - y1
        dx = int(round(bw * (1.0 - ratio) * 0.5))
        dy = int(round(bh * (1.0 - ratio) * 0.5))

        cx1 = x1 + dx
        cy1 = y1 + dy
        cx2 = x2 - dx
        cy2 = y2 - dy
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2

        crop = depth_image_m[cy1:cy2, cx1:cx2]
        out[idx] = _mode_histogram(crop.reshape(-1), histogram_bins, histogram_top_n)

    rot = _rotation_or_identity(rotation_matrix)
    out = out * float(rot[2, 2])
    return out


def create_topdown_image(
    depth_image_m: np.ndarray,
    intrinsics: Dict[str, float],
    rotation_matrix: Optional[np.ndarray],
    *,
    max_depth_m: float = 3.0,
    resolution_cm: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    valid = np.isfinite(depth_image_m) & (depth_image_m > 0) & (depth_image_m < float(max_depth_m))
    v, u = np.nonzero(valid)
    if v.size == 0:
        return np.zeros((0, 0), dtype=np.float32), 0.0, 0.0

    z = depth_image_m[v, u]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])

    x = (u.astype(np.float32) - ppx) * z / max(fx, 1e-6)
    y = (v.astype(np.float32) - ppy) * z / max(fy, 1e-6)
    pts = np.stack([x, y, z], axis=1)

    rot = _rotation_or_identity(rotation_matrix)
    pts_rot = pts @ rot.T
    keep = pts_rot[:, 2] < float(max_depth_m)
    if not np.any(keep):
        return np.zeros((0, 0), dtype=np.float32), 0.0, 0.0

    pts_rot = pts_rot[keep]
    x_cm = pts_rot[:, 0] * 100.0
    y_cm = pts_rot[:, 1] * 100.0
    z_rot = pts_rot[:, 2]

    x_min = float(np.min(x_cm))
    x_max = float(np.max(x_cm))
    y_min = float(np.min(y_cm))
    y_max = float(np.max(y_cm))

    res = max(float(resolution_cm), 0.1)
    img_w = int(np.ceil((x_max - x_min) / res)) + 1
    img_h = int(np.ceil((y_max - y_min) / res)) + 1
    if img_w <= 0 or img_h <= 0:
        return np.zeros((0, 0), dtype=np.float32), 0.0, 0.0

    px = ((x_cm - x_min) / res).astype(np.int32)
    py = ((y_cm - y_min) / res).astype(np.int32)
    px = np.clip(px, 0, img_w - 1)
    py = np.clip(py, 0, img_h - 1)

    topdown = np.full((img_h, img_w), np.nan, dtype=np.float32)
    sort_idx = np.argsort(-z_rot)  # far->near so near points overwrite
    for idx in sort_idx:
        topdown[py[idx], px[idx]] = float(z_rot[idx])

    origin_x = -x_min / res
    origin_y = -y_min / res
    return topdown, float(origin_x), float(origin_y)


def project_bboxes_to_world_coords(
    bboxes_xyxy: np.ndarray,
    depths_m: np.ndarray,
    intrinsics: Dict[str, float],
    rotation_matrix: Optional[np.ndarray],
    *,
    resolution_cm: float = 1.0,
) -> np.ndarray:
    rot = _rotation_or_identity(rotation_matrix)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    res = max(float(resolution_cm), 0.1)

    coords = np.zeros((len(bboxes_xyxy), 4, 2), dtype=np.float32)
    for i, bb in enumerate(bboxes_xyxy):
        d = float(depths_m[i])
        if not np.isfinite(d) or d <= 0.0:
            continue

        x1, y1, x2, y2 = [float(v) for v in bb]
        corners = np.array(
            [[x1, y2], [x2, y2], [x2, y1], [x1, y1]],
            dtype=np.float32,
        )

        x = (corners[:, 0] - ppx) * d / max(fx, 1e-6)
        y = (corners[:, 1] - ppy) * d / max(fy, 1e-6)
        z = np.full_like(x, d)
        pts = np.stack([x, y, z], axis=1)
        pts_rot = pts @ rot.T

        coords[i, :, 0] = (pts_rot[:, 0] * 100.0) / res
        coords[i, :, 1] = (pts_rot[:, 1] * 100.0) / res

    return coords


def extract_inset_depths_with_offset(
    topdown_img: np.ndarray,
    bbox_coords_world: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    inset_ratio: float = 0.5,
) -> np.ndarray:
    out = np.full((len(bbox_coords_world),), np.nan, dtype=np.float32)
    if topdown_img.size == 0:
        return out

    img_h, img_w = topdown_img.shape
    margin = (1.0 - float(np.clip(inset_ratio, 0.1, 1.0))) * 0.5

    for i, corners_world in enumerate(bbox_coords_world):
        corners = corners_world.copy()
        corners[:, 0] += float(origin_x)
        corners[:, 1] += float(origin_y)

        x_min = float(np.min(corners[:, 0]))
        x_max = float(np.max(corners[:, 0]))
        y_min = float(np.min(corners[:, 1]))
        y_max = float(np.max(corners[:, 1]))

        bw = x_max - x_min
        bh = y_max - y_min
        ix1 = int(np.floor(x_min + (bw * margin)))
        ix2 = int(np.ceil(x_max - (bw * margin)))
        iy1 = int(np.floor(y_min + (bh * margin)))
        iy2 = int(np.ceil(y_max - (bh * margin)))

        ix1 = max(0, min(img_w - 1, ix1))
        iy1 = max(0, min(img_h - 1, iy1))
        ix2 = max(0, min(img_w, ix2))
        iy2 = max(0, min(img_h, iy2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        crop = topdown_img[iy1:iy2, ix1:ix2]
        valid = crop[np.isfinite(crop)]
        if valid.size == 0:
            continue
        out[i] = float(np.min(valid))

    return out


def compute_bboxes_depth(
    depth_image: np.ndarray,
    bboxes_xyxy: np.ndarray,
    *,
    intrinsics: Dict[str, float],
    rotation_matrix: Optional[np.ndarray] = None,
    cfg: DepthTopdownConfig = DepthTopdownConfig(),
    rough_estimate: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if bboxes_xyxy.size == 0:
        return np.zeros((0,), dtype=np.float32), None

    depth_m = _to_depth_meters(depth_image, cfg.depth_scale)
    rough = estimate_bbox_depths(
        depth_m,
        bboxes_xyxy,
        histogram_bins=cfg.histogram_bins,
        histogram_top_n=cfg.histogram_top_n,
        center_crop_ratio=cfg.center_crop_ratio,
        rotation_matrix=rotation_matrix,
    )
    if rough_estimate:
        return rough, None

    topdown, ox, oy = create_topdown_image(
        depth_m,
        intrinsics=intrinsics,
        rotation_matrix=rotation_matrix,
        max_depth_m=cfg.max_depth_m,
        resolution_cm=cfg.resolution_cm,
    )
    coords_world = project_bboxes_to_world_coords(
        bboxes_xyxy,
        rough,
        intrinsics=intrinsics,
        rotation_matrix=rotation_matrix,
        resolution_cm=cfg.resolution_cm,
    )
    min_depth = extract_inset_depths_with_offset(
        topdown,
        coords_world,
        origin_x=ox,
        origin_y=oy,
        inset_ratio=cfg.inset_ratio,
    )

    nan_mask = ~np.isfinite(min_depth)
    min_depth[nan_mask] = rough[nan_mask]
    return min_depth, coords_world


def compute_bboxes_floor_depth(
    floor_depth_image: np.ndarray,
    bbox_coords_world: np.ndarray,
    *,
    intrinsics: Dict[str, float],
    rotation_matrix: Optional[np.ndarray] = None,
    cfg: DepthTopdownConfig = DepthTopdownConfig(),
) -> np.ndarray:
    if bbox_coords_world is None or bbox_coords_world.size == 0:
        return np.zeros((0,), dtype=np.float32)

    depth_m = _to_depth_meters(floor_depth_image, cfg.depth_scale)
    topdown, ox, oy = create_topdown_image(
        depth_m,
        intrinsics=intrinsics,
        rotation_matrix=rotation_matrix,
        max_depth_m=cfg.max_depth_m,
        resolution_cm=cfg.resolution_cm,
    )
    return extract_inset_depths_with_offset(
        topdown,
        bbox_coords_world,
        origin_x=ox,
        origin_y=oy,
        inset_ratio=cfg.inset_ratio,
    )


def get_cardboard_depth(
    depth_image: np.ndarray,
    cardboard_bbox_cxcywh: np.ndarray,
    *,
    cfg: DepthTopdownConfig = DepthTopdownConfig(),
    rotation_matrix: Optional[np.ndarray] = None,
) -> float:
    depth_m = _to_depth_meters(depth_image, cfg.depth_scale)
    x, y, w, h = [int(round(v)) for v in cardboard_bbox_cxcywh]
    x1 = max(0, x - (w // 4))
    x2 = min(depth_m.shape[1], x + (w // 4))
    y1 = max(0, y - (h // 4))
    y2 = min(depth_m.shape[0], y + (h // 4))
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    vals = depth_m[y1:y2, x1:x2].reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 0) & (vals < cfg.max_depth_m)]
    if vals.size == 0:
        return float("nan")

    out = float(np.mean(vals))
    rot = _rotation_or_identity(rotation_matrix)
    return out * float(rot[2, 2])
