from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .depth_features import bbox_cxcywh_to_xyxy_clamped, depth_at_bbox_center
from .depth_topdown import (
    DepthTopdownConfig,
    compute_bboxes_depth,
    compute_bboxes_floor_depth,
)
from .types import DetectionRaw, FrameData, PreparedDetection


class FeatureExtractor:
    def __init__(
        self,
        embedder,
        depth_patch_radius: int = 2,
        depth_scale: float = 1.0,
        *,
        intrinsics: Optional[Dict[str, float]] = None,
        rotation_matrix: Optional[np.ndarray] = None,
        depth_topdown_cfg: Optional[DepthTopdownConfig] = None,
        use_depth_topdown: bool = True,
    ):
        self.embedder = embedder
        self.depth_patch_radius = depth_patch_radius
        self.depth_scale = depth_scale
        self.intrinsics = intrinsics
        self.rotation_matrix = rotation_matrix
        self.depth_topdown_cfg = depth_topdown_cfg or DepthTopdownConfig(depth_scale=depth_scale)
        self.use_depth_topdown = use_depth_topdown

    def prepare(
        self,
        frame: FrameData,
        detections: List[DetectionRaw],
        *,
        prev_depth_map: Optional[np.ndarray] = None,
    ) -> List[PreparedDetection]:
        if len(detections) == 0:
            return []

        h, w = frame.image.shape[:2]

        crops = []
        xyxys = []
        center_depths = []
        aspects = []
        centroids = []
        bbox_heights_px = []

        for det in detections:
            bbox = det.bbox_cxcywh
            x1, y1, x2, y2 = bbox_cxcywh_to_xyxy_clamped(bbox, w, h)
            xyxys.append((x1, y1, x2, y2))

            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((8, 8, 3), dtype=np.uint8))
            else:
                crops.append(frame.image[y1:y2, x1:x2])

            depth_center = depth_at_bbox_center(
                frame.depth_map,
                bbox,
                patch_radius=self.depth_patch_radius,
                depth_scale=self.depth_scale,
            )
            center_depths.append(depth_center)

            bw = max(float(bbox[2]), 1e-6)
            bh = max(float(bbox[3]), 1e-6)
            aspects.append(max(bw, bh) / min(bw, bh))
            centroids.append((float(bbox[0]), float(bbox[1])))
            bbox_heights_px.append(float(bh))

        emb = self.embedder.embed(crops)

        n = len(detections)
        product_depths = np.asarray(center_depths, dtype=np.float64)
        floor_depths = np.full((n,), np.nan, dtype=np.float64)

        use_topdown = (
            self.use_depth_topdown
            and frame.depth_map is not None
            and self.intrinsics is not None
        )
        if use_topdown:
            try:
                bboxes_xyxy = np.asarray(xyxys, dtype=np.float32)
                product_depths_td, bbox_coords_world = compute_bboxes_depth(
                    frame.depth_map,
                    bboxes_xyxy,
                    intrinsics=self.intrinsics,
                    rotation_matrix=self.rotation_matrix,
                    cfg=self.depth_topdown_cfg,
                    rough_estimate=False,
                )
                if product_depths_td.shape[0] == n:
                    product_depths = product_depths_td.astype(np.float64)

                if prev_depth_map is not None and bbox_coords_world is not None:
                    floor_depths_td = compute_bboxes_floor_depth(
                        prev_depth_map,
                        bbox_coords_world,
                        intrinsics=self.intrinsics,
                        rotation_matrix=self.rotation_matrix,
                        cfg=self.depth_topdown_cfg,
                    )
                    if floor_depths_td.shape[0] == n:
                        floor_depths = floor_depths_td.astype(np.float64)
            except Exception:
                # Fallback stays as center depths; floor depths remain NaN.
                pass

        prepared: List[PreparedDetection] = []
        for i, det in enumerate(detections):
            d_prod = float(product_depths[i]) if i < product_depths.shape[0] else float("nan")
            d_center = float(center_depths[i]) if i < len(center_depths) else float("nan")
            if not np.isfinite(d_prod):
                d_prod = d_center

            d_floor = float(floor_depths[i]) if i < floor_depths.shape[0] else float("nan")
            d_delta = (d_floor - d_prod) if (np.isfinite(d_floor) and np.isfinite(d_prod)) else float("nan")

            bh = bbox_heights_px[i]
            if np.isfinite(d_prod) and d_prod > 0.0:
                h_depth = float(bh * d_prod)
            else:
                h_depth = float(bh)

            prepared.append(
                PreparedDetection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_cxcywh=det.bbox_cxcywh,
                    bbox_xyxy=xyxys[i],
                    centroid=centroids[i],
                    aspect_ratio=float(aspects[i]),
                    height_depth=float(h_depth),
                    depth_center=float(d_prod),
                    embedding=emb[i],
                    product_depth=float(d_prod),
                    floor_depth=float(d_floor),
                    depth_delta=float(d_delta),
                )
            )

        return prepared
