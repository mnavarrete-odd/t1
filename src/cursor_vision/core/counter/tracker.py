from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .costs import BayesianCostModel, CostConfig, build_cost_matrix
from .types import (
    DetectionAssignment,
    FrameResult,
    LostTrackRecord,
    PreparedDetection,
    TrackState,
)

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


@dataclass
class TrackerConfig:
    max_misses: int = 3
    match_threshold: float = 1.5
    reset_on_no_products: bool = False
    second_pass_greedy: bool = True
    greedy_match_threshold: float = -1.0
    contained_by_rules: Dict[int, Tuple[int, float]] = field(default_factory=dict)
    contained_ioa_threshold: float = 0.1
    contained_depth_delta_threshold: float = 0.02
    # Third-pass recovery for depth/lighting glitches (prevents ID switches).
    spatial_relink_enabled: bool = True
    relink_require_same_class: bool = True
    relink_min_iou: float = 0.55
    relink_max_center_norm: float = 0.04
    relink_max_aspect_rel_diff: float = 0.12
    relink_max_cosine_distance: float = 0.35


class CounterTracker:
    def __init__(self, tracker_cfg: TrackerConfig, cost_cfg: CostConfig):
        self.tracker_cfg = tracker_cfg
        self.cost_cfg = cost_cfg
        self.cost_model = BayesianCostModel(cost_cfg)
        self.tracks: List[TrackState] = []
        self.next_id = 0

    def reset(self):
        self.tracks = []
        self.next_id = 0

    def _new_track(self, det: PreparedDetection, frame_index: int) -> TrackState:
        t = TrackState(
            track_id=self.next_id,
            class_id=det.class_id,
            class_name=det.class_name,
            bbox_cxcywh=det.bbox_cxcywh.copy(),
            centroid=det.centroid,
            aspect_ratio=det.aspect_ratio,
            height_depth=det.height_depth,
            depth_center=det.depth_center,
            embedding=det.embedding.copy(),
            product_depth=det.product_depth,
            floor_depth=det.floor_depth,
            depth_delta=det.depth_delta,
            last_frame_index=frame_index,
        )
        self.next_id += 1
        return t

    @staticmethod
    def _calculate_ioa(inner_bbox_cxcywh: np.ndarray, outer_bbox_cxcywh: np.ndarray) -> float:
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

    @staticmethod
    def _calculate_iou(box_a_cxcywh: np.ndarray, box_b_cxcywh: np.ndarray) -> float:
        ax1 = float(box_a_cxcywh[0] - box_a_cxcywh[2] * 0.5)
        ay1 = float(box_a_cxcywh[1] - box_a_cxcywh[3] * 0.5)
        ax2 = float(box_a_cxcywh[0] + box_a_cxcywh[2] * 0.5)
        ay2 = float(box_a_cxcywh[1] + box_a_cxcywh[3] * 0.5)

        bx1 = float(box_b_cxcywh[0] - box_b_cxcywh[2] * 0.5)
        by1 = float(box_b_cxcywh[1] - box_b_cxcywh[3] * 0.5)
        bx2 = float(box_b_cxcywh[0] + box_b_cxcywh[2] * 0.5)
        by2 = float(box_b_cxcywh[1] + box_b_cxcywh[3] * 0.5)

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-8)
        area_b = max((bx2 - bx1) * (by2 - by1), 1e-8)
        union = max(area_a + area_b - inter, 1e-8)
        return float(inter / union)

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return float("nan")
        if a.size == 0 or b.size == 0:
            return float("nan")
        if a.shape != b.shape:
            return float("nan")

        num = float(np.dot(a, b))
        den = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-8
        sim = np.clip(num / den, -1.0, 1.0)
        return 1.0 - sim

    def _should_suppress_contained_new(
        self,
        det: PreparedDetection,
        unmatched_track_indices: list[int],
    ) -> bool:
        contained_by = self.tracker_cfg.contained_by_rules or {}
        cls = int(det.class_id)
        if cls not in contained_by:
            return False

        container_cls = int(contained_by[cls][0])
        for track_idx in unmatched_track_indices:
            tr = self.tracks[track_idx]
            if int(tr.class_id) != container_cls:
                continue

            ioa = self._calculate_ioa(det.bbox_cxcywh, tr.bbox_cxcywh)
            if ioa <= float(self.tracker_cfg.contained_ioa_threshold):
                continue

            if np.isfinite(det.depth_delta):
                if abs(float(det.depth_delta)) < float(self.tracker_cfg.contained_depth_delta_threshold):
                    return True
        return False

    @staticmethod
    def _update_track(track: TrackState, det: PreparedDetection, frame_index: int) -> None:
        track.bbox_cxcywh = det.bbox_cxcywh.copy()
        track.centroid = det.centroid
        track.aspect_ratio = det.aspect_ratio
        track.height_depth = det.height_depth
        track.depth_center = det.depth_center
        track.embedding = det.embedding.copy()
        track.product_depth = det.product_depth
        track.floor_depth = det.floor_depth
        track.depth_delta = det.depth_delta
        track.last_frame_index = frame_index
        track.hits += 1
        track.age += 1
        track.misses = 0

    def _age_tracks(self):
        for tr in self.tracks:
            tr.misses += 1
            tr.age += 1

    @staticmethod
    def _to_lost_record(track: TrackState) -> LostTrackRecord:
        return LostTrackRecord(
            track_id=int(track.track_id),
            class_id=int(track.class_id),
            class_name=str(track.class_name),
            product_depth=float(track.product_depth),
            floor_depth=float(track.floor_depth),
            depth_delta=float(track.depth_delta),
        )

    def _prune_dead_tracks(self) -> list[TrackState]:
        alive_tracks: list[TrackState] = []
        lost_tracks: list[TrackState] = []
        for track in self.tracks:
            if track.misses <= self.tracker_cfg.max_misses:
                alive_tracks.append(track)
            else:
                lost_tracks.append(track)
        self.tracks = alive_tracks
        return lost_tracks

    def step(
        self,
        frame_index: int,
        detections: List[PreparedDetection],
        image_shape: tuple[int, int],
    ) -> FrameResult:
        result = FrameResult(frame_index=frame_index, num_detections=len(detections))

        if len(detections) == 0:
            if self.tracker_cfg.reset_on_no_products:
                lost_states = list(self.tracks)
                self.reset()
            else:
                self._age_tracks()
                lost_states = self._prune_dead_tracks()
            result.lost_tracks = [self._to_lost_record(t) for t in lost_states]
            result.num_lost_tracks = len(result.lost_tracks)
            result.num_active_tracks = len(self.tracks)
            return result

        if len(self.tracks) == 0:
            for di, det in enumerate(detections):
                new_t = self._new_track(det, frame_index=frame_index)
                self.tracks.append(new_t)
                result.assignments.append(
                    DetectionAssignment(
                        det_idx=di,
                        track_id=new_t.track_id,
                        status="new",
                        cost=-1.0,
                        reason="new_bootstrap",
                        match_probability=1.0,
                        depth_delta=float(det.depth_delta),
                    )
                )
            result.num_new = len(detections)
            result.num_active_tracks = len(self.tracks)
            return result

        h, w = image_shape
        image_diag = float(np.hypot(w, h))

        cost_matrix, pair_features, prob_matrix = build_cost_matrix(
            self.tracks,
            detections,
            image_diag,
            self.cost_cfg,
            model=self.cost_model,
        )
        if cost_matrix.size == 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
            prob_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)

        large_cost = float(self.cost_cfg.large_cost)
        assign_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, large_cost)

        n_tracks, n_dets = assign_matrix.shape
        matched_track_indices = set()
        matched_det_indices = set()
        det_to_assignment: Dict[int, DetectionAssignment] = {}

        if linear_sum_assignment is not None and assign_matrix.size > 0:
            track_idx, det_idx = linear_sum_assignment(assign_matrix)
            for ti, di in zip(track_idx, det_idx):
                c = float(cost_matrix[ti, di])
                if not np.isfinite(c):
                    continue
                if c <= self.tracker_cfg.match_threshold:
                    det = detections[di]
                    self._update_track(self.tracks[ti], det, frame_index)
                    matched_track_indices.add(int(ti))
                    matched_det_indices.add(int(di))

                    feat = pair_features.get((int(ti), int(di)), {})
                    det_to_assignment[int(di)] = DetectionAssignment(
                        det_idx=int(di),
                        track_id=self.tracks[ti].track_id,
                        status="matched",
                        cost=c,
                        reason=str(feat.get("reason", "matched")),
                        center_norm=float(feat.get("center_norm", float("nan"))),
                        aspect_rel_diff=float(feat.get("aspect_rel_diff", float("nan"))),
                        height_depth_rel_diff=float(feat.get("height_depth_rel_diff", float("nan"))),
                        cosine_distance=float(feat.get("cosine_distance", float("nan"))),
                        depth_delta=float(feat.get("depth_delta", float("nan"))),
                        match_probability=float(
                            feat.get("match_probability", prob_matrix[ti, di] if prob_matrix.size else float("nan"))
                        ),
                    )

        # Fallback or second pass if scipy unavailable
        if linear_sum_assignment is None:
            candidates = []
            for ti in range(n_tracks):
                for di in range(n_dets):
                    c = float(cost_matrix[ti, di])
                    if np.isfinite(c) and c <= self.tracker_cfg.match_threshold:
                        candidates.append((c, ti, di))
            candidates.sort(key=lambda x: x[0])
            for c, ti, di in candidates:
                if ti in matched_track_indices or di in matched_det_indices:
                    continue
                det = detections[di]
                self._update_track(self.tracks[ti], det, frame_index)
                matched_track_indices.add(int(ti))
                matched_det_indices.add(int(di))

                feat = pair_features.get((int(ti), int(di)), {})
                det_to_assignment[int(di)] = DetectionAssignment(
                    det_idx=int(di),
                    track_id=self.tracks[ti].track_id,
                    status="matched",
                    cost=float(c),
                    reason=str(feat.get("reason", "matched")),
                    center_norm=float(feat.get("center_norm", float("nan"))),
                    aspect_rel_diff=float(feat.get("aspect_rel_diff", float("nan"))),
                    height_depth_rel_diff=float(feat.get("height_depth_rel_diff", float("nan"))),
                    cosine_distance=float(feat.get("cosine_distance", float("nan"))),
                    depth_delta=float(feat.get("depth_delta", float("nan"))),
                    match_probability=float(
                        feat.get("match_probability", prob_matrix[ti, di] if prob_matrix.size else float("nan"))
                    ),
                )

        # Second pass greedy for unmatched pairs under threshold
        if self.tracker_cfg.second_pass_greedy:
            greedy_th = (
                self.tracker_cfg.match_threshold
                if self.tracker_cfg.greedy_match_threshold <= 0
                else self.tracker_cfg.greedy_match_threshold
            )
            remaining_tracks = [ti for ti in range(n_tracks) if ti not in matched_track_indices]
            remaining_dets = [di for di in range(n_dets) if di not in matched_det_indices]
            candidates = []
            for ti in remaining_tracks:
                for di in remaining_dets:
                    c = float(cost_matrix[ti, di])
                    if np.isfinite(c) and c <= greedy_th:
                        candidates.append((c, ti, di))
            candidates.sort(key=lambda x: x[0])

            for c, ti, di in candidates:
                if ti in matched_track_indices or di in matched_det_indices:
                    continue
                det = detections[di]
                self._update_track(self.tracks[ti], det, frame_index)
                matched_track_indices.add(int(ti))
                matched_det_indices.add(int(di))

                feat = pair_features.get((int(ti), int(di)), {})
                det_to_assignment[int(di)] = DetectionAssignment(
                    det_idx=int(di),
                    track_id=self.tracks[ti].track_id,
                    status="matched",
                    cost=float(c),
                    reason=str(feat.get("reason", "matched_2nd_pass")),
                    center_norm=float(feat.get("center_norm", float("nan"))),
                    aspect_rel_diff=float(feat.get("aspect_rel_diff", float("nan"))),
                    height_depth_rel_diff=float(feat.get("height_depth_rel_diff", float("nan"))),
                    cosine_distance=float(feat.get("cosine_distance", float("nan"))),
                    depth_delta=float(feat.get("depth_delta", float("nan"))),
                    match_probability=float(
                        feat.get("match_probability", prob_matrix[ti, di] if prob_matrix.size else float("nan"))
                    ),
                )

        # Third pass: spatial relink for depth/illumination outliers.
        if self.tracker_cfg.spatial_relink_enabled:
            remaining_tracks = [ti for ti in range(n_tracks) if ti not in matched_track_indices]
            remaining_dets = [di for di in range(n_dets) if di not in matched_det_indices]
            relink_candidates = []

            for ti in remaining_tracks:
                tr = self.tracks[ti]
                for di in remaining_dets:
                    det = detections[di]

                    if self.tracker_cfg.relink_require_same_class and int(tr.class_id) != int(det.class_id):
                        continue

                    iou = self._calculate_iou(tr.bbox_cxcywh, det.bbox_cxcywh)
                    if iou < float(self.tracker_cfg.relink_min_iou):
                        continue

                    cx_t, cy_t = tr.centroid
                    cx_d, cy_d = det.centroid
                    center_dist = np.hypot(cx_t - cx_d, cy_t - cy_d)
                    center_norm = float(center_dist / max(image_diag, 1e-6))
                    if center_norm > float(self.tracker_cfg.relink_max_center_norm):
                        continue

                    aspect_rel_diff = float(
                        abs(float(tr.aspect_ratio) - float(det.aspect_ratio))
                        / max(abs(float(tr.aspect_ratio)), abs(float(det.aspect_ratio)), 1e-6)
                    )
                    if aspect_rel_diff > float(self.tracker_cfg.relink_max_aspect_rel_diff):
                        continue

                    cosine_distance = self._cosine_distance(tr.embedding, det.embedding)
                    if np.isfinite(cosine_distance):
                        if cosine_distance > float(self.tracker_cfg.relink_max_cosine_distance):
                            continue

                    relink_candidates.append(
                        (
                            -iou,  # maximize IoU
                            center_norm,  # minimize center shift
                            cosine_distance if np.isfinite(cosine_distance) else 1.0,  # minimize visual distance
                            ti,
                            di,
                            iou,
                            aspect_rel_diff,
                            cosine_distance,
                        )
                    )

            relink_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            for _, center_norm, cos_dist, ti, di, iou, aspect_rel_diff, cosine_distance in relink_candidates:
                if ti in matched_track_indices or di in matched_det_indices:
                    continue

                det = detections[di]
                self._update_track(self.tracks[ti], det, frame_index)
                matched_track_indices.add(int(ti))
                matched_det_indices.add(int(di))

                c = float(cost_matrix[ti, di]) if cost_matrix.size else float("nan")
                p = (
                    float(prob_matrix[ti, di])
                    if prob_matrix.size and np.isfinite(prob_matrix[ti, di])
                    else float(np.exp(-c)) if np.isfinite(c) else float("nan")
                )
                det_to_assignment[int(di)] = DetectionAssignment(
                    det_idx=int(di),
                    track_id=self.tracks[ti].track_id,
                    status="matched",
                    cost=c,
                    reason="spatial_relink",
                    center_norm=float(center_norm),
                    aspect_rel_diff=float(aspect_rel_diff),
                    height_depth_rel_diff=float("nan"),
                    cosine_distance=float(cosine_distance) if np.isfinite(cosine_distance) else float("nan"),
                    depth_delta=float(abs(float(self.tracks[ti].depth_center) - float(det.depth_center))),
                    match_probability=p,
                )

        # Age unmatched tracks
        unmatched_track_indices = [ti for ti in range(n_tracks) if ti not in matched_track_indices]
        for ti in unmatched_track_indices:
            self.tracks[ti].misses += 1
            self.tracks[ti].age += 1

        # Unmatched detections become new tracks (unless suppressed by containment rule)
        suppressed = 0
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue

            if self._should_suppress_contained_new(det, unmatched_track_indices):
                suppressed += 1
                det_to_assignment[int(di)] = DetectionAssignment(
                    det_idx=int(di),
                    track_id=-1,
                    status="suppressed",
                    cost=float("nan"),
                    reason="contained_suppressed",
                    depth_delta=float(det.depth_delta),
                    match_probability=0.0,
                )
                continue

            new_t = self._new_track(det, frame_index=frame_index)
            self.tracks.append(new_t)
            det_to_assignment[int(di)] = DetectionAssignment(
                det_idx=int(di),
                track_id=new_t.track_id,
                status="new",
                cost=-1.0,
                reason="new_no_match",
                depth_delta=float(det.depth_delta),
                match_probability=1.0,
            )

        lost_states = self._prune_dead_tracks()

        assignments = [det_to_assignment[i] for i in sorted(det_to_assignment.keys())]
        result.assignments = assignments
        result.num_matched = sum(1 for a in assignments if a.status == "matched")
        result.num_new = sum(1 for a in assignments if a.status == "new")
        result.num_suppressed = suppressed
        result.lost_tracks = [self._to_lost_record(t) for t in lost_states]
        result.num_lost_tracks = len(result.lost_tracks)
        result.num_active_tracks = len(self.tracks)
        return result
