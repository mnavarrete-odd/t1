from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .distribution_wrapper import DistributionWrapper
from .types import PreparedDetection, TrackState


@dataclass
class CostConfig:
    use_bayes: bool = True
    correct_distributions_path: str = ""
    incorrect_distributions_path: str = ""
    prob_correct_prior: float = 0.01811192458792822
    hard_class_mismatch: bool = False
    class_mismatch_penalty: float = 0.0
    large_cost: float = 1e8
    eps: float = 1e-10

    # Optional class-specific depth discount (Andina tetra-pack style)
    enable_depth_discount_by_class: bool = False
    depth_discount_class_ids: tuple[int, ...] = ()
    depth_discount_factor: float = 3.0

    # Heuristic fallback weights
    w_cosine: float = 0.55
    w_center: float = 0.20
    w_height_depth: float = 0.20
    w_aspect: float = 0.05

    # Heuristic fallback thresholds
    center_same_norm: float = 0.08
    max_center_dist_norm: float = 0.55
    max_aspect_ratio_rel_diff: float = 0.25
    max_height_depth_rel_diff: float = 0.35
    max_cosine_distance: float = 0.35
    max_move_cosine_distance: float = 0.25


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 1.0
    if a.size == 0 or b.size == 0:
        return 1.0
    if a.shape != b.shape:
        return 1.0

    num = float(np.dot(a, b))
    den = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-8
    sim = np.clip(num / den, -1.0, 1.0)
    return 1.0 - sim


def _relative_diff(a: float, b: float) -> float:
    den = max(abs(a), abs(b), 1e-6)
    return float(abs(a - b) / den)


def _safe_embeddings(detections: list[PreparedDetection], tracks: list[TrackState]):
    if len(detections) == 0 or len(tracks) == 0:
        return np.zeros((len(tracks), 0), dtype=np.float64), np.zeros((len(detections), 0), dtype=np.float64)

    emb_dim = 0
    for det in detections:
        if det.embedding is not None and det.embedding.size > 0:
            emb_dim = int(det.embedding.shape[0])
            break
    if emb_dim == 0:
        for tr in tracks:
            if tr.embedding is not None and tr.embedding.size > 0:
                emb_dim = int(tr.embedding.shape[0])
                break

    if emb_dim == 0:
        return (
            np.zeros((len(tracks), 0), dtype=np.float64),
            np.zeros((len(detections), 0), dtype=np.float64),
        )

    track_embeddings = np.zeros((len(tracks), emb_dim), dtype=np.float64)
    for i, tr in enumerate(tracks):
        if tr.embedding is None:
            continue
        emb = np.asarray(tr.embedding).reshape(-1)
        if emb.size == emb_dim:
            track_embeddings[i] = emb

    det_embeddings = np.zeros((len(detections), emb_dim), dtype=np.float64)
    for i, det in enumerate(detections):
        if det.embedding is None:
            continue
        emb = np.asarray(det.embedding).reshape(-1)
        if emb.size == emb_dim:
            det_embeddings[i] = emb

    return track_embeddings, det_embeddings


def _build_heuristic_cost_matrix(
    tracks: list[TrackState],
    detections: list[PreparedDetection],
    image_diag: float,
    cfg: CostConfig,
):
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost = np.full((n_tracks, n_dets), float("inf"), dtype=np.float64)
    prob = np.zeros((n_tracks, n_dets), dtype=np.float64)
    features = {}

    for ti, tr in enumerate(tracks):
        for di, det in enumerate(detections):
            if cfg.hard_class_mismatch and tr.class_id != det.class_id:
                continue

            cx_t, cy_t = tr.centroid
            cx_d, cy_d = det.centroid

            center_dist = np.hypot(cx_t - cx_d, cy_t - cy_d)
            center_norm = float(center_dist / max(image_diag, 1e-6))

            aspect_rel_diff = _relative_diff(tr.aspect_ratio, det.aspect_ratio)

            height_depth_valid = np.isfinite(tr.height_depth) and np.isfinite(det.height_depth)
            height_depth_rel_diff = (
                _relative_diff(tr.height_depth, det.height_depth) if height_depth_valid else 0.0
            )

            cosine_distance = _cosine_distance(tr.embedding, det.embedding)
            center_same = center_norm <= cfg.center_same_norm
            center_close = center_norm <= cfg.max_center_dist_norm
            ratio_close = aspect_rel_diff <= cfg.max_aspect_ratio_rel_diff
            height_close = (
                True if not height_depth_valid else height_depth_rel_diff <= cfg.max_height_depth_rel_diff
            )
            dino_close = cosine_distance <= cfg.max_cosine_distance
            dino_move_strong = cosine_distance <= cfg.max_move_cosine_distance

            reason = "feature_not_similar"
            c = float("inf")

            if center_same and ratio_close and not height_close:
                reason = "stacked_height_change"
            elif center_same and ratio_close and height_close and dino_close:
                c = (
                    cfg.w_cosine * cosine_distance
                    + cfg.w_center * center_norm
                    + cfg.w_height_depth * height_depth_rel_diff
                    + cfg.w_aspect * aspect_rel_diff
                )
                reason = "same_strict"
            elif center_close and ratio_close and height_close and dino_move_strong:
                c = (
                    cfg.w_cosine * cosine_distance
                    + cfg.w_center * center_norm
                    + cfg.w_height_depth * height_depth_rel_diff
                    + cfg.w_aspect * aspect_rel_diff
                )
                reason = "moved_dino_supported"
            elif center_close and ratio_close and (not center_same) and dino_move_strong:
                height_penalty = min(1.0, height_depth_rel_diff) if height_depth_valid else 0.0
                c = (
                    cfg.w_cosine * cosine_distance
                    + cfg.w_center * center_norm
                    + cfg.w_height_depth * height_penalty
                    + cfg.w_aspect * aspect_rel_diff
                )
                reason = "moved_depth_noisy"

            if np.isfinite(c):
                if tr.class_id != det.class_id and cfg.class_mismatch_penalty > 0.0:
                    c += float(cfg.class_mismatch_penalty)
                cost[ti, di] = float(c)
                prob[ti, di] = float(np.exp(-c))
                features[(ti, di)] = {
                    "reason": reason,
                    "center_norm": center_norm,
                    "aspect_rel_diff": aspect_rel_diff,
                    "height_depth_rel_diff": float(height_depth_rel_diff),
                    "cosine_distance": cosine_distance,
                    "depth_delta": abs(float(tr.depth_center) - float(det.depth_center)),
                    "match_probability": prob[ti, di],
                }

    return cost, features, prob


class BayesianCostModel:
    def __init__(self, cfg: CostConfig):
        self.cfg = cfg
        self.correct = DistributionWrapper()
        self.incorrect = DistributionWrapper()
        self.enabled = False

        if not cfg.use_bayes:
            return

        c_path = str(cfg.correct_distributions_path or "").strip()
        i_path = str(cfg.incorrect_distributions_path or "").strip()
        if not c_path or not i_path:
            return
        if not Path(c_path).exists() or not Path(i_path).exists():
            return

        self.correct.load(c_path)
        self.incorrect.load(i_path)
        self.enabled = True

    def build_cost_matrix(
        self,
        tracks: list[TrackState],
        detections: list[PreparedDetection],
        image_diag: float,
    ):
        if not self.enabled:
            return _build_heuristic_cost_matrix(tracks, detections, image_diag, self.cfg)

        n_tracks = len(tracks)
        n_dets = len(detections)
        if n_tracks == 0 or n_dets == 0:
            return (
                np.zeros((n_tracks, n_dets), dtype=np.float64),
                {},
                np.zeros((n_tracks, n_dets), dtype=np.float64),
            )

        track_embeddings, det_embeddings = _safe_embeddings(detections, tracks)
        track_centroids = np.asarray([t.centroid for t in tracks], dtype=np.float64)
        det_centroids = np.asarray([d.centroid for d in detections], dtype=np.float64)
        track_depth = np.asarray([t.depth_center for t in tracks], dtype=np.float64)
        det_depth = np.asarray([d.depth_center for d in detections], dtype=np.float64)
        track_aspect = np.asarray([t.aspect_ratio for t in tracks], dtype=np.float64)
        det_aspect = np.asarray([d.aspect_ratio for d in detections], dtype=np.float64)
        track_height_depth = np.asarray([t.height_depth for t in tracks], dtype=np.float64)
        det_height_depth = np.asarray([d.height_depth for d in detections], dtype=np.float64)
        track_classes = np.asarray([t.class_id for t in tracks], dtype=np.int32)
        det_classes = np.asarray([d.class_id for d in detections], dtype=np.int32)

        center_dist_matrix = np.linalg.norm(
            track_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :],
            axis=2,
        )
        center_norm_matrix = center_dist_matrix / max(image_diag, 1e-6)

        if track_embeddings.shape[1] > 0 and det_embeddings.shape[1] > 0:
            track_norms = np.linalg.norm(track_embeddings, axis=1, keepdims=True)
            det_norms = np.linalg.norm(det_embeddings, axis=1, keepdims=True)
            denom = (track_norms @ det_norms.T) + self.cfg.eps
            cos_sim_matrix = (track_embeddings @ det_embeddings.T) / denom
            cos_sim_matrix = np.clip(cos_sim_matrix, -1.0, 1.0)
            invalid_pair = (track_norms <= 1e-8) | (det_norms.T <= 1e-8)
        else:
            cos_sim_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
            invalid_pair = np.ones((n_tracks, n_dets), dtype=bool)

        depth_delta_matrix = np.abs(track_depth[:, np.newaxis] - det_depth[np.newaxis, :])
        nan_depth_mask = ~np.isfinite(depth_delta_matrix)

        if self.cfg.enable_depth_discount_by_class and len(self.cfg.depth_discount_class_ids) > 0:
            det_discount_mask = np.isin(det_classes, np.asarray(self.cfg.depth_discount_class_ids, dtype=np.int32))
            depth_delta_matrix[:, det_discount_mask] = (
                depth_delta_matrix[:, det_discount_mask] / max(self.cfg.depth_discount_factor, 1e-6)
            )

        aspect_ratio_matrix = (
            track_aspect[:, np.newaxis] / np.maximum(det_aspect[np.newaxis, :], 1e-6)
        )
        aspect_rel_diff = np.abs(track_aspect[:, np.newaxis] - det_aspect[np.newaxis, :]) / np.maximum(
            np.maximum(track_aspect[:, np.newaxis], det_aspect[np.newaxis, :]),
            1e-6,
        )
        height_depth_rel_diff = np.abs(
            track_height_depth[:, np.newaxis] - det_height_depth[np.newaxis, :]
        ) / np.maximum(
            np.maximum(np.abs(track_height_depth[:, np.newaxis]), np.abs(det_height_depth[np.newaxis, :])),
            1e-6,
        )

        eps = max(self.cfg.eps, 1e-12)

        prob_correct_depth = np.ones((n_tracks, n_dets), dtype=np.float64)
        prob_incorrect_depth = np.ones((n_tracks, n_dets), dtype=np.float64)
        if not np.all(nan_depth_mask):
            v = depth_delta_matrix[~nan_depth_mask]
            prob_correct_depth[~nan_depth_mask] = self.correct.get_probability("depth_delta", v)
            prob_incorrect_depth[~nan_depth_mask] = self.incorrect.get_probability("depth_delta", v)

        prob_correct_center = self.correct.get_probability("center_distance", center_dist_matrix.reshape(-1)).reshape(n_tracks, n_dets)
        prob_incorrect_center = self.incorrect.get_probability("center_distance", center_dist_matrix.reshape(-1)).reshape(n_tracks, n_dets)
        prob_correct_cos = self.correct.get_probability("cos_similarity", cos_sim_matrix.reshape(-1)).reshape(n_tracks, n_dets)
        prob_incorrect_cos = self.incorrect.get_probability("cos_similarity", cos_sim_matrix.reshape(-1)).reshape(n_tracks, n_dets)
        prob_correct_aspect = self.correct.get_probability("bbox_aspect_ratio", aspect_ratio_matrix.reshape(-1)).reshape(n_tracks, n_dets)
        prob_incorrect_aspect = self.incorrect.get_probability("bbox_aspect_ratio", aspect_ratio_matrix.reshape(-1)).reshape(n_tracks, n_dets)

        lr_depth = prob_correct_depth / (prob_incorrect_depth + eps)
        lr_center = prob_correct_center / (prob_incorrect_center + eps)
        lr_cos = prob_correct_cos / (prob_incorrect_cos + eps)
        lr_aspect = prob_correct_aspect / (prob_incorrect_aspect + eps)

        prior = float(np.clip(self.cfg.prob_correct_prior, 1e-6, 1.0 - 1e-6))
        prior_odds = prior / (1.0 - prior)
        posterior_odds = prior_odds * lr_depth * lr_center * lr_cos * lr_aspect
        prob_match = posterior_odds / (1.0 + posterior_odds + eps)
        prob_match = np.clip(prob_match, eps, 1.0)

        cost = -np.log(prob_match + eps)

        if np.any(invalid_pair):
            cost[invalid_pair] = float(self.cfg.large_cost)
            prob_match[invalid_pair] = eps

        if self.cfg.hard_class_mismatch:
            mismatch = track_classes[:, np.newaxis] != det_classes[np.newaxis, :]
            cost[mismatch] = float(self.cfg.large_cost)
            prob_match[mismatch] = eps
        elif self.cfg.class_mismatch_penalty > 0:
            mismatch = track_classes[:, np.newaxis] != det_classes[np.newaxis, :]
            cost[mismatch] += float(self.cfg.class_mismatch_penalty)

        features: Dict[tuple[int, int], dict] = {}
        cosine_distance = 1.0 - cos_sim_matrix
        for ti in range(n_tracks):
            for di in range(n_dets):
                c = float(cost[ti, di])
                if not np.isfinite(c):
                    continue
                features[(ti, di)] = {
                    "reason": "bayes",
                    "center_norm": float(center_norm_matrix[ti, di]),
                    "aspect_rel_diff": float(aspect_rel_diff[ti, di]),
                    "height_depth_rel_diff": float(height_depth_rel_diff[ti, di]),
                    "cosine_distance": float(cosine_distance[ti, di]),
                    "depth_delta": float(depth_delta_matrix[ti, di]),
                    "match_probability": float(prob_match[ti, di]),
                }

        return cost.astype(np.float64), features, prob_match.astype(np.float64)


def build_cost_matrix(
    tracks: list[TrackState],
    detections: list[PreparedDetection],
    image_diag: float,
    cfg: CostConfig,
    model: BayesianCostModel | None = None,
):
    if model is None:
        model = BayesianCostModel(cfg)
    return model.build_cost_matrix(tracks, detections, image_diag=image_diag)
