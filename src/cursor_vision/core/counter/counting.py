from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .types import FrameResult, PreparedDetection


@dataclass
class CountingConfig:
    enabled: bool = True
    ignore_bootstrap_new_on_first_frame: bool = True

    # Unit estimation from depth delta (floor_depth - product_depth)
    use_depth_for_units: bool = True
    stackable_class_ids: Tuple[int, ...] = ()
    default_height_m: float = 0.20
    class_heights_m: Dict[int, float] = field(default_factory=dict)
    tall_height_threshold_m: float = 0.14
    round_up_threshold_tall: float = 0.50
    round_up_threshold_short: float = 0.60
    min_depth_delta_m: float = 0.01
    require_positive_depth_delta: bool = True
    min_units_per_detection: int = 1
    max_units_per_detection: int = 6

    # Format multipliers
    class_factors: Dict[int, float] = field(default_factory=dict)
    use_containment_factors: bool = False
    contained_by_rules: Dict[int, Tuple[int, float]] = field(default_factory=dict)
    contains_rules: Dict[int, Tuple[int, float]] = field(default_factory=dict)

    # State classification
    state_eps: float = 1e-6


class FrameChangeCounter:
    def __init__(self, cfg: CountingConfig):
        self.cfg = cfg
        self.running_units = 0.0
        self._num_frames = 0

    def reset(self) -> None:
        self.running_units = 0.0
        self._num_frames = 0

    def _factor_for_class(self, class_id: int) -> float:
        cls = int(class_id)
        if cls in self.cfg.class_factors:
            return float(self.cfg.class_factors[cls])

        if not self.cfg.use_containment_factors:
            return 1.0

        if cls in self.cfg.contained_by_rules:
            factor = float(self.cfg.contained_by_rules[cls][1])
            return 1.0 / max(factor, 1e-6)

        if cls in self.cfg.contains_rules:
            return float(self.cfg.contains_rules[cls][1])

        return 1.0

    def _estimate_base_units(self, class_id: int, depth_delta: float) -> tuple[float, str]:
        min_units = max(1, int(self.cfg.min_units_per_detection))
        max_units = max(min_units, int(self.cfg.max_units_per_detection))

        if not self.cfg.use_depth_for_units:
            return float(min_units), "depth_disabled"

        if len(self.cfg.stackable_class_ids) > 0 and int(class_id) not in self.cfg.stackable_class_ids:
            return float(min_units), "non_stackable"

        if not np.isfinite(depth_delta):
            return float(min_units), "depth_missing"

        d = float(depth_delta)
        if self.cfg.require_positive_depth_delta and d < float(self.cfg.min_depth_delta_m):
            return 0.0, "depth_too_small"

        height = float(self.cfg.class_heights_m.get(int(class_id), self.cfg.default_height_m))
        if not np.isfinite(height) or height <= 0.0:
            return float(min_units), "invalid_height"

        n = d / height
        n_floor = int(np.floor(n))
        remainder = n - float(n_floor)

        round_thr = (
            float(self.cfg.round_up_threshold_tall)
            if height >= float(self.cfg.tall_height_threshold_m)
            else float(self.cfg.round_up_threshold_short)
        )
        if remainder > round_thr:
            n_floor += 1

        n_floor = max(min_units, n_floor)
        n_floor = min(max_units, n_floor)
        return float(n_floor), "depth_estimated"

    def _estimate_units(self, class_id: int, depth_delta: float) -> tuple[float, float, str]:
        base_units, reason = self._estimate_base_units(class_id=class_id, depth_delta=depth_delta)
        if base_units <= self.cfg.state_eps:
            return 0.0, 1.0, reason

        factor = self._factor_for_class(class_id)
        units = max(0.0, base_units * factor)
        return float(units), float(factor), reason

    def apply(self, result: FrameResult, detections: List[PreparedDetection]) -> FrameResult:
        if not self.cfg.enabled:
            result.added_units = 0.0
            result.removed_units = 0.0
            result.net_units = 0.0
            result.running_units = float(self.running_units)
            result.change_state = "no_change"
            result.change_detail = "none"
            return result

        added_units = 0.0
        removed_units = 0.0
        added_by_class: Dict[str, float] = {}
        removed_by_class: Dict[str, float] = {}

        is_first_frame = self._num_frames == 0
        for assignment in result.assignments:
            if assignment.status != "new":
                assignment.count_action = "none"
                assignment.count_reason = ""
                assignment.count_units = 0.0
                continue

            if assignment.det_idx < 0 or assignment.det_idx >= len(detections):
                assignment.count_action = "ignored"
                assignment.count_reason = "invalid_det_index"
                assignment.count_units = 0.0
                continue

            det = detections[assignment.det_idx]
            if (
                is_first_frame
                and self.cfg.ignore_bootstrap_new_on_first_frame
                and assignment.reason == "new_bootstrap"
            ):
                assignment.count_action = "ignored"
                assignment.count_reason = "bootstrap_ignored"
                assignment.count_units = 0.0
                continue

            units, _, reason = self._estimate_units(
                class_id=int(det.class_id),
                depth_delta=float(det.depth_delta),
            )
            assignment.count_units = float(units)
            assignment.count_reason = reason
            if units <= self.cfg.state_eps:
                assignment.count_action = "ignored"
                continue

            assignment.count_action = "added"
            added_units += float(units)
            key = str(det.class_name)
            added_by_class[key] = float(added_by_class.get(key, 0.0) + units)

        removed_count = 0
        for lost in result.lost_tracks:
            units, factor, reason = self._estimate_units(
                class_id=int(lost.class_id),
                depth_delta=float(lost.depth_delta),
            )
            lost.count_units = float(units)
            lost.count_factor = float(factor)
            lost.count_reason = reason

            if units <= self.cfg.state_eps:
                lost.count_action = "ignored"
                continue

            lost.count_action = "removed"
            removed_count += 1
            removed_units += float(units)
            key = str(lost.class_name)
            removed_by_class[key] = float(removed_by_class.get(key, 0.0) + units)

        net_units = float(added_units - removed_units)
        self.running_units += net_units

        eps = float(self.cfg.state_eps)
        if added_units > eps and removed_units <= eps:
            change_state = "added"
            change_detail = "added_only"
        elif removed_units > eps and added_units <= eps:
            change_state = "removed"
            change_detail = "removed_only"
        elif added_units > eps and removed_units > eps:
            if net_units > eps:
                change_state = "added"
            elif net_units < -eps:
                change_state = "removed"
            else:
                change_state = "no_change"
            change_detail = "mixed"
        else:
            change_state = "no_change"
            change_detail = "none"

        result.num_removed = int(removed_count)
        result.added_units = float(added_units)
        result.removed_units = float(removed_units)
        result.net_units = float(net_units)
        result.running_units = float(self.running_units)
        result.change_state = change_state
        result.change_detail = change_detail
        result.added_by_class = added_by_class
        result.removed_by_class = removed_by_class

        self._num_frames += 1
        return result
