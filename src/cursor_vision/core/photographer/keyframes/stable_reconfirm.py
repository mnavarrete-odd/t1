from __future__ import annotations

from .base import KeyframeHandler


class StableReconfirmHandler(KeyframeHandler):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._pending = False
        self._stable_frames = 0
        self._prev_count: int | None = None
        self._pending_from_occlusion = False
        self._pending_from_hand = False
        self._pending_from_person = False

    def update(self, ctx):
        if not ctx.area_ready:
            self._reset()
            return []

        occlusion_triggered = ctx.metrics.occlusion_ratio >= self.cfg.occlusion_start_ratio
        person_triggered = self.cfg.person_near_enabled and ctx.metrics.has_person_near
        hand_triggered = bool(ctx.hand_near_dets)
        event_triggered = occlusion_triggered or person_triggered or hand_triggered

        if event_triggered:
            if not self._pending:
                self._pending = True
                self._pending_from_occlusion = occlusion_triggered
                self._pending_from_person = person_triggered
                self._pending_from_hand = hand_triggered
            else:
                self._pending_from_occlusion = self._pending_from_occlusion or occlusion_triggered
                self._pending_from_person = self._pending_from_person or person_triggered
                self._pending_from_hand = self._pending_from_hand or hand_triggered
            self._stable_frames = 0
            self._prev_count = ctx.metrics.count_in_area
            return []

        if not self._pending:
            self._prev_count = ctx.metrics.count_in_area
            return []

        if ctx.metrics.movement_score > self.cfg.stable_area_movement_max:
            self._stable_frames = 0
            self._prev_count = ctx.metrics.count_in_area
            return []

        if self.cfg.stable_area_require_count_stability:
            if self._prev_count is not None and ctx.metrics.count_in_area != self._prev_count:
                self._stable_frames = 0
                self._prev_count = ctx.metrics.count_in_area
                return []

        self._stable_frames += 1
        events = []
        if self._stable_frames >= self.cfg.stable_reconfirm_frames:
            skip_kf_test = self._pending_from_occlusion
            events.append(
                ctx.emit_event(
                    "KF-STABLE-RECONFIRM",
                    ctx.frame_index,
                    ctx.event_image,
                    ctx.metrics,
                    ctx.detections,
                    kf_test_skip=skip_kf_test,
                )
            )
            self._pending = False
            self._stable_frames = 0
            self._pending_from_occlusion = False
            self._pending_from_person = False
            self._pending_from_hand = False

        self._prev_count = ctx.metrics.count_in_area
        return events

    def _reset(self) -> None:
        self._pending = False
        self._stable_frames = 0
        self._prev_count = None
        self._pending_from_occlusion = False
        self._pending_from_person = False
        self._pending_from_hand = False
