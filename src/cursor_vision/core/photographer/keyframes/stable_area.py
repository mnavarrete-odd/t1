from __future__ import annotations

from .base import KeyframeHandler


class StableAreaHandler(KeyframeHandler):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._stable_frames = 0
        self._prev_count: int | None = None
        self._prev_area_stable = False

    def update(self, ctx):
        if not ctx.area_ready:
            self._stable_frames = 0
            self._prev_area_stable = False
            self._prev_count = ctx.metrics.count_in_area
            return []

        if ctx.metrics.movement_score > self.cfg.stable_area_movement_max:
            self._stable_frames = 0
            self._prev_area_stable = False
            self._prev_count = ctx.metrics.count_in_area
            return []

        if self.cfg.stable_area_require_count_stability:
            if self._prev_count is not None and ctx.metrics.count_in_area != self._prev_count:
                self._stable_frames = 0
                self._prev_area_stable = False
                self._prev_count = ctx.metrics.count_in_area
                return []

        self._stable_frames += 1
        area_stable = self._stable_frames >= self.cfg.stable_area_frames

        events = []
        if area_stable and not self._prev_area_stable:
            events.append(
                ctx.emit_event(
                    "KF-STABLE-AREA",
                    ctx.frame_index,
                    ctx.event_image,
                    ctx.metrics,
                    ctx.detections,
                )
            )

        self._prev_area_stable = area_stable
        self._prev_count = ctx.metrics.count_in_area
        return events
