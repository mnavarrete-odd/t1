from __future__ import annotations

from .base import KeyframeHandler


class AreaEmptyHandler(KeyframeHandler):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._stable_empty_frames = 0
        self._prev_area_empty = False

    def update(self, ctx):
        if not ctx.area_ready:
            self._stable_empty_frames = 0
            self._prev_area_empty = False
            return []

        if ctx.metrics.count_in_area != 0:
            self._stable_empty_frames = 0
            self._prev_area_empty = False
            return []

        if ctx.metrics.movement_score > self.cfg.stable_area_movement_max:
            self._stable_empty_frames = 0
            self._prev_area_empty = False
            return []

        self._stable_empty_frames += 1
        area_empty = self._stable_empty_frames >= self.cfg.stable_empty_frames

        events = []
        if area_empty and not self._prev_area_empty:
            events.append(
                ctx.emit_event(
                    "KF-AREA-EMPTY",
                    ctx.frame_index,
                    ctx.event_image,
                    ctx.metrics,
                    ctx.detections,
                )
            )

        self._prev_area_empty = area_empty
        return events
