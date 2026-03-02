from __future__ import annotations

from collections import deque

from .base import KeyframeHandler


class ProductInHandHandler(KeyframeHandler):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        history_size = max(
            1,
            self.cfg.product_pre_offset_frames + 1,
            self.cfg.product_post_offset_frames + 1,
        )
        self._history: deque[tuple[int, object, object]] = deque(maxlen=history_size)
        self._last_without_hand: tuple[int, object, object] | None = None
        self._active = False
        self._above_frames = 0
        self._below_frames = 0
        self._event_id = 0
        self._active_id: int | None = None
        self._pending_post_index: int | None = None

    def update(self, ctx):
        events = []
        base_image = ctx.render_labels(ctx.hand_near_dets + ctx.box_dets)
        self._history.append((ctx.frame_index, base_image, ctx.metrics))

        if not ctx.area_ready:
            self._reset_state()
            return events

        in_hand = bool(ctx.hand_near_dets)

        if not in_hand:
            self._last_without_hand = (ctx.frame_index, base_image, ctx.metrics)

        if in_hand:
            self._above_frames += 1
        else:
            self._above_frames = 0

        if not in_hand:
            self._below_frames += 1
        else:
            self._below_frames = 0

        if not self._active and self._above_frames >= self.cfg.product_start_frames:
            self._active = True
            self._event_id += 1
            self._active_id = self._event_id
            if self.cfg.product_save_prepost and self.cfg.product_post_offset_frames > 0:
                self._pending_post_index = ctx.frame_index + self.cfg.product_post_offset_frames

            ev_id = self._active_id or self._event_id
            occ_filename = f"KF-PRODUCT-IN-HAND-{ev_id:03d}_01_{ctx.frame_index:06d}"
            events.append(
                ctx.emit_event(
                    "KF-PRODUCT-IN-HAND",
                    ctx.frame_index,
                    base_image,
                    ctx.metrics,
                    ctx.detections,
                    folder_tag="KF-PRODUCT-IN-HAND",
                    filename_override=occ_filename,
                )
            )

            if self.cfg.product_save_prepost:
                if self.cfg.product_pre_offset_frames > 0:
                    pre_snap = self._history_offset(self.cfg.product_pre_offset_frames)
                else:
                    pre_snap = self._last_without_hand
                if pre_snap is not None:
                    snap_idx, snap_img, snap_metrics = pre_snap
                    pre_filename = f"KF-PRODUCT-IN-HAND-{ev_id:03d}_00_{snap_idx:06d}"
                    events.append(
                        ctx.emit_event(
                            "KF-PRODUCT-IN-HAND",
                            snap_idx,
                            snap_img,
                            snap_metrics,
                            ctx.detections,
                            folder_tag="KF-PRODUCT-IN-HAND",
                            filename_override=pre_filename,
                        )
                    )

        if self._active and self._below_frames >= self.cfg.product_end_frames:
            self._active = False
            if self.cfg.product_save_prepost and self.cfg.product_post_offset_frames <= 0:
                ev_id = self._active_id or self._event_id
                post_filename = f"KF-PRODUCT-IN-HAND-{ev_id:03d}_02_{ctx.frame_index:06d}"
                self._active_id = None
                events.append(
                    ctx.emit_event(
                        "KF-PRODUCT-IN-HAND",
                        ctx.frame_index,
                        base_image,
                        ctx.metrics,
                        ctx.detections,
                        folder_tag="KF-PRODUCT-IN-HAND",
                        filename_override=post_filename,
                    )
                )

        if (
            self.cfg.product_save_prepost
            and self._pending_post_index is not None
            and ctx.frame_index >= self._pending_post_index
        ):
            offset = ctx.frame_index - self._pending_post_index
            snap = self._history_offset(offset) or self._history_offset(0)
            if snap is not None:
                snap_idx, snap_img, snap_metrics = snap
                ev_id = self._active_id or self._event_id
                post_filename = f"KF-PRODUCT-IN-HAND-{ev_id:03d}_02_{snap_idx:06d}"
                events.append(
                    ctx.emit_event(
                        "KF-PRODUCT-IN-HAND",
                        snap_idx,
                        snap_img,
                        snap_metrics,
                        ctx.detections,
                        folder_tag="KF-PRODUCT-IN-HAND",
                        filename_override=post_filename,
                    )
                )
            self._pending_post_index = None
            self._active_id = None

        return events

    def _history_offset(self, offset: int):
        if offset < 0:
            return None
        if len(self._history) <= offset:
            return None
        return self._history[-(offset + 1)]

    def _reset_state(self) -> None:
        self._active = False
        self._above_frames = 0
        self._below_frames = 0
        self._last_without_hand = None
        self._active_id = None
        self._pending_post_index = None
