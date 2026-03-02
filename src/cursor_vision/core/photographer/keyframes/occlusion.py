from __future__ import annotations

from collections import deque
import cv2

from .base import KeyframeHandler
from ..utils import boxes_in_area, coverage_ratio


class OcclusionHandler(KeyframeHandler):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        history_size = max(
            1,
            self.cfg.occlusion_pre_offset_frames + 1,
            self.cfg.occlusion_post_offset_frames + 1,
        )
        self._history: deque[tuple[int, object, object]] = deque(maxlen=history_size)
        self._last_below_start: tuple[int, object, object] | None = None
        self._occlusion_active = False
        self._occlusion_above_frames = 0
        self._occlusion_below_frames = 0
        self._occlusion_id = 0
        self._active_occlusion_id: int | None = None
        self._pending_post_index: int | None = None
        self._item_prev_count: int | None = None
        self._item_prev_coverage: float | None = None
        self._item_change_frames = 0
        self._item_stable_frames = 0
        self._item_event_id = 0
        self._item_active = False
        self._item_cooldown = 0
        self._item_last_stable: tuple[int, object, object] | None = None

    def update(self, ctx):
        events = []
        ratio = ctx.metrics.occlusion_ratio
        person_with_occ = [
            type(p)(
                class_id=p.class_id,
                class_name=p.class_name,
                bbox=p.bbox,
                confidence=p.confidence,
                extra_data={**(p.extra_data or {}), "occ": ratio},
            )
            for p in ctx.person_dets
        ]
        base_image = ctx.render_labels(person_with_occ + ctx.box_dets)
        self._history.append((ctx.frame_index, base_image, ctx.metrics))

        if not ctx.area_ready:
            self._reset_state()
            return events

        if ratio < self.cfg.occlusion_start_ratio:
            self._last_below_start = (ctx.frame_index, base_image, ctx.metrics)

        if ratio >= self.cfg.occlusion_start_ratio:
            self._occlusion_above_frames += 1
        else:
            self._occlusion_above_frames = 0

        if ratio <= self.cfg.occlusion_end_ratio:
            self._occlusion_below_frames += 1
        else:
            self._occlusion_below_frames = 0

        if not self._occlusion_active and self._occlusion_above_frames >= self.cfg.occlusion_start_frames:
            self._occlusion_active = True
            self._occlusion_id += 1
            self._active_occlusion_id = self._occlusion_id
            if self.cfg.occlusion_post_offset_frames > 0:
                self._pending_post_index = ctx.frame_index + self.cfg.occlusion_post_offset_frames

            occ_id = self._active_occlusion_id or self._occlusion_id
            occ_filename = f"KF-OCLUSION-{occ_id:03d}_01_{ctx.frame_index:06d}"
            occ_image = self._draw_ratio(base_image, ratio)
            events.append(
                ctx.emit_event(
                    "KF-OCCLUSION",
                    ctx.frame_index,
                    occ_image,
                    ctx.metrics,
                    ctx.detections,
                    folder_tag="KF-OCLUSION",
                    filename_override=occ_filename,
                )
            )

            if self.cfg.occlusion_pre_offset_frames > 0:
                pre_snap = self._history_offset(self.cfg.occlusion_pre_offset_frames)
            else:
                pre_snap = self._last_below_start
            if pre_snap is not None:
                snap_idx, snap_img, snap_metrics = pre_snap
                pre_filename = f"KF-OCLUSION-{occ_id:03d}_00_{snap_idx:06d}"
                events.append(
                    ctx.emit_event(
                        "KF-OCCLUSION",
                        snap_idx,
                        snap_img,
                        snap_metrics,
                        ctx.detections,
                        folder_tag="KF-OCLUSION",
                        filename_override=pre_filename,
                    )
                )

        if self._occlusion_active and self._occlusion_below_frames >= self.cfg.occlusion_end_frames:
            if self._item_active:
                occ_id = self._item_event_id
                post_filename = f"KF-OCLUSION-ITEM-{occ_id:03d}_02_{ctx.frame_index:06d}"
                events.append(
                    ctx.emit_event(
                        "KF-OCCLUSION-ITEM",
                        ctx.frame_index,
                        base_image,
                        ctx.metrics,
                        ctx.detections,
                        folder_tag="KF-OCLUSION-ITEM",
                        filename_override=post_filename,
                    )
                )
                self._item_active = False
            self._occlusion_active = False
            if self.cfg.occlusion_post_offset_frames <= 0:
                occ_id = self._active_occlusion_id or self._occlusion_id
                post_filename = f"KF-OCLUSION-{occ_id:03d}_02_{ctx.frame_index:06d}"
                self._active_occlusion_id = None
                events.append(
                    ctx.emit_event(
                        "KF-OCCLUSION",
                        ctx.frame_index,
                        base_image,
                        ctx.metrics,
                        ctx.detections,
                        folder_tag="KF-OCLUSION",
                        filename_override=post_filename,
                    )
                )

        if self._pending_post_index is not None and ctx.frame_index >= self._pending_post_index:
            offset = ctx.frame_index - self._pending_post_index
            snap = self._history_offset(offset) or self._history_offset(0)
            if snap is not None:
                snap_idx, snap_img, snap_metrics = snap
                occ_id = self._active_occlusion_id or self._occlusion_id
                post_filename = f"KF-OCLUSION-{occ_id:03d}_02_{snap_idx:06d}"
                events.append(
                    ctx.emit_event(
                        "KF-OCCLUSION",
                        snap_idx,
                        snap_img,
                        snap_metrics,
                        ctx.detections,
                        folder_tag="KF-OCLUSION",
                        filename_override=post_filename,
                    )
                )
            self._pending_post_index = None
            self._active_occlusion_id = None

        self._update_item_events(ctx, base_image, events)

        return events

    def _history_offset(self, offset: int):
        if offset < 0:
            return None
        if len(self._history) <= offset:
            return None
        return self._history[-(offset + 1)]

    def _reset_state(self) -> None:
        self._occlusion_active = False
        self._occlusion_above_frames = 0
        self._occlusion_below_frames = 0
        self._last_below_start = None
        self._active_occlusion_id = None
        self._pending_post_index = None
        self._item_prev_count = None
        self._item_prev_coverage = None
        self._item_change_frames = 0
        self._item_stable_frames = 0
        self._item_active = False
        self._item_cooldown = 0
        self._item_last_stable = None

    def _draw_ratio(self, image, ratio: float):
        output = image.copy()
        text = f"occ={ratio * 100:.1f}%"
        cv2.putText(
            output,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def _update_item_events(self, ctx, base_image, events) -> None:
        if not self._occlusion_active:
            self._item_prev_count = None
            self._item_prev_coverage = None
            self._item_change_frames = 0
            self._item_stable_frames = 0
            self._item_active = False
            self._item_cooldown = 0
            self._item_last_stable = None
            return

        items = boxes_in_area(ctx.detections, ctx.area_bbox, self.cfg.occlusion_item_classes)
        count = len(items)
        coverage = coverage_ratio(ctx.area_bbox, items)

        if self._item_prev_count is None:
            self._item_prev_count = count
            self._item_prev_coverage = coverage
            self._item_last_stable = (ctx.frame_index, base_image, ctx.metrics)
            return

        delta_count = abs(count - self._item_prev_count)
        delta_cov = abs(coverage - (self._item_prev_coverage or 0.0))
        change_detected = (
            delta_count >= self.cfg.occlusion_change_count_min
            or delta_cov >= self.cfg.occlusion_change_coverage_min
        )

        if change_detected:
            self._item_change_frames += 1
            self._item_stable_frames = 0
        else:
            self._item_change_frames = 0
            self._item_stable_frames += 1
            self._item_last_stable = (ctx.frame_index, base_image, ctx.metrics)

        if self._item_cooldown > 0:
            self._item_cooldown -= 1

        if (
            not self._item_active
            and self._item_cooldown == 0
            and self._item_change_frames >= self.cfg.occlusion_change_confirm_frames
        ):
            self._item_active = True
            self._item_event_id += 1
            occ_id = self._item_event_id

            occ_filename = f"KF-OCLUSION-ITEM-{occ_id:03d}_01_{ctx.frame_index:06d}"
            events.append(
                ctx.emit_event(
                    "KF-OCCLUSION-ITEM",
                    ctx.frame_index,
                    base_image,
                    ctx.metrics,
                    ctx.detections,
                    folder_tag="KF-OCLUSION-ITEM",
                    filename_override=occ_filename,
                )
            )

            pre_snap = self._item_last_stable
            if pre_snap is not None:
                snap_idx, snap_img, snap_metrics = pre_snap
                pre_filename = f"KF-OCLUSION-ITEM-{occ_id:03d}_00_{snap_idx:06d}"
                events.append(
                    ctx.emit_event(
                        "KF-OCCLUSION-ITEM",
                        snap_idx,
                        snap_img,
                        snap_metrics,
                        ctx.detections,
                        folder_tag="KF-OCLUSION-ITEM",
                        filename_override=pre_filename,
                    )
                )

            self._item_cooldown = self.cfg.occlusion_change_cooldown_frames
            self._item_stable_frames = 0

        if self._item_active and self._item_stable_frames >= self.cfg.occlusion_change_confirm_frames:
            occ_id = self._item_event_id
            post_filename = f"KF-OCLUSION-ITEM-{occ_id:03d}_02_{ctx.frame_index:06d}"
            events.append(
                ctx.emit_event(
                    "KF-OCCLUSION-ITEM",
                    ctx.frame_index,
                    base_image,
                    ctx.metrics,
                    ctx.detections,
                    folder_tag="KF-OCLUSION-ITEM",
                    filename_override=post_filename,
                )
            )
            self._item_active = False

        self._item_prev_count = count
        self._item_prev_coverage = coverage
