from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, Literal

import cv2


OverlayPosition = Literal["top-left", "top-right", "bottom-left", "bottom-right"]


@dataclass
class OverlayConfig:
    """Static visual configuration for DisplayOverlay."""

    title: str = "ODDIVISION"
    author: str | None = None
    live: bool = True
    blink: bool = True
    live_text: str = "LIVE"
    position: OverlayPosition = "top-left"
    fields: Sequence[str] = field(default_factory=lambda: ["count", "primary", "fps", "status"])
    field_labels: Mapping[str, str] = field(default_factory=dict)
    value_formatters: Mapping[str, Callable[[Any], str]] = field(default_factory=dict)
    width: int | None = 360
    padding: int = 12
    margin: int = 12
    accent_width: int = 5
    header_gap: int = 6
    row_gap: int = 6
    value_gap: int = 12
    label_ratio: float = 0.55
    background_color: tuple[int, int, int] = (0, 0, 0)
    background_alpha: float = 0.6
    accent_color: tuple[int, int, int] = (0, 255, 80)
    title_color: tuple[int, int, int] = (255, 255, 255)
    author_color: tuple[int, int, int] = (160, 160, 160)
    label_color: tuple[int, int, int] = (200, 200, 200)
    value_color: tuple[int, int, int] = (255, 255, 255)
    live_color: tuple[int, int, int] = (0, 255, 80)
    live_off_color: tuple[int, int, int] = (0, 80, 20)
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    title_scale: float = 0.72
    author_scale: float = 0.5
    label_scale: float = 0.5
    value_scale: float = 0.6
    font_thickness: int = 1
    empty_value: str = "--"
    dot_radius: int = 5
    live_gap: int = 10
    reference_size: tuple[int, int] = (1920, 1080)
    scale_mode: str = "mean"
    min_scale: float = 0.6
    max_scale: float = 1.6
    scale_multiplier: float = 1.6


def _alpha_rect(
    img,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _text_size(text: str, font: int, scale: float, thickness: int) -> tuple[int, int]:
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return int(tw), int(th)


def _truncate_text(text: str, max_w: int, font: int, scale: float, thickness: int) -> str:
    if not text or max_w <= 0:
        return ""
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw <= max_w:
        return text
    base = text
    while base:
        candidate = f"{base}.."
        (cw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if cw <= max_w:
            return candidate
        base = base[:-1]
    return ""


def _compute_scale(
    image_w: int,
    image_h: int,
    reference_size: tuple[int, int],
    scale_mode: str,
    min_scale: float,
    max_scale: float,
    scale_multiplier: float,
) -> float:
    ref_w, ref_h = reference_size
    if ref_w <= 0 or ref_h <= 0:
        return 1.0
    ratio_w = image_w / ref_w
    ratio_h = image_h / ref_h
    mode = scale_mode.lower() if scale_mode else "min"
    if mode == "max":
        scale = max(ratio_w, ratio_h)
    elif mode in ("mean", "avg", "average"):
        scale = (ratio_w + ratio_h) / 2.0
    elif mode in ("geom", "geometric"):
        scale = (ratio_w * ratio_h) ** 0.5
    else:
        scale = min(ratio_w, ratio_h)
    scale *= scale_multiplier
    return max(min_scale, min(max_scale, scale))


def _anchor_position(
    position: OverlayPosition,
    image_w: int,
    image_h: int,
    box_w: int,
    box_h: int,
    margin: int,
) -> tuple[int, int]:
    if position == "top-left":
        return margin, margin
    if position == "top-right":
        return max(margin, image_w - box_w - margin), margin
    if position == "bottom-left":
        return margin, max(margin, image_h - box_h - margin)
    if position == "bottom-right":
        return max(margin, image_w - box_w - margin), max(margin, image_h - box_h - margin)
    return margin, margin


class DisplayOverlay:
    """Draw a configurable HUD overlay from external data."""

    def __init__(self, config: OverlayConfig | None = None) -> None:
        self.config = config or OverlayConfig()

    def draw(
        self,
        image,
        data: Mapping[str, Any],
        *,
        blink_on: bool | None = None,
        copy: bool = True,
    ):
        cfg = self.config
        output = image.copy() if copy else image
        img_h, img_w = output.shape[:2]
        scale = _compute_scale(
            img_w,
            img_h,
            cfg.reference_size,
            cfg.scale_mode,
            cfg.min_scale,
            cfg.max_scale,
            cfg.scale_multiplier,
        )

        padding = max(1, int(round(cfg.padding * scale)))
        margin = max(1, int(round(cfg.margin * scale)))
        accent_width = max(1, int(round(cfg.accent_width * scale)))
        header_gap = max(1, int(round(cfg.header_gap * scale)))
        row_gap = max(1, int(round(cfg.row_gap * scale)))
        value_gap = max(1, int(round(cfg.value_gap * scale)))
        dot_radius = max(1, int(round(cfg.dot_radius * scale)))
        live_gap = max(1, int(round(cfg.live_gap * scale)))
        font_thickness = max(1, int(round(cfg.font_thickness * scale)))
        title_scale = cfg.title_scale * scale
        author_scale = cfg.author_scale * scale
        label_scale = cfg.label_scale * scale
        value_scale = cfg.value_scale * scale

        fields = list(cfg.fields)
        rows = len(fields)

        title_h = _text_size(cfg.title, cfg.font, title_scale, font_thickness)[1] if cfg.title else 0
        live_h = _text_size(cfg.live_text, cfg.font, label_scale, font_thickness)[1] if cfg.live else 0
        header_h = max(title_h, live_h)
        author_h = _text_size(cfg.author, cfg.font, author_scale, font_thickness)[1] if cfg.author else 0
        row_h = _text_size("Ag", cfg.font, value_scale, font_thickness)[1]

        height = padding * 2
        if header_h:
            height += header_h
        if header_h and (author_h or rows):
            height += header_gap
        if author_h:
            height += author_h
        if author_h and rows:
            height += header_gap
        if rows:
            height += rows * row_h + max(0, rows - 1) * row_gap

        box_w = cfg.width
        if box_w is None:
            content_w = 0
            if cfg.title:
                title_w, _ = _text_size(cfg.title, cfg.font, title_scale, font_thickness)
                if cfg.live:
                    live_w, _ = _text_size(cfg.live_text, cfg.font, label_scale, font_thickness)
                    title_w += live_gap + live_w + dot_radius * 2
                content_w = max(content_w, title_w)
            if cfg.author:
                author_w, _ = _text_size(cfg.author, cfg.font, author_scale, font_thickness)
                content_w = max(content_w, author_w)
            for field in fields:
                label = cfg.field_labels.get(field, field).upper()
                value = self._format_value(field, data.get(field))
                label_w, _ = _text_size(label, cfg.font, label_scale, font_thickness)
                value_w, _ = _text_size(value, cfg.font, value_scale, font_thickness)
                content_w = max(content_w, label_w + value_gap + value_w)
            box_w = accent_width + padding * 2 + content_w
        else:
            box_w = int(round(box_w * scale))

        box_w = int(box_w)
        box_h = int(height)

        x, y = _anchor_position(cfg.position, img_w, img_h, box_w, box_h, margin)
        x2 = min(img_w - 1, x + box_w)
        y2 = min(img_h - 1, y + box_h)

        if x2 <= x or y2 <= y:
            return output

        _alpha_rect(output, (x, y), (x2, y2), cfg.background_color, cfg.background_alpha)
        cv2.rectangle(output, (x, y), (x + accent_width, y2), cfg.accent_color, -1)

        cursor_y = y + padding
        if header_h:
            header_baseline = cursor_y + header_h
            if cfg.title:
                cv2.putText(
                    output,
                    cfg.title,
                    (x + accent_width + padding, header_baseline),
                    cfg.font,
                    title_scale,
                    cfg.title_color,
                    font_thickness + 1,
                    cv2.LINE_AA,
                )
            if cfg.live:
                live_text = cfg.live_text
                live_w, live_h = _text_size(live_text, cfg.font, label_scale, font_thickness)
                live_x = x + box_w - padding - live_w
                live_y = header_baseline
                cv2.putText(
                    output,
                    live_text,
                    (live_x, live_y),
                    cfg.font,
                    label_scale,
                    cfg.accent_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                dot_on = True if not cfg.blink else (blink_on if blink_on is not None else True)
                dot_color = cfg.live_color if dot_on else cfg.live_off_color
                dot_x = live_x - live_gap
                dot_y = live_y - live_h // 2
                cv2.circle(output, (dot_x, dot_y), dot_radius, dot_color, -1)
            cursor_y = header_baseline + header_gap

        if author_h:
            author_baseline = cursor_y + author_h
            cv2.putText(
                output,
                cfg.author,
                (x + accent_width + padding, author_baseline),
                cfg.font,
                author_scale,
                cfg.author_color,
                font_thickness,
                cv2.LINE_AA,
            )
            cursor_y = author_baseline + header_gap

        if rows:
            content_w = box_w - accent_width - padding * 2
            label_max_w = int(content_w * cfg.label_ratio)
            value_max_w = max(0, content_w - label_max_w - value_gap)

            for idx, field in enumerate(fields):
                label = cfg.field_labels.get(field, field).upper()
                value = self._format_value(field, data.get(field))

                label = _truncate_text(label, label_max_w, cfg.font, label_scale, font_thickness)
                value = _truncate_text(value, value_max_w, cfg.font, value_scale, font_thickness)

                label_w, label_h = _text_size(label, cfg.font, label_scale, font_thickness)
                value_w, value_h = _text_size(value, cfg.font, value_scale, font_thickness)

                row_baseline = cursor_y + row_h
                label_x = x + accent_width + padding
                value_x = x + box_w - padding - value_w

                cv2.putText(
                    output,
                    label,
                    (label_x, row_baseline),
                    cfg.font,
                    label_scale,
                    cfg.label_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    value,
                    (value_x, row_baseline),
                    cfg.font,
                    value_scale,
                    cfg.value_color,
                    font_thickness + 1,
                    cv2.LINE_AA,
                )

                if idx < rows - 1:
                    cursor_y += row_h + row_gap
        return output

    def _format_value(self, field: str, value: Any) -> str:
        if field in self.config.value_formatters:
            return self.config.value_formatters[field](value)
        if value is None:
            return self.config.empty_value
        return str(value)
