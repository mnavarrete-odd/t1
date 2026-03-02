from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import cv2

from .detection import Detection, coerce_detections


@dataclass
class LabelRendererConfig:
    """Static visual configuration for LabelRenderer."""

    base_thickness: int = 1
    corner_thickness: int = 5
    corner_ratio: float = 0.18
    corner_min: int = 8
    tick_ratio: float = 0.08
    tick_min: int = 6
    show_ticks: bool = True

    font_scale: float = 0.6
    font_thickness: int = 1
    meta_scale_delta: float = 0.08
    meta_min_scale: float = 0.45

    show_class_name: bool = True
    show_confidence: bool = True
    show_tracking_id: bool = True
    confidence_as_percent: bool = True
    confidence_fmt: str = "{:.2f}"
    label_separator: str = " | "
    uppercase_class: bool = True

    use_class_colors: bool = True
    class_color_map: Mapping[int | str, tuple[int, int, int]] = field(default_factory=dict)
    class_color_palette: Sequence[tuple[int, int, int]] | None = None
    palette_offset: int = 0
    palette_cycle: bool = False
    default_color: tuple[int, int, int] = (0, 255, 0)

    reference_size: tuple[int, int] = (1920, 1080)
    scale_mode: str = "mean"
    min_scale: float = 0.45
    max_scale: float = 1.6
    scale_multiplier: float = 1.1

    label_bg_color: tuple[int, int, int] = (10, 10, 10)
    label_bg_alpha: float = 0.7
    label_border_thickness: int = 1
    label_text_color: tuple[int, int, int] = (255, 255, 255)
    label_meta_color: tuple[int, int, int] = (200, 200, 200)
    label_pad_x: int = 7
    label_pad_y: int = 5
    label_accent_w: int = 4
    label_margin: int = 4
    max_text_width: int | None = None


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    # Golden-ratio hue stepping for better separation and modern tones.
    hue = (class_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.68, 0.92)
    return int(b * 255), int(g * 255), int(r * 255)


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


def _resolve_color(det: Detection, cfg: LabelRendererConfig) -> tuple[int, int, int]:
    if not cfg.use_class_colors:
        return cfg.default_color

    if cfg.class_color_map:
        if det.class_id in cfg.class_color_map:
            return cfg.class_color_map[det.class_id]
        if det.class_name in cfg.class_color_map:
            return cfg.class_color_map[det.class_name]

    palette = cfg.class_color_palette
    if palette:
        idx = det.class_id + cfg.palette_offset
        if cfg.palette_cycle:
            idx %= len(palette)
            return palette[idx]
        if 0 <= idx < len(palette):
            return palette[idx]

    return _color_for_class(det.class_id)


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


def _truncate_text(text: str, max_w: int, font_scale: float, thickness: int) -> str:
    if not text or max_w <= 0:
        return ""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if tw <= max_w:
        return text
    base = text
    while base:
        candidate = f"{base}.."
        (cw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if cw <= max_w:
            return candidate
        base = base[:-1]
    return ""


def _clip_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if width <= 0 or height <= 0:
        return None
    x1c = max(0, min(width - 1, int(round(x1))))
    y1c = max(0, min(height - 1, int(round(y1))))
    x2c = max(0, min(width - 1, int(round(x2))))
    y2c = max(0, min(height - 1, int(round(y2))))
    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c


class LabelRenderer:
    """Draw bounding boxes and labels for detections."""

    def __init__(self, config: LabelRendererConfig | None = None) -> None:
        self.config = config or LabelRendererConfig()

    def render(
        self,
        image,
        labels: Iterable[Detection] | Iterable[dict],
        *,
        copy: bool = True,
    ):
        cfg = self.config
        output = image.copy() if copy else image
        h, w = output.shape[:2]
        scale = _compute_scale(
            w,
            h,
            cfg.reference_size,
            cfg.scale_mode,
            cfg.min_scale,
            cfg.max_scale,
            cfg.scale_multiplier,
        )

        base_thickness = max(1, int(round(cfg.base_thickness * scale)))
        corner_thickness = max(1, int(round(cfg.corner_thickness)))
        corner_min = max(1, int(round(cfg.corner_min * scale)))
        tick_min = max(1, int(round(cfg.tick_min * scale)))
        font_scale = cfg.font_scale * scale
        font_thickness = max(1, int(round(cfg.font_thickness * scale)))
        meta_scale_delta = cfg.meta_scale_delta * scale
        meta_min_scale = cfg.meta_min_scale * scale
        label_border_thickness = max(1, int(round(cfg.label_border_thickness * scale)))
        label_pad_x = max(1, int(round(cfg.label_pad_x * scale)))
        label_pad_y = max(1, int(round(cfg.label_pad_y * scale)))
        label_accent_w = max(1, int(round(cfg.label_accent_w * scale)))
        label_margin = max(1, int(round(cfg.label_margin * scale)))
        max_text_width = None
        if cfg.max_text_width is not None:
            max_text_width = max(10, int(round(cfg.max_text_width * scale)))
        detections = coerce_detections(labels)

        for det in detections:
            x1, y1, x2, y2 = det.to_xyxy()
            clipped = _clip_bbox(x1, y1, x2, y2, w, h)
            if clipped is None:
                continue
            x1i, y1i, x2i, y2i = clipped

            color = _resolve_color(det, cfg)
            w_box = max(1, x2i - x1i)
            h_box = max(1, y2i - y1i)

            # Base thin frame
            cv2.rectangle(output, (x1i, y1i), (x2i, y2i), color, base_thickness)

            # Corner brackets (thick)
            corner_len = max(corner_min, int(min(w_box, h_box) * cfg.corner_ratio))
            corner_th = max(1, corner_thickness)
            # top-left
            cv2.line(output, (x1i, y1i), (x1i + corner_len, y1i), color, corner_th)
            cv2.line(output, (x1i, y1i), (x1i, y1i + corner_len), color, corner_th)
            # top-right
            cv2.line(output, (x2i, y1i), (x2i - corner_len, y1i), color, corner_th)
            cv2.line(output, (x2i, y1i), (x2i, y1i + corner_len), color, corner_th)
            # bottom-left
            cv2.line(output, (x1i, y2i), (x1i + corner_len, y2i), color, corner_th)
            cv2.line(output, (x1i, y2i), (x1i, y2i - corner_len), color, corner_th)
            # bottom-right
            cv2.line(output, (x2i, y2i), (x2i - corner_len, y2i), color, corner_th)
            cv2.line(output, (x2i, y2i), (x2i, y2i - corner_len), color, corner_th)

            # Mid-edge ticks
            if cfg.show_ticks:
                tick_len = max(tick_min, int(min(w_box, h_box) * cfg.tick_ratio))
                cv2.line(
                    output,
                    (x1i + w_box // 2 - tick_len, y1i),
                    (x1i + w_box // 2 + tick_len, y1i),
                    color,
                    max(1, int(round(2 * scale))),
                )
                cv2.line(
                    output,
                    (x1i + w_box // 2 - tick_len, y2i),
                    (x1i + w_box // 2 + tick_len, y2i),
                    color,
                    max(1, int(round(2 * scale))),
                )

            # Label chip inside box (top-left)
            name = str(det.class_name) if cfg.show_class_name else ""
            if cfg.uppercase_class and name:
                name = name.upper()

            conf_text = ""
            if cfg.show_confidence:
                if cfg.confidence_as_percent:
                    pct = int(round(det.confidence * 100))
                    conf_text = f"{pct}%"
                else:
                    try:
                        conf_text = cfg.confidence_fmt.format(det.confidence)
                    except (ValueError, KeyError):
                        conf_text = f"{det.confidence:.2f}"

            meta_parts: list[str] = []
            if cfg.show_tracking_id and det.tracking_id is not None:
                meta_parts.append(f"ID {det.tracking_id}")
            if conf_text:
                meta_parts.append(conf_text)
            occ_val = None
            if det.extra_data:
                occ_val = det.extra_data.get("occ")
            if occ_val is not None:
                try:
                    occ_f = float(occ_val)
                    if 0.0 <= occ_f <= 1.0:
                        occ_text = f"OCC {occ_f * 100:.0f}%"
                    else:
                        occ_text = f"OCC {occ_f:.0f}%"
                    meta_parts.append(occ_text)
                except (TypeError, ValueError):
                    pass
            dist_val = None
            if det.extra_data:
                dist_val = det.extra_data.get("dist_px")
                if dist_val is None:
                    dist_val = det.extra_data.get("dist")
            if dist_val is not None:
                try:
                    dist_f = float(dist_val)
                    meta_parts.append(f"DIST {dist_f:.0f}px")
                except (TypeError, ValueError):
                    pass
            custom_meta = None
            if det.extra_data:
                custom_meta = det.extra_data.get("meta")
            if custom_meta is not None:
                text = str(custom_meta).strip()
                if text:
                    meta_parts.append(text)
            meta = cfg.label_separator.join(meta_parts)

            if not name and meta:
                name, meta = meta, ""

            if not name:
                continue

            pad_x, pad_y = label_pad_x, label_pad_y
            accent_w = label_accent_w
            margin = label_margin
            font_main = font_scale
            font_meta = max(meta_min_scale, font_scale - meta_scale_delta)

            max_label_w = max(10, w_box - margin * 2)
            max_text_w = max(10, max_label_w - (accent_w + pad_x * 2))
            if max_text_width is not None:
                max_text_w = min(max_text_w, max_text_width)

            name = _truncate_text(name, max_text_w, font_main, font_thickness)
            meta = _truncate_text(meta, max_text_w, font_meta, font_thickness)

            (tw1, th1), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_main, font_thickness)
            (tw2, th2), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_SIMPLEX, font_meta, font_thickness)
            label_h = th1 + th2 + pad_y * 3
            label_w = max(tw1, tw2) + pad_x * 2 + accent_w

            # If the box is too tight, drop the meta line.
            if label_h > h_box - margin * 2:
                meta = ""
                (tw2, th2) = (0, 0)
                label_h = th1 + pad_y * 2
                label_w = max(tw1, 0) + pad_x * 2 + accent_w

            bx1 = x1i + margin
            by1 = y1i + margin
            bx2 = min(bx1 + label_w, x2i - margin)
            by2 = min(by1 + label_h, y2i - margin)

            if bx2 > bx1 and by2 > by1:
                _alpha_rect(output, (bx1, by1), (bx2, by2), cfg.label_bg_color, cfg.label_bg_alpha)
                cv2.rectangle(output, (bx1, by1), (bx2, by2), color, label_border_thickness)
                cv2.rectangle(output, (bx1, by1), (bx1 + accent_w, by2), color, -1)

                text_y = by1 + pad_y + th1
                cv2.putText(
                    output,
                    name,
                    (bx1 + accent_w + pad_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_main,
                    cfg.label_text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
                if meta:
                    meta_y = text_y + th2 + pad_y
                    cv2.putText(
                        output,
                        meta,
                        (bx1 + accent_w + pad_x, meta_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_meta,
                        cfg.label_meta_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )
        return output
