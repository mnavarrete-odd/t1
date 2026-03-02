from pathlib import Path
from typing import List

import cv2
import numpy as np

from .types import FrameData, FrameResult, PreparedDetection
from core.visualizer import (
    Detection,
    DisplayOverlay,
    LabelRenderer,
    LabelRendererConfig,
    OverlayConfig,
    VisualizerWrapper,
)

STATUS_MATCHED = "matched"
STATUS_NEW = "new"
STATUS_SUPPRESSED = "suppressed"
STATUS_NONE = "none"

STATUS_CLASS_ID = {
    STATUS_MATCHED: 1000,
    STATUS_NEW: 1001,
    STATUS_SUPPRESSED: 1003,
    STATUS_NONE: 1002,
}

STATUS_COLOR_MAP = {
    STATUS_CLASS_ID[STATUS_MATCHED]: (0, 200, 0),
    STATUS_CLASS_ID[STATUS_NEW]: (0, 0, 220),
    STATUS_CLASS_ID[STATUS_SUPPRESSED]: (0, 215, 255),
    STATUS_CLASS_ID[STATUS_NONE]: (128, 128, 128),
}

_PREV_FRAME_DETECTIONS: list[PreparedDetection] = []
_PREV_FRAME_INDEX: int | None = None
_NEW_REF_MAX_ITEMS = 4
_NEW_REF_MAX_CENTER_NORM = 0.30


def _assignment_map(result: FrameResult):
    return {a.det_idx: a for a in result.assignments}


def _build_visualizer() -> VisualizerWrapper:
    label_cfg = LabelRendererConfig(
        show_confidence=False,
        show_class_name=False,
        font_scale=0.46,
        meta_min_scale=0.34,
        meta_scale_delta=0.16,
        label_pad_x=5,
        label_pad_y=3,
        uppercase_class=False,
        class_color_map=STATUS_COLOR_MAP,
    )
    overlay_cfg = OverlayConfig(
        title="COUNTER",
        fields=(
            "frame",
            "count_total",
            "new",
            "new_refs",
            "active_tracks",
        ),
        field_labels={
            "frame": "Frame",
            "count_total": "Conteo total",
            "new": "New",
            "new_refs": "New refs",
            "active_tracks": "Active tracks",
        },
        value_formatters={
            "frame": lambda v: "--" if v is None else str(int(v)),
            "count_total": lambda v: "--" if v is None else f"{float(v):.0f}",
            "new": lambda v: "--" if v is None else str(int(v)),
            "new_refs": lambda v: "--" if not v else str(v),
            "active_tracks": lambda v: "--" if v is None else str(int(v)),
        },
        width=560,
        label_ratio=0.35,
    )
    return VisualizerWrapper(
        label_renderer=LabelRenderer(label_cfg),
        overlay=DisplayOverlay(overlay_cfg),
    )


_VISUALIZER = _build_visualizer()


def _class_alias(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"cajas", "caja"}:
        return "caja"
    return n if n else "obj"


def _format_height(height_depth: float) -> str:
    if np.isfinite(height_depth):
        return f"{float(height_depth):.1f}"
    return "na"


def _infer_prev_height_for_new(
    det: PreparedDetection,
    prev_detections: List[PreparedDetection],
    image_diag: float,
) -> float:
    if image_diag <= 0.0:
        return float("nan")
    if not prev_detections:
        return float("nan")

    best_norm = float("inf")
    best_height = float("nan")
    for prev in prev_detections:
        if int(prev.class_id) != int(det.class_id):
            continue
        if not np.isfinite(prev.height_depth):
            continue

        dx = float(det.centroid[0]) - float(prev.centroid[0])
        dy = float(det.centroid[1]) - float(prev.centroid[1])
        center_norm = float(np.hypot(dx, dy) / max(image_diag, 1e-6))
        if center_norm < best_norm:
            best_norm = center_norm
            best_height = float(prev.height_depth)

    if not np.isfinite(best_norm) or best_norm > _NEW_REF_MAX_CENTER_NORM:
        return float("nan")
    return float(best_height)


def _build_feature_meta(
    det: PreparedDetection,
    assignment,
    prev_height_ref: float,
) -> str:
    now_text = _format_height(det.height_depth)
    if assignment is not None and assignment.status == STATUS_NEW:
        return f"h={now_text} prev={_format_height(prev_height_ref)}"
    return f"h={now_text}"


def _label_for_detection(
    det: PreparedDetection,
    assignment,
    *,
    prev_height_ref: float = float("nan"),
) -> tuple[str, int, int | None, dict]:
    if assignment is None:
        status = STATUS_NONE
        track_id = None
    else:
        status = assignment.status
        track_id = assignment.track_id

    class_name = _class_alias(det.class_name)
    class_id = STATUS_CLASS_ID.get(status, STATUS_CLASS_ID[STATUS_NONE])
    extra = {"meta": _build_feature_meta(det, assignment, prev_height_ref)}
    return class_name, class_id, track_id, extra


def draw_tracking_frame(
    frame: FrameData,
    detections: List[PreparedDetection],
    result: FrameResult,
    out_path: str,
):
    global _PREV_FRAME_DETECTIONS, _PREV_FRAME_INDEX

    amap = _assignment_map(result)
    labels: list[Detection] = []
    new_refs: list[str] = []

    prev_detections = (
        _PREV_FRAME_DETECTIONS
        if (_PREV_FRAME_INDEX is not None and frame.frame_index > _PREV_FRAME_INDEX)
        else []
    )
    image_diag = float(np.hypot(frame.image.shape[1], frame.image.shape[0]))

    for det_idx, det in enumerate(detections):
        assignment = amap.get(det_idx)
        prev_height_ref = float("nan")
        if assignment is not None and assignment.status == STATUS_NEW:
            prev_height_ref = _infer_prev_height_for_new(det, prev_detections, image_diag)
            if len(new_refs) < _NEW_REF_MAX_ITEMS:
                track_text = str(int(assignment.track_id)) if int(assignment.track_id) >= 0 else "?"
                new_refs.append(
                    f"id{track_text}:{_format_height(det.height_depth)}|{_format_height(prev_height_ref)}"
                )

        class_name, class_id, track_id, extra_data = _label_for_detection(
            det,
            assignment,
            prev_height_ref=prev_height_ref,
        )
        labels.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                bbox=det.bbox_xyxy,
                confidence=det.confidence,
                bbox_format="xyxy",
                tracking_id=track_id,
                extra_data=extra_data,
            )
        )

    new_refs_text = " ".join(new_refs)
    if result.num_new > len(new_refs):
        remaining = result.num_new - len(new_refs)
        new_refs_text = f"{new_refs_text} +{remaining}".strip()

    display_data = {
        "frame": frame.frame_index,
        "count_total": result.running_units,
        "new": result.num_new,
        "new_refs": new_refs_text,
        "active_tracks": result.num_active_tracks,
    }
    vis = _VISUALIZER.render(
        frame.image,
        labels,
        display_data=display_data,
        blink_on=True,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

    _PREV_FRAME_DETECTIONS = list(detections)
    _PREV_FRAME_INDEX = int(frame.frame_index)


def _resolve_colormap(name: str) -> int:
    n = (name or "").strip().lower()
    cmap = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    return cmap.get(n, cv2.COLORMAP_TURBO)


def _draw_depth_colorbar(
    vis: np.ndarray,
    *,
    lo: float,
    hi: float,
    colormap: str = "turbo",
    ticks: int = 5,
) -> None:
    if vis.size == 0:
        return

    h, w = vis.shape[:2]
    right_margin = max(8, int(round(w * 0.01)))
    bar_width = max(12, int(round(w * 0.015)))
    top_margin = max(10, int(round(h * 0.04)))
    bottom_margin = max(10, int(round(h * 0.04)))

    y1 = top_margin
    y2 = h - bottom_margin
    x2 = w - right_margin
    x1 = x2 - bar_width
    bar_h = y2 - y1
    if x1 <= 0 or bar_h <= 20:
        return

    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
    bar_norm = np.repeat(gradient, bar_width, axis=1)
    bar_vis = cv2.applyColorMap(bar_norm, _resolve_colormap(colormap))
    vis[y1:y2, x1:x2] = bar_vis
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)

    tick_count = max(0, int(ticks))
    if tick_count <= 0:
        return

    if tick_count == 1:
        tick_positions = np.array([y1 + (bar_h // 2)], dtype=np.int32)
        tick_values = np.array([(lo + hi) * 0.5], dtype=np.float32)
    else:
        ratios = np.linspace(0.0, 1.0, tick_count, dtype=np.float32)
        tick_positions = np.round(y1 + (ratios * float(bar_h - 1))).astype(np.int32)
        tick_values = hi - ((hi - lo) * ratios)

    tick_len = max(6, int(round(bar_width * 0.8)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.42
    text_thickness = 1

    for py, val in zip(tick_positions.tolist(), tick_values.tolist()):
        cv2.line(
            vis,
            (x1 - tick_len, py),
            (x1 - 1, py),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        label = f"{float(val):.2f}m"
        (tw, th), _ = cv2.getTextSize(label, font, text_scale, text_thickness)
        tx = max(2, x1 - tick_len - 6 - tw)
        ty = int(np.clip(py + (th // 2), th + 1, h - 2))
        cv2.putText(
            vis,
            label,
            (tx, ty),
            font,
            text_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (tx, ty),
            font,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )


def save_depth_frame(
    frame: FrameData,
    out_path: str,
    *,
    depth_scale: float = 0.001,
    colormap: str = "turbo",
    show_colorbar: bool = True,
    colorbar_ticks: int = 5,
    show_range_text: bool = False,
    percentile_lo: float = 2.0,
    percentile_hi: float = 98.0,
    range_override: tuple[float, float] | None = None,
) -> None:
    if frame.depth_map is None:
        return

    depth_m = frame.depth_map.astype(np.float32) * float(depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return

    values = depth_m[valid]

    lo = float("nan")
    hi = float("nan")
    if range_override is not None and len(range_override) == 2:
        try:
            lo = float(range_override[0])
            hi = float(range_override[1])
        except Exception:
            lo = float("nan")
            hi = float("nan")

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        p_lo = float(np.clip(percentile_lo, 0.0, 100.0))
        p_hi = float(np.clip(percentile_hi, 0.0, 100.0))
        if p_hi <= p_lo:
            p_lo, p_hi = 2.0, 98.0
        lo = float(np.percentile(values, p_lo))
        hi = float(np.percentile(values, p_hi))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            hi = lo + 1e-6

    norm = np.zeros_like(depth_m, dtype=np.uint8)
    norm_val = np.clip((depth_m[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm[valid] = (norm_val * 255.0).astype(np.uint8)

    vis = cv2.applyColorMap(norm, _resolve_colormap(colormap))
    vis[~valid] = 0
    if show_colorbar:
        _draw_depth_colorbar(
            vis,
            lo=lo,
            hi=hi,
            colormap=colormap,
            ticks=colorbar_ticks,
        )

    range_text = f" range={lo:.3f}-{hi:.3f}" if show_range_text else ""
    text = (
        f"depth m: min={float(np.min(values)):.3f} "
        f"p50={float(np.percentile(values, 50)):.3f} "
        f"p95={float(np.percentile(values, 95)):.3f}"
        f"{range_text}"
    )
    cv2.putText(
        vis,
        text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)


def build_video(frames_dir: str, out_video_path: str, fps: int = 4):
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_paths:
        return

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return

    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)

    writer.release()
