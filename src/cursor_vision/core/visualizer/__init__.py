"""Reusable visualizer wrapper for YOLO-style pipelines."""

from .detection import BBoxFormat, Detection, coerce_detection, coerce_detections
from .display_overlay import DisplayOverlay, OverlayConfig, OverlayPosition
from .label_renderer import LabelRenderer, LabelRendererConfig
from .wrapper import MODERN_CLASS_PALETTE, VisualizerWrapper

__all__ = [
    "BBoxFormat",
    "Detection",
    "coerce_detection",
    "coerce_detections",
    "DisplayOverlay",
    "OverlayConfig",
    "OverlayPosition",
    "LabelRenderer",
    "LabelRendererConfig",
    "MODERN_CLASS_PALETTE",
    "VisualizerWrapper",
]
