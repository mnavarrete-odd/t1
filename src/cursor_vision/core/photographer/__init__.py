from .config import PhotographerConfig
from .photographer import Photographer
from .routing import build_keyframe_signal
from .types import DetectionData, FrameMetrics, KeyframeEvent, KeyframeSaveRequest, KeyframeSignal

__all__ = [
    "Photographer",
    "build_keyframe_signal",
    "PhotographerConfig",
    "DetectionData",
    "FrameMetrics",
    "KeyframeEvent",
    "KeyframeSaveRequest",
    "KeyframeSignal",
    "KeyframeWriter",
]


def __getattr__(name: str):
    if name == "KeyframeWriter":
        from .writer import KeyframeWriter

        return KeyframeWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
