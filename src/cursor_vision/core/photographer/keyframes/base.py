from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import FrameContext
    from ..types import KeyframeEvent


class KeyframeHandler:
    def update(self, ctx: "FrameContext") -> list["KeyframeEvent"]:
        return []
