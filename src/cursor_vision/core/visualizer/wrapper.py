from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .detection import Detection
from .display_overlay import DisplayOverlay
from .label_renderer import LabelRenderer, LabelRendererConfig

MODERN_CLASS_PALETTE: tuple[tuple[int, int, int], ...] = (
    (70, 70, 110),    # Azul grisáceo
    (90, 35, 35),     # Rojo vino
    (30, 85, 60),     # Verde bosque
    (85, 70, 25),     # Mostaza oscura
    (65, 45, 95),     # Púrpura profundo
    (25, 90, 95),     # Cian oscuro
    (95, 60, 25),     # Naranja quemado
    (95, 30, 65),     # Magenta vino
    (45, 95, 30),     # Verde lima oscuro
    (35, 35, 38),     # Negro grafito
    (110, 60, 60),    # Rojo polvo
    (25, 70, 95),     # Azul petróleo
)


class VisualizerWrapper:
    """Orchestrate label rendering and HUD overlay rendering."""

    def __init__(
        self,
        *,
        label_renderer: LabelRenderer | None = None,
        overlay: DisplayOverlay | None = None,
        use_modern_palette: bool = True,
        class_palette: Sequence[tuple[int, int, int]] | None = None,
        class_color_map: Mapping[int | str, tuple[int, int, int]] | None = None,
        palette_offset: int = 0,
        palette_cycle: bool = False,
    ) -> None:
        if label_renderer is None:
            cfg = LabelRendererConfig()
            if class_palette is None and use_modern_palette:
                class_palette = MODERN_CLASS_PALETTE
            if class_palette is not None:
                cfg.class_color_palette = tuple(class_palette)
                cfg.palette_offset = palette_offset
                cfg.palette_cycle = palette_cycle
            if class_color_map:
                cfg.class_color_map = dict(class_color_map)
            label_renderer = LabelRenderer(cfg)
        self.label_renderer = label_renderer
        self.overlay = overlay

    def render(
        self,
        image,
        labels: Iterable[Detection] | Iterable[Mapping[str, Any]],
        display_data: Mapping[str, Any] | None = None,
        *,
        blink_on: bool | None = None,
    ):
        output = image.copy()
        if labels is not None:
            output = self.label_renderer.render(output, labels, copy=False)
        if self.overlay is not None and display_data is not None:
            output = self.overlay.draw(output, display_data, blink_on=blink_on, copy=False)
        return output
