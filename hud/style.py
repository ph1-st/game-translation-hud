"""One look for every cover.

The translation is drawn in the colours of the Japanese it replaces (sampled per line, see
:mod:`hud.extent`), on the background the Japanese was drawn on: the Japanese strokes are
painted over with what surrounds them (:mod:`hud.backdrop`). Under a label lies a faint,
soft-edged glow, the same for every cover: one constant shine colour and strength, chosen
once at start-up, so a screen full of translations reads as one layer, not a patchwork of
boxes. Dialogue lies on frosted glass instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from hud.looks import Color


@dataclass(frozen=True)
class CoverStyle:
    name: str
    shine: Color  # the glow's colour (also the light line along frosted glass)
    label: float  # its strength under a UI label (0-1: the share of the shine colour)


STYLES: dict[str, CoverStyle] = {
    "glow": CoverStyle("glow", (255, 255, 255), 0.16),  # a faint white shine
    "shade": CoverStyle("shade", (14, 16, 20), 0.30),  # a faint dark one
}
DEFAULT_STYLE = "glow"


def get_style(name: str) -> CoverStyle:
    try:
        return STYLES[name]
    except KeyError as exc:
        raise ValueError(f"unknown cover style {name!r}; choose from {sorted(STYLES)}") from exc
