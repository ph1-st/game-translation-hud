"""What the HUD core hands the overlay window: items in frame pixels, free of Qt, so the
core (pipeline, boards, dialogue, ticker) runs and is tested without a window.

Styles:

* ``label``: a UI line or the speaker name: the Japanese painted over, a glow, the text
* ``panel``: a pane of frosted glass (:attr:`OverlayItem.frost`) under dialogue
* ``text``: a dialogue line on its glass; it may type out with the Japanese
  (:attr:`OverlayItem.reveal`), left to right, a soft edge in front
* ``solid``: an opaque box in the fill colour (the capture self-check)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hud.ink import Rect
from hud.looks import PLAIN_LIGHT, Color
from hud.textwidth import STRETCH

SHINE: Color = (255, 255, 255)


@dataclass
class OverlayItem:
    rect: Rect  # the box the text is laid out in
    lines: list[str] = field(default_factory=list)
    font_px: int = 16
    row_top: float = 0.0  # top of the first Japanese row the lines sit on
    row_h: float = 0.0  # glyph height of a row (also scales the glow and the erase)
    pitch: float = 0.0  # distance between rows
    align: str = "center"  # left | center | right: how the game lays the text out
    stretch: int = STRETCH
    style: str = "label"  # label | panel | text | solid
    fg: Color = PLAIN_LIGHT  # the text: the Japanese's fill colour...
    edge: Color | None = None  # ...and its outline
    erase: Rect | None = None  # the Japanese painted over
    gate: bool = False  # erase only pixels in the text's colours (a wide, moving area)
    under: Rect | None = None  # the Japanese replaced: the glow lies on it too
    glow: float = 0.0  # strength of the shine under the text (0-1)
    shine: Color = SHINE
    frost: Rect | None = None  # frosted glass over this area (style panel)
    reveal: tuple[float, float, float] | None = None  # the text types out: (share shown at
    # the start, the start (monotonic seconds), seconds from there to all of it)
    fade_in: float = 0.0  # seconds it fades in (0: the window's default)
    fade_out: float = 0.0  # seconds it fades out once it is gone (0: it goes at once)
    rtl: bool = False  # written right to left (Arabic, Hebrew)
    id: str = ""
    provisional: bool = False  # translated live: shown a little faded
    plain: bool = False  # the text drawn white with a black outline, whatever the colours
    # of the Japanese (fg and edge still say what the Japanese looks like: it is erased by
    # them)

    def identity(self) -> str:
        return self.id or f"{self.style}|{'/'.join(self.lines)}|{self.rect}"


def painted_align(item: OverlayItem) -> str:
    """How the text sits in its box: as the game lays it out, except that right-to-left
    text starts at the right where the game's starts at the left."""
    if item.rtl and item.align == "left":
        return "right"
    return item.align


def reveal_share(reveal: tuple[float, float, float], now: float) -> float:
    """How much of a typing line (0-1) shows at *now*: from where it was at its start, at
    an even pace to all of it at its end."""
    s0, t0, seconds = reveal
    if seconds <= 0:
        return 1.0
    return min(1.0, s0 + (1.0 - s0) * max(0.0, now - t0) / seconds)


def reveal_target(item: OverlayItem, now: float) -> float:
    """How much of *item*'s text shows at *now* (all of it when it does not type out)."""
    return 1.0 if item.reveal is None else reveal_share(item.reveal, now)
