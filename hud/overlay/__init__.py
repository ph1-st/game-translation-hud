"""The overlay: the window that shows the HUD core's items over the game, and the painting
it shares with the recorder and offline previews.

* :mod:`hud.overlay.window`: the click-through, always-on-top window
* :mod:`hud.overlay.paint`: backdrops and text, painted the same way everywhere
* :mod:`hud.overlay.reveal`: a dialogue line typing out
* :mod:`hud.overlay.builder`: backdrops built off the GUI thread

The items themselves are Qt-free (:mod:`hud.items`), so the core never imports this.
"""

from hud.items import OverlayItem, reveal_share, reveal_target
from hud.overlay.builder import BackdropBuilder
from hud.overlay.paint import (
    build_backdrop,
    build_layers,
    paint_all,
    paint_text,
    qimage,
    render_text,
    text_lines,
)
from hud.overlay.reveal import reveal_mask, revealed
from hud.overlay.window import FADE_IN_S, OverlayWindow

__all__ = [
    "FADE_IN_S",
    "BackdropBuilder",
    "OverlayItem",
    "OverlayWindow",
    "build_backdrop",
    "build_layers",
    "paint_all",
    "paint_text",
    "qimage",
    "render_text",
    "reveal_mask",
    "reveal_share",
    "reveal_target",
    "revealed",
    "text_lines",
]
