"""A dialogue line typing out: how much of its painted text shows, line after line, left to
right, with a soft front (the share comes from :func:`hud.items.reveal_target`)."""

from __future__ import annotations

import numpy as np

from hud.items import OverlayItem
from hud.overlay.paint import Layer, text_lines

REVEAL_EDGE = 0.6  # the front of typing text is soft over this many font sizes
MIN_FONT = 6


def reveal_mask(
    shape: tuple[int, int],
    origin: tuple[int, int],
    lines: list[tuple[float, float, float, float]],
    share: float,
    edge: float,
) -> np.ndarray:
    """How much of each pixel (0-1) of an image of *shape* at *origin* shows when *share*
    of the text has typed out: line after line, left to right, a soft front."""
    h, w = shape
    ox, oy = origin
    edge = max(1.0, edge)
    xs = np.arange(w, dtype=np.float32) + ox + 0.5
    ys = np.arange(h, dtype=np.float32) + oy + 0.5
    lengths = [(r - left) + 2.0 * edge for left, r, _t, _b in lines]
    done = share * sum(lengths)
    mask = np.zeros((h, w), np.float32)
    for i, (left, _r, top, bottom) in enumerate(lines):
        before = sum(lengths[:i])
        front = left - edge + min(max(done - before, 0.0), lengths[i])
        cols = np.clip((front - xs) / edge, 0.0, 1.0)
        # the rows of this line: halfway to the lines above and below
        y0 = -1e9 if i == 0 else (lines[i - 1][3] + top) / 2.0
        y1 = 1e9 if i == len(lines) - 1 else (bottom + lines[i + 1][2]) / 2.0
        mask[(ys >= y0) & (ys < y1), :] = cols
    return mask


def revealed(full: Layer | None, item: OverlayItem, share: float) -> Layer | None:
    """The text image *full* (x, y, premultiplied RGBA) with *share* of it typed out."""
    if full is None or share >= 1.0:
        return full
    x, y, rgba = full
    edge = REVEAL_EDGE * max(MIN_FONT, item.font_px)
    m = reveal_mask(rgba.shape[:2], (x, y), text_lines(item), share, edge)
    return x, y, (rgba.astype(np.float32) * m[:, :, None] + 0.5).astype(np.uint8)
