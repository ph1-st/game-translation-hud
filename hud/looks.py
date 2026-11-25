"""The sizes and colours of the game's text, shared by every cover.

A glyph height measured on one reading is a pixel or so off, and labels the game draws at
the same size must still get exactly the same translation size. Every measured height is
snapped to a size the game is already known to use when it is within a pixel (or 7 %) of
it; otherwise it becomes a new size (:class:`Looks`).

Colours work the same way. The translation is drawn in the colours of the Japanese it
replaces (its fill, and its outline when the glyphs have one), sampled from the strokes; a
sample is a little off depending on the background and the size of the glyphs, so a sampled
pair is snapped to a pair the game is already known to use when it is close
(:class:`Palette`).

Both tables are persisted with the cover cache (heights relative to the frame height), so a
size or a colour, once seen, never changes.
"""

from __future__ import annotations

import math
import threading

Color = tuple[int, int, int]  # RGB
Colours = tuple[Color, Color | None]  # (fill, outline); outline None for plain text
PLAIN_LIGHT: Color = (240, 240, 240)  # when no colour can be sampled
PLAIN_DARK: Color = (40, 42, 46)

TOLERANCE_PX = 1.0  # a measured height this close to a known size is that size...
TOLERANCE_REL = 0.07  # ...or this close, relatively
FILL_REACH = 40.0  # RGB distance within which two sampled fills are one colour
EDGE_REACH = 70.0  # ...and two outlines (dark, thin and blended with the background)


def is_light(c: Color) -> bool:
    """Light rather than dark (by luma)."""
    r, g, b = c
    return 0.299 * r + 0.587 * g + 0.114 * b >= 128


class Looks:
    """The game's text sizes (glyph heights relative to the frame height)."""

    def __init__(self, sizes: list[float] | None = None) -> None:
        self.sizes: list[float] = sorted(sizes or [])
        self.dirty = False
        self._lock = threading.Lock()

    def snap(self, glyph_px: float, frame_h: int) -> float:
        """The game's text size (pixels) that this measured glyph height is."""
        with self._lock:
            best = None
            for rel in self.sizes:
                px = rel * frame_h
                d = abs(px - glyph_px)
                if d <= max(TOLERANCE_PX, TOLERANCE_REL * glyph_px) and (
                    best is None or d < best[0]
                ):
                    best = (d, px)
            if best is not None:
                return best[1]
            self.sizes.append(glyph_px / frame_h)
            self.sizes.sort()
            self.dirty = True
            return float(glyph_px)

    def to_list(self) -> list[float]:
        with self._lock:
            return [round(s, 6) for s in self.sizes]


def colour_distance(a: Color, b: Color) -> float:
    return math.sqrt(sum((int(x) - int(y)) ** 2 for x, y in zip(a, b, strict=True)))


def same_colour(a: Color, b: Color) -> bool:
    """Two sampled fills that are one colour of the game's text."""
    return colour_distance(a, b) <= FILL_REACH


class Palette:
    """The text colours the game uses: (fill, outline) pairs."""

    def __init__(self, pairs: list | None = None) -> None:
        self.pairs: list[Colours] = []
        for fill, edge in pairs or []:
            self.pairs.append((tuple(fill), tuple(edge) if edge is not None else None))
        self.dirty = False
        self._lock = threading.Lock()

    def snap(self, fill: Color, edge: Color | None) -> Colours:
        """The known pair this sample is, or the sample as a new pair."""
        with self._lock:
            best = None
            for f, e in self.pairs:
                if (e is None) != (edge is None):
                    continue
                d = colour_distance(f, fill) / FILL_REACH
                if e is not None and edge is not None:
                    d = max(d, colour_distance(e, edge) / EDGE_REACH)
                if d <= 1.0 and (best is None or d < best[0]):
                    best = (d, (f, e))
            if best is not None:
                return best[1]
            pair = (
                tuple(int(v) for v in fill),
                tuple(int(v) for v in edge) if edge is not None else None,
            )
            self.pairs.append(pair)
            self.dirty = True
            return pair

    def to_list(self) -> list:
        with self._lock:
            return [[list(f), list(e) if e is not None else None] for f, e in self.pairs]
