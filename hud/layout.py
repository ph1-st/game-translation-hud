"""Where the translation goes: the cover, the box it is drawn in, its size and its lines.

* the cover is the Japanese being replaced, within a pixel or two (:mod:`hud.extent`)
* the translation is the Japanese size: its capitals stand in a fixed proportion to the
  glyphs, one size per look of Japanese (:mod:`hud.looks`)
* shorter than the Japanese: the cover keeps the Japanese width
* longer: the cover is extended sideways until the text fits, never made taller, with no
  space beyond what the text needs. It grows the way the game lays the text out (a
  left-aligned column to the right, a right-aligned one to the left, centred text both
  ways) and never over another text, an icon, a badge or another cover; what one side
  cannot take goes to the other
* a block of rows (a paragraph) is one cover with one text wrapped into the rows the
  Japanese has, at the Japanese row pitch; when it does not fit, the block gets wider
* only when the box cannot grow enough is the text condensed, and only when condensing is
  not enough is it made smaller, step by step: it never runs past its box, and a block
  never gets more lines than the Japanese has rows
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from hud.ink import Rect, crop_gray
from hud.looks import Color
from hud.textwidth import METRICS, STRETCH, text_width, wrap, wrap_width

PAD_X = 3  # pixels between the text and the edge of a widened cover
GAP = 2  # pixels kept clear of an obstacle
MIN_STRETCH = 62  # condensing limit
SHRINK_STEP = 0.94  # when condensing is not enough, the size goes down by this factor...
MIN_FONT_PX = 6  # ...to this at the very least (where the last line ends in an ellipsis)
ELLIPSIS = "…"
BLOCK_GAP = 0.9  # rows closer than this many glyph heights (ink gap) form a paragraph
BLOCK_ALIGN = 1.6  # ...and start within this many glyph heights of each other (indent)
BLOCK_MIN_CHARS = 8  # a paragraph row is a run of text, not a short label
BLOB_CONTRAST = 45  # grey levels from the surroundings that make a shape a UI element
BLOB_MIN_H, BLOB_MIN_W = 0.5, 0.35  # ...at least this share of a glyph tall and wide
NEAR = 1.0  # glyph heights beside a cover where an undetected badge or arrow is looked for
BASELINE_DESCENT = 0.85  # the share of the descent that counts when centring on a row


@dataclass
class Geom:
    """A placed cover: everything the overlay needs, in frame pixels."""

    cover: Rect  # the Japanese (+ PAD): what the pixel watch looks at
    box: Rect  # what is drawn: the cover, extended when the text needs it
    font_px: int
    align: str  # left | center | right: how the game lays the text out
    lines: list[str]
    row_top: float  # top of the first Japanese row's glyphs
    row_h: float  # glyph height: each line is centred on one Japanese row
    pitch: float  # distance between rows (0 for one row)
    stretch: int = STRETCH
    fg: Color | None = None  # the colours of the Japanese: the text is drawn in them
    edge: Color | None = None  # (outline; None when the glyphs have none)

    def to_rel(self, fw: int, fh: int) -> dict:
        def r(rect: Rect) -> list[float]:
            x, y, w, h = rect
            return [x / fw, y / fh, w / fw, h / fh]

        return {
            "cover": r(self.cover),
            "box": r(self.box),
            "font": self.font_px / fh,
            "align": self.align,
            "row_top": self.row_top / fh,
            "row_h": self.row_h / fh,
            "pitch": self.pitch / fh,
            "rows": len(self.lines),
            "stretch": self.stretch,
            "fg": list(self.fg) if self.fg is not None else None,
            "edge": list(self.edge) if self.edge is not None else None,
        }

    @classmethod
    def from_rel(cls, d: dict, lines: list[str], fw: int, fh: int) -> "Geom":
        def p(v: list[float]) -> Rect:
            x, y, w, h = v
            return (int(round(x * fw)), int(round(y * fh)), int(round(w * fw)), int(round(h * fh)))

        return cls(
            cover=p(d["cover"]),
            box=p(d["box"]),
            font_px=max(MIN_FONT_PX, int(round(d["font"] * fh))),
            align=d.get("align", "center"),
            lines=lines,
            row_top=d["row_top"] * fh,
            row_h=d["row_h"] * fh,
            pitch=d.get("pitch", 0.0) * fh,
            stretch=int(d.get("stretch", STRETCH)),
            fg=tuple(d["fg"]) if d.get("fg") is not None else None,
            edge=tuple(d["edge"]) if d.get("edge") is not None else None,
        )

    def moved(self, dx: int, dy: int) -> "Geom":
        def mv(rect: Rect) -> Rect:
            x, y, w, h = rect
            return (x + dx, y + dy, w, h)

        return Geom(
            cover=mv(self.cover),
            box=mv(self.box),
            font_px=self.font_px,
            align=self.align,
            lines=self.lines,
            row_top=self.row_top + dy,
            row_h=self.row_h,
            pitch=self.pitch,
            stretch=self.stretch,
            fg=self.fg,
            edge=self.edge,
        )


def rows_overlap(a: Rect, b: Rect) -> bool:
    return a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def intersects(a: Rect, b: Rect) -> bool:
    return not (
        a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]
    )


def union(a: Rect, b: Rect) -> Rect:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return x0, y0, x1 - x0, y1 - y0


def widen(cover: Rect, need: int, align: str, obstacles: list[Rect], fw: int) -> tuple[Rect, int]:
    """(the drawing box, pixels still missing). Never taller, never wider than *need*."""
    x, y, w, h = cover
    if need <= w:
        return cover, 0
    extra = need - w
    left_lim, right_lim = 0, fw
    for o in obstacles:
        if not rows_overlap(o, cover) or intersects(o, cover):
            continue
        if o[0] + o[2] <= x:
            left_lim = max(left_lim, o[0] + o[2] + GAP)
        elif o[0] >= x + w:
            right_lim = min(right_lim, o[0] - GAP)
    free_l, free_r = max(0, x - left_lim), max(0, right_lim - (x + w))
    if align == "left":
        take_r = min(extra, free_r)
        take_l = min(extra - take_r, free_l)
    elif align == "right":
        take_l = min(extra, free_l)
        take_r = min(extra - take_l, free_r)
    else:
        take_l = min(int(math.ceil(extra / 2)), free_l)
        take_r = min(extra - take_l, free_r)
        take_l = min(extra - take_r, free_l)
    box = (x - take_l, y, w + take_l + take_r, h)
    return box, extra - take_l - take_r


def condensed(stretch_needed: float) -> int:
    return max(MIN_STRETCH, min(STRETCH, int(math.floor(stretch_needed))))


def drawn_size(font_px: float, stretch: int) -> float:
    """The size the width table is read at for *font_px* condensed to *stretch*."""
    return font_px * stretch / STRETCH


def fit_line(text: str, font_px: int, inner: float) -> tuple[int, int, str]:
    """(font size, stretch, text) at which *text* fits in *inner* pixels: condensed first,
    then smaller step by step; at the smallest size, cut with an ellipsis."""
    tw = text_width(text, font_px)
    if tw <= inner:
        return font_px, STRETCH, text
    stretch = condensed(STRETCH * inner / max(1.0, tw))
    size = float(font_px)
    while text_width(text, drawn_size(size, stretch)) > inner and size > MIN_FONT_PX:
        size = max(float(MIN_FONT_PX), size * SHRINK_STEP)
    font = int(math.floor(size))
    if text_width(text, drawn_size(font, stretch)) > inner:
        text = _cut(text, drawn_size(font, stretch), inner)
    return font, stretch, text


def _cut(text: str, size: float, inner: float) -> str:
    """*text* shortened to fit *inner* pixels at *size*, ending in an ellipsis."""
    while text and text_width(text + ELLIPSIS, size) > inner:
        text = text[:-1]
    return text.rstrip() + ELLIPSIS


def layout_line(
    cover: Rect,
    text: str,
    font_px: int,
    align: str,
    obstacles: list[Rect],
    fw: int,
    row_top: float,
    row_h: float,
) -> Geom:
    """One row of Japanese: one line of text, never wrapped, never wider than its box."""
    tw = text_width(text, font_px)
    box, short = widen(cover, int(math.ceil(tw + 2 * PAD_X)), align, obstacles, fw)
    if short <= 0:
        return Geom(cover, box, font_px, align, [text], row_top, row_h, 0.0)
    font, stretch, shown = fit_line(text, font_px, box[2] - 2 * PAD_X)
    return Geom(cover, box, font, align, [shown], row_top, row_h, 0.0, stretch)


def fit_block(
    text: str, font_px: int, rows: int, inner: float, spaced: bool = True
) -> tuple[int, int, list[str]]:
    """(font size, stretch, lines) that put *text* into at most *rows* lines of *inner*
    pixels: condensed first, then smaller step by step; at the smallest size the last line
    ends in an ellipsis."""
    stretch = STRETCH
    lines = wrap(text, font_px, inner, spaced)
    while len(lines) > rows and stretch > MIN_STRETCH:
        stretch = max(MIN_STRETCH, stretch - 2)
        lines = wrap(text, drawn_size(font_px, stretch), inner, spaced)
    size = float(font_px)
    while len(lines) > rows and size > MIN_FONT_PX:
        size = max(float(MIN_FONT_PX), size * SHRINK_STEP)
        lines = wrap(text, drawn_size(size, stretch), inner, spaced)
    font = int(math.floor(size))
    if len(lines) > rows:
        head = lines[: rows - 1]
        rest = (" " if spaced else "").join(lines[rows - 1 :])
        lines = head + [_cut(rest, drawn_size(font, stretch), inner)]
    return font, stretch, lines


def layout_block(
    cover: Rect,
    text: str,
    font_px: int,
    rows: int,
    pitch: float,
    align: str,
    obstacles: list[Rect],
    fw: int,
    row_top: float,
    row_h: float,
    spaced: bool = True,
) -> Geom:
    """Several rows of Japanese: one text wrapped into at most that many lines at the
    Japanese row pitch; the block widens when the lines do not fit, then condenses, then
    gets smaller."""
    inner = cover[2] - 2 * PAD_X
    lines = wrap(text, font_px, inner, spaced)
    if len(lines) <= rows:
        return Geom(cover, cover, font_px, align, lines, row_top, row_h, pitch)
    need = int(math.ceil(wrap_width(text, font_px, rows, spaced) + 2 * PAD_X + 1))
    box, _short = widen(cover, need, align, obstacles, fw)
    font, stretch, lines = fit_block(text, font_px, rows, box[2] - 2 * PAD_X, spaced)
    # exactly as wide as the widest line it holds
    widest = max(text_width(t, drawn_size(font, stretch)) for t in lines)
    box, _short = widen(cover, int(math.ceil(widest + 2 * PAD_X)), align, obstacles, fw)
    return Geom(cover, box, font, align, lines, row_top, row_h, pitch, stretch)


def blobs(frame: np.ndarray, rect: Rect, glyph_h: float) -> Rect | None:
    """The bounding box of the things drawn in *rect* that a cover must leave bare: crisp,
    high-contrast shapes at least half a glyph tall (a key badge, a list arrow, a
    stat icon, a bracket the database line does not have). A detector's slack around the
    text and the soft detail of a scene behind it do not count."""
    g = crop_gray(frame, rect)
    if g is None:
        return None
    gi = g.astype(np.int16)
    m = (np.abs(gi - int(np.median(gi))) > BLOB_CONTRAST).astype(np.uint8)
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    found = None
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if h < BLOB_MIN_H * glyph_h or w < BLOB_MIN_W * glyph_h:
            continue
        r = (max(0, rect[0]) + x, max(0, rect[1]) + y, w, h)
        found = r if found is None else union(found, r)
    return found


def near_blobs(frame: np.ndarray, cover: Rect, glyph_h: float) -> list[Rect]:
    """UI elements right beside a cover in its rows, within a glyph of the text: a key
    badge or an arrow the detector did not report. Further out, a longer translation may
    cover whatever is there (the scene, an empty panel)."""
    x, y, w, h = cover
    reach = int(np.ceil(NEAR * glyph_h))
    out: list[Rect] = []
    for zone in ((x - reach, y, reach, h), (x + w, y, reach, h)):
        found = blobs(frame, zone, glyph_h)
        if found is not None:
            out.append(found)
    return out


def cap_centre_baseline(band_top: float, band_h: float, font_px: int) -> float:
    """Baseline that centres the text (capital tops to descender bottoms) on a Japanese
    row."""
    cap = METRICS["cap"] * font_px
    desc = METRICS["descent"] * font_px * BASELINE_DESCENT
    return band_top + band_h / 2.0 + (cap - desc) / 2.0
