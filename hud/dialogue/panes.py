"""The frosted glass under dialogue.

Two modes: ``fixed``, one pane over the layout's whole glass area, the same for every line
(a window); ``grow``, one pane over everything shown so far, that grows with the text and
never shrinks while the scene runs. The pane takes in the dialogue rows on screen (found
from their strokes every watch, and from every reading) and the translation's lines, centred
like the game's rows (a row's first characters tell where it ends: the box is mirrored about
the centre) or flush left in a message box, always inside the layout's glass area.
"""

from __future__ import annotations

import itertools

import numpy as np

from hud.backdrop import text_mask
from hud.dialogue.scene import DIALOGUE_PITCH, Pane, Scene, centre_x, text_area
from hud.ink import Rect
from hud.layout import PAD_X, drawn_size, union
from hud.textwidth import text_width

FIXED, GROW = "fixed", "grow"
MODES = (FIXED, GROW)
BOX_PAD = 0.5  # the glass reaches this many glyph heights beyond its text...
BOX_STEP = 4  # ...its edges on a grid of this many pixels (noise does not resize it)
TEXT_MIN = 25  # stroke pixels that mean "dialogue text in the area"
TEXT_GAP = 2.0  # stroke columns further apart than this many glyph heights are not one row
ROW_JOIN = 0.35  # stroke rows closer than this many glyph heights are one row of text...
ROW_DENSE = 0.12  # ...counting rows with at least this share of the densest one's strokes
SPECK_ROWS = 0.3  # a band of strokes thinner than this many glyph heights is no row
_PANE_IDS = itertools.count(1)  # (unique across scenes: a fading pane is never another's)


def fixed_pane(sc: Scene, area: Rect) -> None:
    """The one pane of the fixed mode, over the layout's whole glass area."""
    sc.fixed, sc.pane_seq = True, next(_PANE_IDS)
    sc.panes = [Pane(area, sc.pane_seq)]


def grow(sc: Scene, rect: Rect) -> bool:
    """Take *rect* into the scene's pane (made with the first text); it never shrinks.
    Returns whether it changed."""
    if sc.fixed:
        return False
    r = tuple(int(v) for v in rect)
    if not sc.panes:
        sc.pane_seq = next(_PANE_IDS)
        sc.panes.append(Pane(r, sc.pane_seq))
        return True
    pane = sc.panes[0]
    grown = union(pane.rect, r)
    if all(abs(a - b) <= 1 for a, b in zip(grown, pane.rect, strict=True)):
        return False
    pane.rect = tuple(int(v) for v in grown)
    return True


def from_read(sc: Scene, rows: list[Rect], fw: int, fh: int) -> bool:
    """The glass over the rows a reading found and over the translation (worker thread)."""
    changed = False
    for r in rows:
        changed = grow(sc, symmetric(sc, r, fw, fh)) or changed
    return over_text(sc, fw, fh) or changed


def from_strokes(frame: np.ndarray, sc: Scene) -> bool:
    """The glass over the dialogue rows on screen now (the watch, every 50 ms)."""
    fh, fw = frame.shape[:2]
    changed = False
    for r in stroke_rows(frame, sc):
        changed = grow(sc, r) or changed
    return over_text(sc, fw, fh) or changed


def over_text(sc: Scene, fw: int, fh: int) -> bool:
    """The glass over each line of the translation (it may be wider than its row)."""
    if sc.line is None or not sc.lines or sc.text_box is None:
        return False
    changed = False
    size = drawn_size(sc.text_font or sc.font_px, sc.text_stretch)
    cx = centre_x(sc, fw, fh)
    for i, line in enumerate(sc.lines):
        w = text_width(line, size) + 2 * PAD_X
        top = sc.text_top + i * sc.text_pitch
        x = sc.text_box[0] if sc.align == "left" else cx - w / 2.0
        changed = grow(sc, symmetric(sc, (x, top, w, sc.text_band), fw, fh)) or changed
    return changed


def stroke_rows(frame: np.ndarray, sc: Scene) -> list[Rect]:
    """A box for each row of dialogue strokes on screen in the layout's glass area (a row's
    columns: the run with the most strokes), top to bottom."""
    fh, fw = frame.shape[:2]
    got = text_mask(frame, text_area(sc, fw, fh), sc.glyph, sc.fill, sc.edge)
    if got is None:
        return []
    ox, oy, mask = got
    if int(mask.sum()) < TEXT_MIN:
        return []
    prof = mask.sum(axis=1)  # a few stray stroke pixels between two rows are no text
    ys = np.nonzero(prof >= max(2, ROW_DENSE * int(prof.max())))[0]
    if len(ys) == 0:
        return []
    bands = np.split(ys, np.nonzero(np.diff(ys) > ROW_JOIN * sc.glyph)[0] + 1)
    out: list[Rect] = []
    for band in bands:
        y0, y1 = int(band[0]), int(band[-1]) + 1
        cols = mask[y0:y1].sum(axis=0)
        if int(cols.sum()) < TEXT_MIN // 2 or y1 - y0 < SPECK_ROWS * sc.glyph:
            continue
        xs = np.nonzero(cols)[0]
        runs = np.split(xs, np.nonzero(np.diff(xs) > TEXT_GAP * sc.glyph)[0] + 1)
        run = max(runs, key=lambda r: int(cols[r].sum()))
        x0, x1 = int(run[0]), int(run[-1]) + 1
        out.append(symmetric(sc, (ox + x0, oy + y0, x1 - x0, y1 - y0), fw, fh))
    return out


def symmetric(sc: Scene, rect: tuple[float, float, float, float], fw: int, fh: int) -> Rect:
    """A box over *rect* of the dialogue rows, centred like the rows are (as far right of
    the middle as left: a row's first characters tell where it ends), with a margin (above
    and below, half way to the next row at least: the boxes of two rows meet), its edges on
    a coarse grid, inside the layout's glass area."""
    cx = centre_x(sc, fw, fh)
    pad = BOX_PAD * sc.glyph
    pitch = sc.row_pitch or DIALOGUE_PITCH * sc.font_px
    vpad = max(pad, (pitch - sc.glyph) / 2.0 + 2.0)
    x, y, w, h = rect
    top = BOX_STEP * np.floor((y - vpad) / BOX_STEP)
    bottom = BOX_STEP * np.ceil((y + h + vpad) / BOX_STEP)
    gx, gy, gw, gh = text_area(sc, fw, fh)
    top, bottom = max(top, gy), min(bottom, gy + gh)
    if sc.align == "left":  # (rows start at the left edge: no mirroring)
        left = max(BOX_STEP * np.floor((x - pad) / BOX_STEP), gx)
        right = min(BOX_STEP * np.ceil((x + w + pad) / BOX_STEP), gx + gw)
        return (int(left), int(top), int(right - left), int(bottom - top))
    half = BOX_STEP * np.ceil((max(cx - x, x + w - cx) + pad) / BOX_STEP)
    half = min(half, cx - gx, gx + gw - cx)
    return (int(round(cx - half)), int(top), int(round(2 * half)), int(bottom - top))
