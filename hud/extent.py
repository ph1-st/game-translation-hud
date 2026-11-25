"""Exactly where the matched Japanese is: the cover is that and nothing else.

The detector box of a line is looser than its glyphs (headroom above and below) and often
takes in a neighbour: a stat icon, a list arrow, a key badge, an icon the recognizer even
read as a character (日物理防御力). Two steps cut it down to the text we translate:

1. :func:`align_span`: which characters of the reading are the matched database line.
   Characters the line does not have (an icon read as 日, a trailing NEW) are left out.
2. :func:`measure`: which pixels those characters are. The recognizer's per-character cells
   bound the columns (so an icon or badge outside them is never included, even when the
   detector glued it to the line); inside those bounds the glyph strokes give the exact
   ink. The cover is that ink plus :data:`PAD` pixels.

The glyph height (what the translation is sized to) and the colours the glyphs are drawn
in (what the translation is drawn in) are measured on the same strokes.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

from capture.ocr import OcrLine
from db.normalize import normalize_char
from hud.ink import Rect, crop_gray, flanked, longest_runs, polarity_masks, runs, thin
from hud.looks import PLAIN_DARK, PLAIN_LIGHT, Color

PAD = 2  # pixels of cover around the ink
CROP_MARGIN = 3  # pixels around a detector box that are looked at (its glyphs may reach out)
GLYPH_PER_EM = 0.95  # a Japanese glyph is about as tall as its advance
# ...and within these of it (wide: the recognizer may add or drop a character)
GLYPH_MIN_EM, GLYPH_MID_EM, GLYPH_MAX_EM = 0.65, 0.9, 1.2
CELL_SLACK = 0.3  # an end character's ink may start this many pitches outside its cell
LINE_ROW = 2.5  # a row whose longest stroke run exceeds this many pitches is a rule/rim
EDGE_SHARE = 0.2  # a band edge row needs this share of the body's mean strokes per row
MIN_MASS = 0.35  # the text's polarity holds at least this share of the strokes of the other
BAND_TOLERANCE = 0.4  # ...and its band is within this share of the expected glyph height
FLANK_MIN = 0.25  # the core of the glyphs: this share of its strokes is flanked by the other
FLANK_GAP = 0.1  # ...clearly more than the other side is flanked by it
MIN_CONTRAST = 30  # grey levels between the text's strokes and the crop's background
MIN_CELL_COVERAGE = 0.75  # the text's polarity has strokes in this share of its characters
DECORATION = set("一ー－—―‐‒–-~〜・…‥.")  # a line's dashes and dots, however they are read
BAND_REACH = 0.25  # the text body may grow this share of a glyph height for sparse rows
CORE_SHARE = 0.08  # the fill colour: this share of the strongest stroke pixels (thin strokes
# blend with the background in all but a few pixels)
OUTLINE_SHARE = 0.6  # an outline has at least this many pixels per stroke pixel next to them
OUTLINE_FLANK = 0.25  # ...and hugs this share of the strokes on both sides
SATURATE = 1.4  # sampled colours are this much paler than the text looks
RULE_SPAN = 0.6  # ...or that spans this share of the crop
BAND_MIN_SHARE = 0.6  # the text body is never trimmed below this share of a glyph height
GLYPH_RUN = 1.3  # a run of ink this many pitches wide, centred in its cell, is one glyph
CUT_REACH = 0.35  # a run glued across a cell's edge is cut within this many pitches of it
CELL_MIN = 2  # stroke pixels a character's cell needs to count as holding ink
MIN_BOX_PX = 6  # a detector box lower than this holds no measurable text
BOX_PITCH = 0.85  # without character cells: the pitch is this share of the box height...
BOX_GLYPH = 0.65  # ...and the glyphs this share
SOFT_LEVEL = 0.5  # a glyph's edge rows: where the stroke profile crosses this share of it
SOFT_RANGE = (0.6, 1.4)  # a sub-pixel height outside this share of the whole rows is noise


@dataclass
class _Ink:
    left: int
    right: int
    top: int
    bottom: int
    coverage: float  # share of the characters whose cell holds strokes
    mass: int  # stroke pixels in the box


@dataclass
class Extent:
    cover: Rect  # the ink plus PAD: what gets covered
    ink: Rect  # the glyph strokes of the matched characters
    glyph_h: float  # height of the full-size glyphs (pixels)
    cells: list[tuple[int, int]]  # per matched character: its columns (frame x)
    light: bool  # light text on a darker background
    excluded: list[Rect]  # cells of the characters read but not matched (an icon read as 台)
    fill: Color = PLAIN_LIGHT  # the glyphs' colour (RGB)
    edge: Color | None = None  # their outline's colour, if they have one


def _norm_seq(text: str) -> tuple[str, list[int]]:
    """Folded text and, per folded character, the index of the raw character."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        for n in normalize_char(ch):
            out.append(n)
            idx.append(i)
    return "".join(out), idx


def align_span(reading: str, ref: str) -> tuple[int, int] | None:
    """[start, end) of the characters of *reading* that are the stored line *ref*.

    Characters at either end that the line does not have (deletions in the alignment) are
    left out; misread characters (substitutions) are part of the text. A reading that only
    has the start of the line (typewriter) is all text. None if nothing aligns."""
    a, ia = _norm_seq(reading)
    b, _ = _norm_seq(ref)
    if not a or not b:
        return None
    kept = [op for op in Levenshtein.opcodes(a, b) if op.tag in ("equal", "replace")]
    if not kept:
        return None
    s0, s1 = kept[0].src_start, kept[-1].src_end
    if s1 <= s0:
        return None
    return ia[s0], ia[s1 - 1] + 1


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _line_rows(mask: np.ndarray, pitch: float) -> np.ndarray:
    """Rows that are a rule, an underline or the rim of a pill: one long run of strokes,
    where glyph rows break into runs no longer than a glyph."""
    limit = max(LINE_ROW * pitch, RULE_SPAN * mask.shape[1])
    return longest_runs(mask) >= limit


def _band(profile: np.ndarray, expect: float, reach: float) -> tuple[int, int] | None:
    """The text rows. Glyphs of this size are about *expect* rows tall: the window of
    that height holding the most strokes is the text body. Its edge rows are trimmed when
    nearly empty and it grows by at most *reach* rows into rows that still carry a fair
    share of strokes (a dakuten, the top dot of ト). A scene behind the text adds a few
    strokes to every row (streaks, blobs) and must not stretch the band."""
    n = len(profile)
    if n == 0 or int(profile.max()) < 2:
        return None
    hw = int(min(n, max(3, round(expect))))
    sums = np.convolve(profile.astype(np.float64), np.ones(hw), mode="valid")
    top = int(np.argmax(sums))
    bottom = top + hw
    thr = max(2.0, EDGE_SHARE * float(profile[top:bottom].mean()))
    while bottom - top > BAND_MIN_SHARE * hw and profile[top] < thr:
        top += 1
    while bottom - top > BAND_MIN_SHARE * hw and profile[bottom - 1] < thr:
        bottom -= 1
    lo, hi = max(0, int(top - reach)), min(n, int(bottom + reach + 0.999))
    while top > lo and profile[top - 1] >= thr:
        top -= 1
    while bottom < hi and profile[bottom] >= thr:
        bottom += 1
    return top, bottom


def _ink(
    mask: np.ndarray,
    lim_l: float,
    lim_r: float,
    pitch: float,
    cells: list[tuple[int, int]] | None,
    expect: float,
) -> "_Ink | None":
    """Where the strokes of *mask* that belong to the columns [lim_l, lim_r) are (crop
    coordinates), how many of the characters' cells hold strokes, and how many strokes."""
    li, ri = int(np.floor(lim_l)), int(np.ceil(lim_r))
    reach = BAND_REACH * expect
    # rows: the text body of the columns that can be ours
    sub = mask[:, li:ri]
    band = _band(sub.sum(axis=1), expect, reach)
    if band is None:
        return None
    top, bottom = band
    # columns: ink runs inside the limits; a run glued across a limit is cut at its
    # weakest column there, a glyph-sized run that only overhangs its cell stays whole
    col = mask[top:bottom].sum(axis=0)
    kept: list[tuple[int, int]] = []
    for a, b in runs(col > 0):
        if b <= lim_l or a >= lim_r:
            continue
        if a >= lim_l and b <= lim_r:
            kept.append((a, b))
            continue
        centre = (a + b) / 2.0
        if lim_l <= centre <= lim_r and b - a <= GLYPH_RUN * pitch:
            kept.append((a, b))
            continue
        na, nb = a, b
        if a < lim_l:
            na = _cut(col, a, b, lim_l, CUT_REACH * pitch)
        if b > lim_r:
            nb = _cut(col, a, b, lim_r, CUT_REACH * pitch)
        if nb - na >= 1:
            kept.append((na, nb))
    if not kept:
        return None
    left, right = min(a for a, _ in kept), max(b for _, b in kept)
    # rows again, on exactly our columns
    sub = mask[:, left:right]
    band = _band(sub.sum(axis=1), expect, reach)
    if band is None:
        return None
    top, bottom = band
    body = mask[top:bottom]
    if cells:
        hit = sum(1 for a, b in cells if body[:, max(a, 0) : max(b, a + 1)].sum() >= CELL_MIN)
        coverage = hit / len(cells)
    else:
        coverage = float((body[:, left:right].sum(axis=0) > 0).mean())
    return _Ink(left, right, top, bottom, coverage, int(body[:, left:right].sum()))


def _cut(col: np.ndarray, a: int, b: int, at: float, reach: float) -> int:
    """The weakest column of run [a, b) near *at*: where a glued icon meets the text."""
    lo = max(a + 1, int(round(at - reach)))
    hi = min(b - 1, int(round(at + reach)))
    if hi <= lo:
        return int(round(min(max(at, a + 1), b - 1)))
    seg = col[lo : hi + 1]
    return lo + int(np.argmin(seg))


def _contrast(g: np.ndarray, ink: _Ink, mask: np.ndarray) -> float:
    """How far the strokes are from the crop's background (its median grey). Glyph
    strokes stand far out; the counters inside dark glyphs on a light panel, which the
    high-pass also marks as light, are barely brighter than the panel."""
    m = mask[ink.top : ink.bottom, ink.left : ink.right]
    if not m.any():
        return 0.0
    box = g[ink.top : ink.bottom, ink.left : ink.right]
    return abs(float(np.median(box[m])) - float(np.median(g)))


def _polarity(g: np.ndarray, found: list, expect: float) -> tuple[_Ink, np.ndarray, bool]:
    """The text's own polarity among the candidates [(ink, mask, light)].

    It has strokes in (nearly) every character, a fair share of all strokes (the other
    side of dark text on a flat panel is a few speckles), a band about as tall as
    Japanese glyphs of this pitch are, and it stands out from the background (the
    counters of dark glyphs do not). When both sides qualify, the core of the glyphs is
    the side flanked by the other (an outline or a drop shadow hugs light glyphs from
    outside, the high-pass leaves a bright halo around dark ones); without such a
    relation the side that stands out more from its surroundings is the text."""
    most = max(f[0].mass for f in found)
    ok = [
        f
        for f in found
        if f[0].coverage >= MIN_CELL_COVERAGE
        and f[0].mass >= MIN_MASS * most
        and abs((f[0].bottom - f[0].top) - expect) <= BAND_TOLERANCE * expect
        and _contrast(g, f[0], f[1]) >= MIN_CONTRAST
    ]
    if len(ok) == 1:
        return ok[0]
    if len(ok) == 2:
        (a, ma, _), (b, mb, _) = ok
        t, bt = min(a.top, b.top), max(a.bottom, b.bottom)
        lft, rgt = min(a.left, b.left), max(a.right, b.right)
        fa = flanked(ma[t:bt, lft:rgt], mb[t:bt, lft:rgt])
        fb = flanked(mb[t:bt, lft:rgt], ma[t:bt, lft:rgt])
        if max(fa, fb) >= FLANK_MIN and abs(fa - fb) >= FLANK_GAP:
            return ok[0] if fa > fb else ok[1]
        return max(ok, key=lambda f: _contrast(g, f[0], f[1]))
    cov = [f for f in found if f[0].coverage >= MIN_CELL_COVERAGE] or found
    return min(cov, key=lambda f: (abs((f[0].bottom - f[0].top) - expect), -f[0].mass))


@dataclass
class _Crop:
    """A reading's detector box and the grey crop around it (frame pixels)."""

    grey: np.ndarray
    x0: int  # the crop's origin
    y0: int
    box: tuple[int, int, int, int]  # the detector box: x0, y0, x1, y1


def _crop(frame: np.ndarray, line: OcrLine) -> _Crop | None:
    fh, fw = frame.shape[:2]
    bx0, by0 = int(np.floor(line.box.x * fw)), int(np.floor(line.box.y * fh))
    bx1, by1 = int(np.ceil(line.box.x2 * fw)), int(np.ceil(line.box.y2 * fh))
    cx0, cy0 = max(0, bx0 - CROP_MARGIN), max(0, by0 - CROP_MARGIN)
    cx1, cy1 = min(fw, bx1 + CROP_MARGIN), min(fh, by1 + CROP_MARGIN)
    g = crop_gray(frame, (cx0, cy0, cx1 - cx0, cy1 - cy0))
    if g is None or by1 - by0 < MIN_BOX_PX:
        return None
    return _Crop(g, cx0, cy0, (bx0, by0, bx1, by1))


def _limits(
    crop: _Crop, cells: list[tuple[float, float]] | None, k0: int, k1: int
) -> tuple[float, float, float] | None:
    """(left limit, right limit, character pitch) of the matched characters k0..k1 in the
    crop: their cells with a little slack, never into a neighbour's half; the whole box
    without cells."""
    bx0, by0, bx1, by1 = crop.box
    box_h = by1 - by0
    if cells is not None:
        widths = [b - a for a, b in cells[k0:k1] if b > a]
        pitch = float(np.median(widths)) if widths else BOX_PITCH * box_h
        slack = CELL_SLACK * pitch
        lim_l = cells[k0][0] - slack
        if k0 > 0:
            lim_l = max(lim_l, (cells[k0 - 1][1] + cells[k0][0]) / 2.0)
        lim_r = cells[k1 - 1][1] + slack
        if k1 < len(cells):
            lim_r = min(lim_r, (cells[k1 - 1][1] + cells[k1][0]) / 2.0)
    else:
        pitch = BOX_PITCH * box_h
        lim_l, lim_r = float(bx0 - crop.x0), float(bx1 - crop.x0)
    lim_l, lim_r = max(0.0, lim_l), min(float(crop.grey.shape[1]), lim_r)
    if lim_r - lim_l < 2:
        return None
    return lim_l, lim_r, pitch


def _masks(crop: _Crop, ignore: list[Rect] | None) -> tuple[np.ndarray, np.ndarray]:
    """The light and dark strokes of the crop, none in an *ignore* rect."""
    g = crop.grey
    light_m, dark_m = polarity_masks(g, sigma=max(2.0, g.shape[0] / 3.0))
    keep = np.ones_like(light_m)
    for ix, iy, iw, ih in ignore or []:
        ys = slice(max(0, iy - crop.y0), max(0, iy + ih - crop.y0))
        xs = slice(max(0, ix - crop.x0), max(0, ix + iw - crop.x0))
        keep[ys, xs] = False
    return light_m & keep, dark_m & keep


def measure(
    frame: np.ndarray,
    line: OcrLine,
    span: tuple[int, int] | None = None,
    ignore: list[Rect] | None = None,
) -> Extent | None:
    """The exact ink of characters [span) of *line* (all of them by default); pixels in
    the *ignore* rects (frame coordinates) are never text."""
    fw = frame.shape[1]
    crop = _crop(frame, line)
    if crop is None:
        return None
    text = line.text
    i0, i1 = span if span is not None else (0, len(text))
    nonspace = [i for i, c in enumerate(text) if not c.isspace()]
    cells = None
    if line.chars is not None and len(line.chars) == len(nonspace):
        cells = [(c.x0 * fw - crop.x0, c.x1 * fw - crop.x0) for c in line.chars]
    k0 = sum(1 for i in nonspace if i < i0)
    k1 = sum(1 for i in nonspace if i < i1)
    if k1 <= k0:
        return None
    limits = _limits(crop, cells, k0, k1)
    if limits is None:
        return None
    lim_l, lim_r, pitch = limits
    span_cells = None
    if cells is not None:
        span_cells = [(int(np.floor(a)), int(np.ceil(b))) for a, b in cells[k0:k1]]
    span_chars = [text[i] for i in nonspace[k0:k1]]
    em = em_pitch(span_chars, None if cells is None else cells[k0:k1])
    _bx0, by0, _bx1, by1 = crop.box
    expect = GLYPH_PER_EM * em if em else BOX_GLYPH * (by1 - by0)
    light_m, dark_m = _masks(crop, ignore)
    found = []
    for mask, light in ((light_m, True), (dark_m, False)):
        # only thin strokes, no rims of pills or underlines
        mask = thin(mask)
        mask = mask & ~_line_rows(mask, pitch)[:, None]
        got = _ink(mask, lim_l, lim_r, pitch, span_cells, expect)
        if got is not None:
            found.append((got, mask, light))
    if not found:
        return None
    best, _mask, light = _polarity(crop.grey, found, expect)
    left, right, top, bottom = best.left, best.right, best.top, best.bottom
    # each character's columns: the watch checks every glyph (a counter's digit changes)
    own_cells: list[tuple[int, int]] = []
    for j in range(len(span_chars)):
        a, b = span_cells[j] if span_cells is not None else (left, right)
        a, b = max(left, a), min(right, b)
        if b > a:
            own_cells.append((crop.x0 + a, crop.x0 + b))
    # what was read beside the matched text and must stay bare (an icon read as a
    # character); a dash or dots the stored line lacks are its own decoration, not that
    excluded = [
        (crop.x0 + int(np.floor(a)), by0, max(1, int(np.ceil(b - a))), by1 - by0)
        for k, (a, b) in enumerate(cells or [])
        if (k < k0 or k >= k1) and text[nonspace[k]] not in DECORATION
    ]
    em_long = em if em is not None and sum(1 for c in span_chars if is_wide(c)) >= 4 else None
    glyph_h = plausible(soft_height(crop.grey, left, right, top, bottom, light), em_long)
    ink = (crop.x0 + left, crop.y0 + top, right - left, bottom - top)
    cover = (ink[0] - PAD, ink[1] - PAD, ink[2] + 2 * PAD, ink[3] + 2 * PAD)
    fill, edge = _colours(frame, crop, (left, top, right, bottom), light_m, dark_m, light)
    return Extent(cover, ink, glyph_h, own_cells, light, excluded, fill, edge)


def _colours(
    frame: np.ndarray,
    crop: _Crop,
    ink: tuple[int, int, int, int],
    light_m: np.ndarray,
    dark_m: np.ndarray,
    light: bool,
) -> tuple[Color, Color | None]:
    """The colours of the glyphs in *ink* (crop coordinates: left, top, right, bottom),
    sampled on them and the pixels right around them."""
    left, top, right, bottom = ink
    ch, cw = crop.grey.shape
    rows = slice(max(0, top - PAD), min(ch, bottom + PAD))
    cols = slice(max(0, left - PAD), min(cw, right + PAD))
    text_m, other_m = (light_m, dark_m) if light else (dark_m, light_m)
    bgr = frame[crop.y0 : crop.y0 + ch, crop.x0 : crop.x0 + cw]
    return ink_colours(bgr[rows, cols], crop.grey[rows, cols], text_m[rows, cols],
                       other_m[rows, cols], light)  # fmt: skip


def _rgb(bgr: np.ndarray) -> Color:
    b, g, r = (int(round(float(v))) for v in bgr)
    return r, g, b


def _saturated(rgb: Color) -> Color:
    """The colour of the glyphs before anti-aliasing blended it with the background: no
    pixel of a thin stroke holds the full colour, every sample is paler than the text
    looks. Saturation is restored by SATURATE (white stays white, grey stays grey)."""
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    hsv[1] = min(255.0, hsv[1] * SATURATE)
    r, g, b = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
    return int(r), int(g), int(b)


def ink_colours(
    bgr: np.ndarray, g: np.ndarray, text: np.ndarray, other: np.ndarray, light: bool
) -> tuple[Color, Color | None]:
    """The colours glyphs are drawn in (RGB): the fill, from the strongest stroke pixels
    (the inside of a stroke; its edges are blended with the background), and the outline,
    when the glyphs have one: the other side's strokes right next to them, hugging them on
    both sides. A glow (the halo of a title) is the glyphs' own side and is no outline; a
    plain label on a panel has none."""
    plain = PLAIN_LIGHT if light else PLAIN_DARK
    if not text.any():
        return plain, None
    gi = g.astype(np.float32)
    bg = float(np.median(gi))
    strength = (gi - bg) if light else (bg - gi)
    vals = strength[text]
    core = text & (strength >= np.percentile(vals, 100.0 * (1.0 - CORE_SHARE)))
    fill = _saturated(_rgb(np.median(bgr[core], axis=0)))
    near = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring = near & other & ~text
    n = int(text.sum())
    if ring.sum() < OUTLINE_SHARE * n or flanked(text, other, reach=4) < OUTLINE_FLANK:
        return fill, None
    depth = -strength
    darkest = ring & (depth >= np.percentile(depth[ring], 50.0))
    return fill, _rgb(np.median(bgr[darkest], axis=0))


def row_em(line: OcrLine, fw: int, min_wide: int = 4) -> float | None:
    """The advance of a full-width character on a long row, from the recognizer's
    character positions (None for short rows, where the positions are too coarse)."""
    if line.chars is None:
        return None
    chars = [c.ch for c in line.chars]
    if sum(1 for c in chars if is_wide(c)) < min_wide:
        return None
    return em_pitch(chars, [(c.x0 * fw, c.x1 * fw) for c in line.chars])


def plausible(glyph_h: float, em: float | None) -> float:
    """A measured glyph height, checked against the row's advance: a Japanese glyph is
    about 0.8 to 1.0 of it (letter-spacing only widens the advance; the bounds allow for a
    character the recognizer added or dropped). An ink band stretched by a bright patch of
    scene behind the text is replaced by the middle of that range."""
    if em is None or GLYPH_MIN_EM * em <= glyph_h <= GLYPH_MAX_EM * em:
        return glyph_h
    return GLYPH_MID_EM * em


def em_pitch(chars: list[str], cells: list[tuple[float, float]] | None) -> float | None:
    """The advance of a full-width character (about the em size), from the recognizer's
    character positions: over each run of neighbouring wide characters, the distance from
    the first centre to the last per step. Single steps alternate (the positions come in
    steps of the recognizer's output columns), a run averages that out."""
    if cells is None or len(cells) != len(chars):
        return None
    centres = [((a + b) / 2.0, is_wide(c)) for c, (a, b) in zip(chars, cells, strict=True)]
    dist, steps = 0.0, 0
    i = 0
    while i < len(centres):
        if not centres[i][1]:
            i += 1
            continue
        j = i
        while j + 1 < len(centres) and centres[j + 1][1]:
            j += 1
        if j > i:
            dist += centres[j][0] - centres[i][0]
            steps += j - i
        i = j + 1
    if steps:
        return dist / steps
    widths = [b - a for (a, b), c in zip(cells, chars, strict=True) if is_wide(c) and b > a]
    return float(np.median(widths)) if widths else None


def soft_height(g: np.ndarray, left: int, right: int, top: int, bottom: int, light: bool) -> float:
    """The glyph height with sub-pixel edges: on the line's profile of stroke strength
    (how far each pixel is from the background, on the text's side), the rows where it
    crosses half the level of the text body, interpolated. A row of whole pixels is a
    pixel too short or too long depending on where the text happens to sit on the pixel
    grid; this is the same for every line drawn at one size."""
    n = bottom - top
    t0, t1 = max(0, top - 3), min(g.shape[0], bottom + 3)
    sub = g[t0:t1, left:right].astype(np.float32)
    bg = float(np.median(g))
    prof = np.clip((sub - bg) if light else (bg - sub), 0, None).sum(axis=1)
    lo = (top - t0) + n // 4
    core = prof[lo : lo + max(1, n - 2 * (n // 4))]
    ref = SOFT_LEVEL * float(np.median(core)) if core.size else 0.0
    if ref <= 0:
        return float(n)

    def rise(seq: np.ndarray) -> float:
        for i in range(1, len(seq)):
            if seq[i - 1] < ref <= seq[i]:
                return i - 1 + (ref - seq[i - 1]) / max(1e-6, float(seq[i] - seq[i - 1]))
        return 0.0

    h = (len(prof) - rise(prof[::-1])) - rise(prof)
    lo, hi = SOFT_RANGE
    return float(h) if lo * n <= h <= hi * n else float(n)
