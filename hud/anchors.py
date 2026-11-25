"""Pixel anchors: the only authority on whether the text under a cover is still there.

When a cover is placed, the grey crop under it and the positions of its glyph strokes (the
pixels that stand out from the crop's median) are stored. Every 50 ms the same crop of the
newest frame is compared with that reference, without any OCR:

* a static screen correlates almost perfectly with the reference: the text is there
* otherwise only the stroke pixels are judged. If they kept their values (raw, or after
  removing a tint that hit the whole box) and still stand out from the background around
  them, the text is there, whatever happens to the rest of the box (a scene moving behind a
  translucent panel, a hover highlight). If most of them changed, the text is gone.

The same test decides whether a reading may be placed at all: the text must have held
still between the frame the OCR read and the newest frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from hud.ink import INK_THRESHOLD, Rect, crop_gray, polarity_masks, stroke_mask, thin

MIN_STROKES = 12  # fewer stroke pixels than this: nothing to watch
CORE_SHARE = 30  # percent of the text-polarity pixels, the most extreme, that are watched
SETTLED_DELTA = 18  # placing: the stroke cores changed by no more than this (a fade has)
STAND_SHARE = 0.4  # strokes still stand out by this share of their contrast when placed
CELL_MIN_STROKES = 8  # a character with fewer watched stroke pixels is not checked alone
STEADY_MEDIAN = 6  # grey levels: the crop's median change when nothing but a glyph moved
NCC_HOLD = 0.6  # structure correlation at or above this: unchanged
STROKE_DELTA_MAX = 40  # mean change of the stroke pixels that still counts as the same text
STROKE_DELTA_GONE = 60  # beyond this (with nothing standing out) the text is plainly gone
STROKE_RECALL_MIN = 0.5  # fallback: share of the strokes that must still stand out
STROKE_PRECISION_MIN = 0.35  # fallback: share of what stands out that must be the old strokes
STANDING_SHARE = 0.6  # fallback: a pixel stands out at this share of the ink threshold
GLYPH_CHANGED = 0.4  # a character changed: this share of its strokes changed a lot

HOLDS, WEAK_FAIL, HARD_FAIL = 2, 1, 0


@dataclass
class Anchor:
    rect: Rect
    gray: np.ndarray  # uint8 crop under the cover when it was placed
    strokes: np.ndarray  # bool mask of the glyph strokes in that crop
    raw_vals: np.ndarray  # int16 grey values at the strokes
    norm_vals: np.ndarray  # the same minus the crop's background median
    light: bool | None = None  # the text's polarity, when known
    local: float = 0.0  # how far the strokes stood out from their neighbourhood
    cells: list[np.ndarray] | None = None  # per character: its share of the strokes


def make_anchor(
    frame: np.ndarray,
    rect: Rect,
    light: bool | None = None,
    cells: list[tuple[int, int]] | None = None,
) -> Anchor | None:
    """The reference for the text under *rect*. When the text's polarity is known (light
    glyphs, or dark), the strokes watched are the cores of the glyph strokes: the most
    extreme pixels of that polarity. Game text is drawn in one flat colour, so its cores
    share one value and keep it, whatever moves behind translucent text or cuts to another
    shot behind a subtitle; anti-aliased edges and bright or dark specks of the scene sit
    between that value and the background and are no part of what is watched."""
    g = crop_gray(frame, rect)
    if g is None:
        return None
    strokes = None
    if light is not None:
        lm, dm = polarity_masks(g)
        strokes = thin(lm if light else dm) & stroke_mask(g, INK_THRESHOLD // 2)
        if int(strokes.sum()) >= MIN_STROKES:
            vals = g[strokes]
            if light:
                core = strokes & (g >= np.percentile(vals, 100 - CORE_SHARE))
            else:
                core = strokes & (g <= np.percentile(vals, CORE_SHARE))
            if int(core.sum()) >= MIN_STROKES:
                strokes = core
        if int(strokes.sum()) < MIN_STROKES:
            strokes = None
    if strokes is None:
        strokes = stroke_mask(g)
    if int(strokes.sum()) < MIN_STROKES:
        return None
    gi = g.astype(np.int16)
    non = ~strokes
    bg = int(np.median(gi[non])) if int(non.sum()) >= 8 else int(np.median(gi))
    raw = gi[strokes]
    a = Anchor(rect, g, strokes, raw, raw - bg, light)
    a.local = _local(gi, strokes, light, raw - bg)
    if cells:
        # which of the watched strokes are which character's (columns, frame x)
        xs = np.nonzero(strokes)[1] + max(0, rect[0])
        a.cells = [(xs >= x0) & (xs < x1) for x0, x1 in cells]
    return a


def settled(frame: np.ndarray, anchor: Anchor) -> bool:
    """Has the text under the anchor stayed exactly as it was (not merely still there)?
    A cover is only placed on text that held still while the OCR read it: a label still
    fading in or sliding would be measured mid-animation and removed a moment later. The
    stroke cores must have kept their values (the scene behind may have moved)."""
    cur = crop_gray(frame, anchor.rect)
    if cur is None or cur.shape != anchor.gray.shape:
        return False
    ci = cur.astype(np.int16)
    vals = ci[anchor.strokes]
    if float(np.percentile(np.abs(vals - anchor.raw_vals), 60)) > SETTLED_DELTA:
        return False
    return _stands(ci, anchor, vals)


def recognizes(frame: np.ndarray, anchor: Anchor) -> bool:
    """Is the text of *anchor* at its place in *frame* exactly as it was, so that a cover
    can be put back on it without the OCR: the strokes at the same pixels with the values
    they had (as strict as placing), standing out, and every character's own strokes too
    (a line that differs in one character is another line)."""
    cur = crop_gray(frame, anchor.rect)
    if cur is None or cur.shape != anchor.gray.shape:
        return False
    ci = cur.astype(np.int16)
    vals = ci[anchor.strokes]
    if float(np.percentile(np.abs(vals - anchor.raw_vals), 60)) > SETTLED_DELTA:
        return False
    if not _stands(ci, anchor, vals):
        return False
    for sel in anchor.cells or []:
        if int(sel.sum()) < CELL_MIN_STROKES:
            continue
        if float(np.percentile(np.abs(vals[sel] - anchor.raw_vals[sel]), 60)) > STROKE_DELTA_MAX:
            return False
    return True


def anchor_state(frame: np.ndarray, anchor: Anchor) -> int:
    """HOLDS: the text is there. HARD_FAIL: the strokes are plainly gone (nothing stands
    out where they were, their values changed a lot): the cover goes at once. WEAK_FAIL:
    changed, but not beyond doubt (a cursor, a transition frame, one character of the
    line): the watch waits for one more frame.

    The main test is lenient on purpose: if the stroke pixels kept their values (raw, or
    after removing a tint over the whole box) the text is there, however bright or busy the
    background behind a translucent HUD element has become. Where nothing else in the crop
    moved, a line that holds as a whole is still a changed line when one of its characters
    changed (a counter's digit); over a moving scene a single character's few stroke
    pixels are too few to tell, and the line as a whole decides."""
    cur = crop_gray(frame, anchor.rect)
    if cur is None or cur.shape != anchor.gray.shape:
        return HARD_FAIL
    state = _line_state(cur, anchor)
    if state == HOLDS and anchor.cells and _steady(cur, anchor):
        if _a_glyph_changed(cur.astype(np.int16), anchor):
            return WEAK_FAIL
    return state


def why(frame: np.ndarray, anchor: Anchor) -> str:
    """For the log: what changed under a cover that went."""
    cur = crop_gray(frame, anchor.rect)
    if cur is None or cur.shape != anchor.gray.shape:
        return "off screen"
    state = _line_state(cur, anchor)
    if state == HARD_FAIL:
        return "gone"
    if state == WEAK_FAIL:
        return "changed"
    return "one character changed"


def _steady(cur: np.ndarray, anchor: Anchor) -> bool:
    """Is (almost) everything in the crop as it was? On a static panel a changed digit is
    the only thing that moves; behind text over the scene, nearly every pixel changes."""
    d = np.abs(cur.astype(np.int16) - anchor.gray.astype(np.int16))
    return float(np.median(d)) <= STEADY_MEDIAN


def _ncc(cur: np.ndarray, anchor: Anchor) -> float:
    """Correlation of the crop with the one under the cover when it was placed: near 1
    while nothing around the text moved."""
    a = anchor.gray.astype(np.float32) - float(anchor.gray.mean())
    b = cur.astype(np.float32) - float(cur.mean())
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / denom if denom > 1e-6 else 0.0


def _line_state(cur: np.ndarray, anchor: Anchor) -> int:
    ncc = _ncc(cur, anchor)
    ci = cur.astype(np.int16)
    vals = ci[anchor.strokes]
    # a screen that stayed still correlates with the reference, and the text is there,
    # as long as its strokes still stand out: over a busy scene that holds still, the
    # scene alone correlates when a subtitle has gone
    if ncc >= NCC_HOLD and _stands(ci, anchor, vals):
        return HOLDS
    strokes = anchor.strokes
    non = ~strokes
    bg = int(np.median(ci[non])) if int(non.sum()) >= 8 else int(np.median(ci))
    # the strokes include some background texture (a character moving behind a translucent
    # panel); the glyphs are the majority, so the decision is on the steadier 60% of the
    # stroke pixels, not the mean
    delta = min(
        float(np.percentile(np.abs(vals - anchor.raw_vals), 60)),
        float(np.percentile(np.abs((vals - bg) - anchor.norm_vals), 60)),
    )
    stands = _stands(ci, anchor, vals)
    if delta <= STROKE_DELTA_MAX and stands:
        return HOLDS  # the strokes kept their values and still stand out: the text is there
    standing = np.abs(ci - bg) > int(INK_THRESHOLD * STANDING_SHARE)
    recall = float(standing[strokes].mean())
    precision = float((standing & strokes).sum()) / float(max(1, int(standing.sum())))
    if recall >= STROKE_RECALL_MIN and precision >= STROKE_PRECISION_MIN:
        return HOLDS
    if recall < 0.15 and ncc < 0.3 and (delta > STROKE_DELTA_GONE or not stands):
        return HARD_FAIL
    return WEAK_FAIL


def _a_glyph_changed(ci: np.ndarray, anchor: Anchor) -> bool:
    """Did the strokes of one character change while the line as a whole looks the same?
    Judged like the whole line (raw values or after a tint), glyph by glyph."""
    if not anchor.cells:
        return False
    non = ~anchor.strokes
    bg = int(np.median(ci[non])) if int(non.sum()) >= 8 else int(np.median(ci))
    vals = ci[anchor.strokes]
    raw = np.abs(vals - anchor.raw_vals) > STROKE_DELTA_MAX
    tint = np.abs((vals - bg) - anchor.norm_vals) > STROKE_DELTA_MAX
    cells = np.stack(anchor.cells)  # characters x strokes, all at once (every 50 ms)
    n = cells.sum(axis=1)
    # in raw values and after a tint
    gone = ((cells & raw).sum(axis=1) > GLYPH_CHANGED * n) & (
        (cells & tint).sum(axis=1) > GLYPH_CHANGED * n
    )
    return bool(np.any(gone & (n >= CELL_MIN_STROKES)))


def _local(ci: np.ndarray, strokes: np.ndarray, light: bool | None, norm: np.ndarray) -> float:
    """How far the strokes stand out from their immediate neighbourhood (5x5), on the
    text's side: light glyph cores above the darkest pixel next to them (the glyph's own
    outline, when it has one), dark ones below the brightest."""
    if light is None:
        light = float(np.median(norm)) > 0
    k = np.ones((5, 5), np.uint8)
    g = ci.astype(np.uint8)
    if light:
        c = ci - cv2.erode(g, k).astype(np.int16)
    else:
        c = cv2.dilate(g, k).astype(np.int16) - ci
    return float(np.median(c[strokes]))


def _stands(ci: np.ndarray, anchor: Anchor, vals: np.ndarray) -> bool:
    """Do the strokes still stand out from what is right around them, on the same side
    (light text stays lighter)? Strokes that merely kept a value close to a plain field
    that replaced the text do not; outlined glyphs over a scene that turned as bright as
    the glyphs still do (their outline is next to them)."""
    now = _local(ci, anchor.strokes, anchor.light, anchor.norm_vals)
    return now >= STAND_SHARE * anchor.local and now > 0
