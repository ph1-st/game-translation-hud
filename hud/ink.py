"""Glyph strokes in screen pixels: what the covers are cut to and what the watch looks at.

* :func:`polarity_masks` are local contrast on either side (light glyphs on a dark scene,
  dark glyphs on a light panel). A scene showing through a translucent pill is a smooth
  gradient and drops out; thin strokes stay. Used to find the exact extent of a line
  (:mod:`hud.extent`) and the stroke cores the pixel watch looks at (:mod:`hud.anchors`).
* :func:`stroke_mask` is plain distance from the crop's median.

The grey copy of a frame is made once and shared (:func:`derived`): the OCR tick, the pixel
watch and memory all look at the newest frame many times.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import cv2
import numpy as np

Rect = tuple[int, int, int, int]
MEMO_FRAMES = 2  # frames whose derived pictures are kept: the OCR's and the watch's

HP_THRESHOLD = 22  # high-pass response that counts as an edge of a stroke
SIDE_THRESHOLD = 15  # ...and the stroke must sit this far on its side of the median
INK_THRESHOLD = 40  # grey distance from the crop's median that counts as a stroke (watch)
# grey spread around text on a popup panel (measured: popups 1.3-1.4; a blurred close-up
# of the scene behind dialogue is 7-15, a busy scene 20 and up)
FLAT_STD = 5.0


_memo: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []  # newest first
_memo_lock = threading.Lock()


def _memo_get(frame: np.ndarray, name: str) -> np.ndarray | None:
    with _memo_lock:
        for f, d in _memo:
            if f is frame:
                return d.get(name)
    return None


def derived(frame: np.ndarray, name: str, make: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """A picture derived from *frame* (its grey copy, a small copy), made once per frame.
    Frames are never changed once captured, so a frame object stands for its pixels; the
    entries hold their frames, so an id is never reused while it is remembered."""
    got = _memo_get(frame, name)
    if got is not None:
        return got
    value = make(frame)
    with _memo_lock:
        for f, d in _memo:
            if f is frame:
                d[name] = value
                break
        else:
            _memo.insert(0, (frame, {name: value}))
            del _memo[MEMO_FRAMES:]
    return value


def grey(frame: np.ndarray) -> np.ndarray:
    """The grey copy of a BGR frame (made once per frame)."""
    return derived(frame, "grey", lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))


def clip_rect(rect: Rect, fw: int, fh: int) -> Rect:
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def crop_gray(frame: np.ndarray, rect: Rect) -> np.ndarray | None:
    """Grey pixels of *rect* (clipped to the frame); None when almost nothing is left. Cut
    from the frame's grey copy when one was made already, else converted on its own."""
    h, w = frame.shape[:2]
    x0, y0, cw, ch = clip_rect(rect, w, h)
    if cw < 4 or ch < 4:
        return None
    whole = _memo_get(frame, "grey")
    if whole is not None:  # (a copy: an anchor keeps its crop, never the whole frame's grey)
        return whole[y0 : y0 + ch, x0 : x0 + cw].copy()
    return cv2.cvtColor(frame[y0 : y0 + ch, x0 : x0 + cw], cv2.COLOR_BGR2GRAY)


def polarity_masks(g: np.ndarray, sigma: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(light strokes, dark strokes) of a grey crop, by local contrast."""
    s = sigma if sigma is not None else max(2.0, g.shape[0] / 3.0)
    gf = g.astype(np.float32)
    hp = gf - cv2.GaussianBlur(gf, (0, 0), s)
    gi = g.astype(np.int16)
    med = int(np.median(gi))
    light = (hp > HP_THRESHOLD) & ((gi - med) > SIDE_THRESHOLD)
    dark = (hp < -HP_THRESHOLD) & ((med - gi) > SIDE_THRESHOLD)
    return light, dark


def thin(mask: np.ndarray, width: int = 4) -> np.ndarray:
    """Only the thin parts of a mask: glyph strokes are a few pixels wide, a patch of
    bright ground or a dark garment behind translucent text is not."""
    if not mask.any():
        return mask
    k = np.ones((width, width), np.uint8)
    fat = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    return mask & ~fat


def _shifted(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(m)
    h, w = m.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = m[ys, xs]
    return out


def flanked(a: np.ndarray, b: np.ndarray, reach: int = 3) -> float:
    """Share of the pixels of *a* that have *b* on both sides (left and right, or above
    and below) within *reach* pixels. The core of outlined glyphs is flanked by its outline
    (and dark text on a light panel by the bright halo the high-pass leaves around it);
    the outline itself has the core on one side only."""
    n = int(a.sum())
    if n == 0 or not b.any():
        return 0.0
    return float(flanked_mask(a, b, reach).sum()) / n


def flanked_mask(a: np.ndarray, b: np.ndarray, reach: int = 3) -> np.ndarray:
    """The pixels of *a* that have *b* on both sides within *reach* pixels."""
    left = np.zeros_like(b)
    right = np.zeros_like(b)
    up = np.zeros_like(b)
    down = np.zeros_like(b)
    for d in range(1, reach + 1):
        left |= _shifted(b, 0, d)  # b at x - d
        right |= _shifted(b, 0, -d)  # b at x + d
        up |= _shifted(b, d, 0)
        down |= _shifted(b, -d, 0)
    return a & ((left & right) | (up & down))


def on_panel(frame: np.ndarray, rect: Rect) -> bool:
    """Is the text in *rect* drawn on a popup panel rather than on the scene? A popup
    reaches well beyond its text in one perfectly flat colour, which even a blurred
    close-up behind a line of dialogue is not (judged on a margin of about a glyph)."""
    x, y, w, h = rect
    m = max(8, int(round(0.8 * h)))
    g = crop_gray(frame, (x - m, y - m, w + 2 * m, h + 2 * m))
    if g is None:
        return False
    light, dark = polarity_masks(g)
    ink = cv2.dilate((light | dark).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bg = g[~ink]
    if bg.size < 0.1 * g.size:
        return False
    return float(np.std(bg.astype(np.float32))) < FLAT_STD


def stroke_mask(g: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    gi = g.astype(np.int16)
    return np.abs(gi - int(np.median(gi))) > threshold


def runs(on: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) intervals where a 1-D boolean array is True."""
    edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def longest_runs(mask: np.ndarray) -> np.ndarray:
    """Per row of a 2-D boolean mask, the length of its longest run of True."""
    h = mask.shape[0]
    padded = np.zeros((h, mask.shape[1] + 2), np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    rows, starts = np.nonzero(edges == 1)
    _rows, ends = np.nonzero(edges == -1)  # (row-major: the k-th end closes the k-th start)
    out = np.zeros(h, np.int64)
    if len(rows):
        np.maximum.at(out, rows, ends - starts)
    return out
