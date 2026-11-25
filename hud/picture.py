"""Where the game's picture is in its window: the whole client area, or a centred picture
of the game's shape between black bars (a 16:9 game or video shown full screen on a 21:9
monitor, an emulator's window of another shape). The profile's regions are fractions of
the picture, so the HUD reads and draws within the picture only.

Only the middle of the bars is judged (a video player's title and controls come and go at
the top and bottom), and a picture that is all black tells nothing (a fade to black)."""

from __future__ import annotations

import numpy as np

Rect = tuple[int, int, int, int]

BAR_MAX = 16  # a bar's pixels are at most this bright (black, plus video noise)
BAR_SHARE = 0.995  # ...nearly all of them
MIN_BAR = 8  # px: a thinner margin is rounding, not a bar
MIDDLE = (0.2, 0.8)  # the part of a bar that is judged (of its length)
UNKNOWN = "unknown"  # an all-black screen: nothing to tell by


def find_picture(frame: np.ndarray, aspect: float) -> Rect | None | str:
    """The picture of shape *aspect* (width / height) in *frame*, as (x, y, w, h), if black
    bars surround it; None if the picture fills the frame (it has the shape, or the sides
    show something); UNKNOWN if everything is black."""
    h, w = frame.shape[:2]
    a, b = MIDDLE
    if w / h > aspect:  # wider: bars left and right
        pw = int(round(h * aspect))
        bar = (w - pw) // 2
        if bar < MIN_BAR:
            return None
        rect = (bar, 0, pw, h)
        mid = slice(int(h * a), int(h * b))
        bars = (frame[mid, :bar], frame[mid, bar + pw :])
        picture = frame[:, bar : bar + pw]
    else:  # taller: bars above and below
        ph = int(round(w / aspect))
        bar = (h - ph) // 2
        if bar < MIN_BAR:
            return None
        rect = (0, bar, w, ph)
        mid = slice(int(w * a), int(w * b))
        bars = (frame[:bar, mid], frame[bar + ph :, mid])
        picture = frame[bar : bar + ph]
    for side in bars:
        if side.size and (side.max(axis=-1) <= BAR_MAX).mean() < BAR_SHARE:
            return None  # the sides show something: the game fills the window
    if picture[::8, ::8].max() <= 2 * BAR_MAX:
        return UNKNOWN
    return rect
