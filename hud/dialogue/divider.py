"""The divider: the line with a star the game draws under a speaker name. While its star is
on screen, a conversation runs in that layout: the moment it shows, the conversation's
glass is up (before any OCR), and once it has been gone a moment with no line shown, the
conversation is over.

The star is found by template matching on the picture's fine structure (its own blur
taken away), so whatever scene lies behind it matches alike.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from capture.profile import GAMES_DIR, DialogueLayout, Divider, Profile
from hud.backdrop import text_mask
from hud.covercache import CoverCache
from hud.covers import px
from hud.looks import PLAIN_LIGHT

log = logging.getLogger("hud")

AGAIN_WITHIN_S = 6.0  # a conversation seen this recently comes back at the lower threshold
GONE_S = 0.25  # a conversation is over once its divider has been gone this long
OFFSET = 0.007  # the star may sit this far (of the frame height) from where it should be
TEXT_MIN = 25  # stroke pixels of the layout's text that mean "its text is there"
HIGHPASS = 2.0  # the fine structure: the picture minus its blur at this sigma (at 900 px)
REFERENCE_H = 900.0


class DividerWatch:
    """Which layouts' dividers are up."""

    def __init__(self, profile: Profile, cache: CoverCache) -> None:
        self.profile = profile
        self.cache = cache
        self._seen_at: dict[str, float] = {}  # when each layout's divider was last seen
        self._stars: dict[tuple[str, int], np.ndarray | None] = {}  # pictures, per height

    def up(self, frame: np.ndarray, now: float) -> list[DialogueLayout]:
        """The layouts whose divider is on screen now (their last sighting updated)."""
        out = []
        for lay in self.profile.dialogue_layouts.values():
            div = lay.divider
            if div is None:
                continue
            score = self.score(frame, lay)
            recent = now - self._seen_at.get(lay.name, -1e9) < AGAIN_WITHIN_S
            if score >= div.match or (
                recent and score >= div.again and self.text_there(frame, lay)
            ):
                out.append(lay)
        for lay in out:
            self._seen_at[lay.name] = now
        return out

    def gone(self, lay: DialogueLayout, now: float) -> bool:
        """Has *lay*'s divider been off screen long enough for its conversation to end?"""
        return now - self._seen_at.get(lay.name, -1e9) > GONE_S

    def any_seen(self, now: float) -> bool:
        """Is any layout's divider on screen (or was it a moment ago)?"""
        return any(now - t <= GONE_S for t in self._seen_at.values())

    def score(self, frame: np.ndarray, lay: DialogueLayout) -> float:
        """How well the star of *lay*'s divider matches where it sits (centred, at its
        height): -1 to 1."""
        div = lay.divider
        if div is None:
            return -1.0
        fh, fw = frame.shape[:2]
        star = self._star(div, fh)
        if star is None:
            return -1.0
        th, tw = star.shape
        m = max(4, int(round(OFFSET * fh)))
        x0, y0 = fw // 2 - tw // 2 - m, int(round(div.y * fh)) - th // 2 - m
        if x0 < 0 or y0 < 0 or x0 + tw + 2 * m > fw or y0 + th + 2 * m > fh:
            return -1.0
        area = cv2.cvtColor(frame[y0 : y0 + th + 2 * m, x0 : x0 + tw + 2 * m], cv2.COLOR_BGR2GRAY)
        score = cv2.matchTemplate(highpass(area, fh), star, cv2.TM_CCOEFF_NORMED)
        return float(score.max())

    def text_there(self, frame: np.ndarray, lay: DialogueLayout) -> bool:
        """Are there strokes of the layout's dialogue (its colours, from the cache) in its
        glass area?"""
        rec = self.cache.dialogue_font(lay.name)
        if rec is None or lay.glass is None or "glyph" not in rec:
            return False
        fh, fw = frame.shape[:2]
        fill = tuple(rec.get("fill") or PLAIN_LIGHT)
        edge = tuple(rec["edge"]) if rec.get("edge") is not None else None
        got = text_mask(frame, px(lay.glass, fw, fh), rec["glyph"] * fh, fill, edge)
        return got is not None and int(got[2].sum()) >= TEXT_MIN

    def _star(self, div: Divider, fh: int) -> np.ndarray | None:
        """The divider's picture at this frame height, its fine structure only."""
        key = (div.template, fh)
        if key not in self._stars:
            img = read_grey(GAMES_DIR / self.profile.game / div.template)
            if img is None:
                log.warning("divider picture missing: %s", div.template)
            else:
                s = fh / float(div.height)
                if abs(s - 1.0) > 0.01:
                    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                img = highpass(img, fh)
            self._stars[key] = img
        return self._stars[key]


def highpass(grey: np.ndarray, fh: int) -> np.ndarray:
    """A picture's fine structure (its own blur taken away): the same whatever lies behind."""
    g = grey.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), HIGHPASS * fh / REFERENCE_H)


def read_grey(path: Path) -> np.ndarray | None:
    """A picture from any folder (cv2.imread fails on paths with accents or kana)."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
