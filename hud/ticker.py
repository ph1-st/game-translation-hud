"""Marquee text: a slot the game scrolls its text through.

A long title may rest in its slot, cut off, then slide through the slot, clipped at both
ends, and start over (a quest tracker's chapter title). Covered line by line, every frame
of the slide is another fragment: the cover would come and go and change its translation.
A slot marked ``ticker: true`` in the game profile is handled here instead:

* its translation is the line read while the text rests in the slot (a reading taken while it
  slides holds a fragment and is never settled, so it is never taken)
* it is shown in a fixed box and left as it is while the text slides underneath; the
  Japanese is painted over in the title's colour across the whole slot as it moves
* it goes only when the slot holds no text in that colour any more (the tracker is hidden,
  another screen is up)
* while it is shown, the OCR leaves the slot alone (the slide would keep it reading) and
  looks again every few seconds: a new title at rest replaces the line
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from capture.profile import Profile, Region
from hud.anchors import make_anchor, settled
from hud.covers import Placement
from hud.ink import Rect, clip_rect, thin
from hud.items import OverlayItem
from hud.layout import Geom, intersects
from hud.looks import PLAIN_LIGHT
from hud.style import CoverStyle

log = logging.getLogger("hud")

GONE_S = 0.4  # the slot without text in the title's colour this long: the line goes
REREAD_S = 3.0  # while a line is shown, the OCR looks at the slot this often
PRESENT_DIST = 45  # RGB distance to the title's colour of a pixel of its strokes
PRESENT_MIN = 20  # stroke pixels in that colour that mean "the title is there"
PRESENT_EDGE = 10  # local contrast of a stroke pixel
ERASE_MARGIN = 2  # pixels around the slot that are painted over too


@dataclass
class Title:
    """The line a slot shows."""

    template: str
    text: str
    geom: Geom
    absent_since: float | None = None


class Ticker:
    def __init__(self, profile: Profile, style: CoverStyle) -> None:
        self.regions: list[Region] = profile.tickers
        self.style = style
        self.shown: dict[str, Title] = {}  # by slot
        self.rtl = False  # the translation language is written right to left
        self.read_at: dict[str, float] = {}
        self.version = 0
        self._lock = threading.Lock()

    # ---- where ---------------------------------------------------------------------
    def slots(self, fw: int, fh: int) -> list[tuple[str, Rect]]:
        out = []
        for r in self.regions:
            x0, y0, x1, y1 = r.to_pixels(fw, fh)
            out.append((r.name, (x0, y0, x1 - x0, y1 - y0)))
        return out

    def slot_of(self, rect: Rect, fw: int, fh: int) -> str | None:
        cx, cy = (rect[0] + rect[2] / 2) / fw, (rect[1] + rect[3] / 2) / fh
        for r in self.regions:
            if r.contains_point(cx, cy):
                return r.name
        return None

    def boxes(self) -> list[Rect]:
        """Where the slots' translations are drawn."""
        with self._lock:
            return [s.geom.box for s in self.shown.values()]

    def quiet(self, fw: int, fh: int, now: float) -> list[Rect]:
        """Slots the OCR leaves alone now: a line is shown there and no look is due."""
        with self._lock:
            return [
                rect
                for name, rect in self.slots(fw, fh)
                if name in self.shown and now - self.read_at.get(name, -1e9) < REREAD_S
            ]

    def due(self, fw: int, fh: int, now: float) -> list[Rect]:
        """Slots to read now although nothing changed: a line is shown, a look is due."""
        with self._lock:
            return [
                rect
                for name, rect in self.slots(fw, fh)
                if name in self.shown and now - self.read_at.get(name, -1e9) >= REREAD_S
            ]

    # ---- the OCR side (worker thread) ------------------------------------------------
    def offer(
        self,
        placements: list[Placement],
        read: np.ndarray,
        newest: np.ndarray,
        now: float,
        looked: list[Rect],
    ) -> None:
        """Readings of the slots. A line read at rest becomes the slot's line (the widest
        reading: the title as a whole); fragments read while it slides are not settled and
        change nothing. *looked*: the slots the OCR looked at in this tick."""
        fh, fw = read.shape[:2]
        best: dict[str, Placement] = {}
        for p in placements:
            name = self.slot_of(p.geom.cover, fw, fh)
            if name is None:
                continue
            ref = make_anchor(read, p.geom.cover, p.light, p.cells)
            if ref is None or (newest is not read and not settled(newest, ref)):
                continue
            if name not in best or p.geom.cover[2] > best[name].geom.cover[2]:
                best[name] = p
        with self._lock:
            for name, rect in self.slots(fw, fh):
                if any(intersects(rect, r) for r in looked):
                    self.read_at[name] = now
            for name, p in best.items():
                cur = self.shown.get(name)
                if cur is not None and cur.template == p.template:
                    continue
                self.shown[name] = Title(p.template, p.text, p.geom)
                self.read_at[name] = now
                self.version += 1
                log.info("ticker %s: %s at %s", name, p.text[:40], p.geom.box)

    # ---- the watch (GUI thread, every 50 ms) -------------------------------------------
    def watch(self, frame: np.ndarray, now: float) -> None:
        fh, fw = frame.shape[:2]
        with self._lock:
            for name, rect in self.slots(fw, fh):
                cur = self.shown.get(name)
                if cur is None:
                    continue
                if present(frame, rect, cur.geom.fg or PLAIN_LIGHT) >= PRESENT_MIN:
                    cur.absent_since = None
                    continue
                if cur.absent_since is None:
                    cur.absent_since = now
                elif now - cur.absent_since >= GONE_S:
                    del self.shown[name]
                    self.version += 1
                    log.info("ticker %s gone: %s", name, cur.text[:40])

    def clear(self) -> None:
        with self._lock:
            if self.shown:
                self.shown.clear()
                self.version += 1

    # ---- drawing -----------------------------------------------------------------------
    def items(self, fw: int, fh: int) -> list[OverlayItem]:
        out = []
        with self._lock:
            for name, rect in self.slots(fw, fh):
                cur = self.shown.get(name)
                if cur is None:
                    continue
                g = cur.geom
                x, y, w, h = rect
                m = ERASE_MARGIN
                out.append(
                    OverlayItem(
                        g.box,
                        lines=g.lines,
                        font_px=g.font_px,
                        row_top=g.row_top,
                        row_h=g.row_h,
                        pitch=g.pitch,
                        align=g.align,
                        stretch=g.stretch,
                        style="label",
                        fg=g.fg or PLAIN_LIGHT,
                        edge=g.edge,
                        # the whole slot: the Japanese slides through all of it
                        erase=(x - m, y - m, w + 2 * m, h + 2 * m),
                        gate=True,
                        under=g.cover,
                        glow=self.style.label,
                        shine=self.style.shine,
                        rtl=self.rtl,
                        id=f"ticker:{name}:{cur.template}",
                    )
                )
        return out


def present(frame: np.ndarray, rect: Rect, fill: tuple[int, int, int]) -> int:
    """Stroke pixels in the title's colour in *rect*: is the title (or part of it) there?"""
    fh, fw = frame.shape[:2]
    x, y, w, h = clip_rect(rect, fw, fh)
    if w < 4 or h < 4:
        return 0
    rgb = frame[y : y + h, x : x + w, ::-1].astype(np.float32)
    d = rgb - np.array(fill, np.float32)
    near = np.einsum("ijk,ijk->ij", d, d) < PRESENT_DIST * PRESENT_DIST
    grey = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(grey - cv2.GaussianBlur(grey, (0, 0), max(2.0, h / 3.0)))
    return int(thin(near & (hp > PRESENT_EDGE)).sum())
