"""Boards: the live covers of one owner (the UI pipeline; the dialogue band's labels),
shared between the OCR thread that places them and the pixel watch that removes them.

* the OCR places covers (:mod:`hud.pipeline`), and so does memory, for text covered before
  that is on screen again (:mod:`hud.recall`); only the watch removes them: every 50 ms
  the Japanese strokes under each cover are compared with how they looked when it was
  placed (:mod:`hud.anchors`); when they vanish or change, the cover goes at once and its
  area is handed back to the OCR to be read again. (A cover memory brought back gives way
  when the OCR reads another line there.)
* a cover never moves and never changes shape while it is shown
"""

from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass

import numpy as np

from hud.anchors import HARD_FAIL, HOLDS, Anchor, anchor_state, why
from hud.ink import Rect
from hud.items import OverlayItem
from hud.layout import Geom, intersects
from hud.looks import PLAIN_LIGHT
from hud.style import CoverStyle

log = logging.getLogger("hud")

DOUBT_FRAMES = 2  # consecutive doubtful frames before a cover goes (a plain loss: one)
LIST_MOVED = 3  # covers of one column gone at once: the list moved (_list_mates)
ERASE_MARGIN = 1  # pixels around the cover that are painted over too (the outline's edge)

_ids = itertools.count(1)


@dataclass
class Cover:
    template: str  # the line's translation with its placeholders: what identifies the line
    text: str  # what is shown (placeholders filled in)
    geom: Geom
    anchor: Anchor
    placed: float = 0.0  # when it was placed
    cached: bool = False  # settled: its block is in the permanent cache (or never will be)
    id: str = ""
    doubtful: int = 0
    light: bool | None = None  # the polarity of its Japanese...
    cells: list[tuple[int, int]] | None = None  # ...and its characters' columns
    recalled: bool = False  # brought back from memory, not yet read by the OCR
    story: bool = False  # dialogue, narration, a choice (hud.content), not a menu label

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(next(_ids))


class Board:
    def __init__(self, name: str, style: CoverStyle) -> None:
        self.name = name
        self.style = style
        self.lock = threading.Lock()
        self.covers: dict[str, Cover] = {}
        self.version = 0  # bumped on every visible change
        self.provisional: set[str] = set()  # lines translated live: shown a little faded
        self.rtl = False  # the translation language is written right to left

    # ---- read side -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.covers)

    def snapshot(self) -> list[Cover]:
        with self.lock:
            return list(self.covers.values())

    def items(self) -> list[OverlayItem]:
        with self.lock:
            return [self._item(c) for c in self.covers.values()]

    def _item(self, c: Cover) -> OverlayItem:
        g = c.geom
        x, y, w, h = g.cover
        m = ERASE_MARGIN
        return OverlayItem(
            rect=g.box,
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
            erase=(x - m, y - m, w + 2 * m, h + 2 * m),  # the Japanese, its outline included
            under=g.cover,
            glow=self.style.label,
            shine=self.style.shine,
            rtl=self.rtl,
            id=f"{self.name}:{c.id}",
            provisional=c.template in self.provisional,
        )

    def find(self, template: str, cover: Rect) -> Cover | None:
        """The live cover of this line at this spot."""
        with self.lock:
            for c in self.covers.values():
                if c.template == template and intersects(c.geom.cover, cover):
                    return c
        return None

    def taken(self, box: Rect) -> bool:
        """Is any part of *box* already drawn on?"""
        with self.lock:
            return any(intersects(c.geom.box, box) for c in self.covers.values())

    def boxes(self) -> list[Rect]:
        with self.lock:
            return [c.geom.box for c in self.covers.values()]

    def clashing(self, box: Rect) -> list[Cover]:
        with self.lock:
            return [c for c in self.covers.values() if intersects(c.geom.box, box)]

    # ---- write side ------------------------------------------------------------------
    def add_if_free(self, cover: Cover, others: list["Board"] = ()) -> bool:
        """Add *cover* unless its box is drawn on already (here or on *others*)."""
        if any(b.taken(cover.geom.box) for b in others):
            return False
        with self.lock:
            if any(intersects(c.geom.box, cover.geom.box) for c in self.covers.values()):
                return False
            self.covers[cover.id] = cover
            self.version += 1
        if not cover.recalled:
            log.info(
                "place: %s at %s font %d (%s)",
                cover.text[:40], cover.geom.box, cover.geom.font_px, self.name,
            )  # fmt: skip
        return True

    def retext(self, cover: Cover, text: str, geom: Geom, anchor: Anchor) -> None:
        """The same line with other numbers (a counter, a timer): its text and its strokes
        are the new ones, the cover stays (no fade: the text changes in place)."""
        with self.lock:
            c = self.covers.get(cover.id)
            if c is None:
                return
            c.text, c.geom, c.anchor = text, geom, anchor
            self.version += 1
        log.info("retext: %s at %s (%s)", text[:40], geom.box, self.name)

    def remove(self, covers: list[Cover], reason: str) -> list[Rect]:
        with self.lock:
            gone = [c for c in covers if self.covers.pop(c.id, None) is not None]
            if gone:
                self.version += 1
        for c in gone:
            log.info("gone: %s (%s) [%s]", c.text[:40], self.name, reason)
        return [c.geom.box for c in gone]

    def clear(self) -> list[Rect]:
        with self.lock:
            freed = [c.geom.box for c in self.covers.values()]
            if self.covers:
                self.covers.clear()
                self.version += 1
            return freed

    def watch(self, frame: np.ndarray) -> list[Rect]:
        """Remove every cover whose Japanese is no longer under it; returns the freed
        boxes (to be read again). A plain disappearance goes on the first frame, a doubtful
        change on the second."""
        gone: list[Cover] = []
        reasons: dict[str, str] = {}
        with self.lock:
            for c in self.covers.values():
                state = anchor_state(frame, c.anchor)
                if state == HOLDS:
                    c.doubtful = 0
                    continue
                c.doubtful += 1
                if state == HARD_FAIL or c.doubtful >= DOUBT_FRAMES:
                    gone.append(c)
                    reasons[c.id] = why(frame, c.anchor)
            for c in _list_mates(gone, self.covers.values()):
                gone.append(c)
                reasons[c.id] = "its list moved"
            for c in gone:
                del self.covers[c.id]
            if gone:
                self.version += 1
        for c in gone:
            log.info("gone: %s (%s) [%s]", c.text[:40], self.name, reasons[c.id])
        return [c.geom.box for c in gone]


def _list_mates(gone: list[Cover], left: object) -> list[Cover]:
    """Covers that go with a list that moved: when LIST_MOVED or more covers of one column
    (their left edges within half a glyph) went at once, every other cover of that column
    between them goes too. Its own strokes may still pass for its text in a moment, when
    the row that scrolled in looks much like it (a list of "...品質"), and it would stay
    behind over the wrong line."""
    out: list[Cover] = []
    rest = [c for c in left if c not in gone]
    for c0 in gone:
        x0, _y, _w, h0 = c0.geom.box
        tol = max(6.0, h0 / 2)
        column = [c for c in gone if abs(c.geom.box[0] - x0) <= tol]
        if len(column) < LIST_MOVED:
            continue
        top = min(c.geom.box[1] for c in column)
        bottom = max(c.geom.box[1] + c.geom.box[3] for c in column)
        for c in rest:
            x, y, _w, h = c.geom.box
            if c not in out and abs(x - x0) <= tol and top <= y + h / 2 <= bottom:
                out.append(c)
    return out
