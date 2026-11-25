"""Permanent standardization: the blocks of menu text are remembered for good.

A cover is keyed by its line (the translation template: OCR variants of one label are many
database keys with one translation) and its place. When the same line shows up in the same
area again, it gets exactly the block it had before (cover, drawing box, font size,
alignment, row layout), whatever the new reading's box looks like, including after it
briefly disappears and across restarts. A record is written once, when the cover is first
settled, and never changed. Story text (dialogue, narration, choices) is never kept here:
the same sentence is rarely seen twice in the same place.

The same file keeps the tables of the game's text sizes and colours (:mod:`hud.looks`) and
the fixed font and colours of each dialogue layout and speaker name, so those never change
either. Lines are kept by last use, up to :data:`MAX_LINES`. Geometry is relative to the
frame size. Persisted to ``covers.json`` in the game's cache folder (debounced: at most
every :data:`SAVE_EVERY_S`, and at stop); a file written by another version of the HUD is
ignored (its boxes came from other measuring rules).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

from hud.ink import Rect
from hud.layout import Geom
from hud.looks import Looks, Palette

log = logging.getLogger("hud")

VERSION = 4
MAX_PER_LINE = 8  # distinct places one line is remembered at
MAX_LINES = 5000  # lines remembered (the least recently used go first)
MAX_NAMES = 500  # speaker names whose colours are remembered
EXACT_PX = 2  # a cached block is reused in place when the text is this close to it
SAVE_EVERY_S = 5.0  # changes are written at most this often (and at stop)
SAME_ROW = 0.5  # same place: centres within this share of a row vertically...
SAME_COLUMN = 0.25  # ...this share of the width horizontally (at least a row height)...
SAME_AREA = (0.6, 1.65)  # ...and an area ratio within these


def same_place(a: list[float], b: list[float]) -> bool:
    """The same area of the screen (relative rects). The next row of a list or a table is
    another place."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    h = max(ah, bh, 1e-6)
    w = max(aw, bw, 1e-6)
    if abs((ay + ah / 2) - (by + bh / 2)) > SAME_ROW * h:
        return False
    if abs((ax + aw / 2) - (bx + bw / 2)) > max(SAME_COLUMN * w, h):
        return False
    lo, hi = SAME_AREA
    return lo <= (aw * ah) / max(1e-12, bw * bh) <= hi


class CoverCache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.covers: OrderedDict[str, list[dict]] = OrderedDict()  # least recently used first
        self.dialogue: dict[str, dict] = {}
        self.names: OrderedDict[str, list] = OrderedDict()  # speaker -> [fill, outline]
        self.looks = Looks()
        self.palette = Palette()
        self.dirty = False
        self.hits = 0
        self.misses = 0
        self._saved_at = time.monotonic()
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self.load()

    # ---- persistence -----------------------------------------------------------------
    def load(self) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict) or data.get("version") != VERSION:
            log.info("cover cache %s is from another HUD version: starting afresh", self.path)
            return
        self.covers = OrderedDict(data.get("covers", {}))
        self.dialogue = data.get("dialogue", {})
        self.names = OrderedDict(data.get("names", {}))
        self.looks = Looks(data.get("looks", []))
        self.palette = Palette(data.get("palette", []))
        n = sum(len(v) for v in self.covers.values())
        log.info("cover cache: %d covers for %d lines from %s", n, len(self.covers), self.path)

    @property
    def changed(self) -> bool:
        return self.dirty or self.looks.dirty or self.palette.dirty

    def save_soon(self) -> None:
        """Write the changes when the last write is a while ago (the OCR thread calls this
        on every tick; writing the whole file each time would cost a tick's time)."""
        if (
            self.path is not None
            and self.changed
            and time.monotonic() - self._saved_at >= SAVE_EVERY_S
        ):
            self.save()

    def save(self) -> None:
        """Write the changes now (at stop)."""
        if self.path is None or not self.changed:
            return
        with self._save_lock:  # (the OCR thread and a stopping app may both save)
            with self._lock:  # the text is made under the lock: one consistent picture
                text = json.dumps(
                    {
                        "version": VERSION,
                        "looks": self.looks.to_list(),
                        "palette": self.palette.to_list(),
                        "dialogue": self.dialogue,
                        "names": self.names,
                        "covers": self.covers,
                    },
                    ensure_ascii=False,
                )
                self.dirty = self.looks.dirty = self.palette.dirty = False
                self._saved_at = time.monotonic()
            tmp = self.path.with_suffix(".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(text, encoding="utf-8")
                tmp.replace(self.path)
            except OSError as exc:  # another HUD instance holds the file: try again later
                log.warning("could not save %s: %s", self.path, exc)
                self.dirty = True

    # ---- covers ----------------------------------------------------------------------
    @staticmethod
    def _rel(rect: Rect, fw: int, fh: int) -> list[float]:
        x, y, w, h = rect
        return [x / fw, y / fh, w / fw, h / fh]

    def find(self, template: str, cover: Rect, fw: int, fh: int) -> dict | None:
        """The remembered block of this line at this place (relative geometry)."""
        rel = self._rel(cover, fw, fh)
        with self._lock:
            for rec in self.covers.get(template, []):
                if same_place(rec["cover"], rel):
                    self.hits += 1
                    self.covers.move_to_end(template)
                    return rec
            self.misses += 1
        return None

    def diagnostics(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "saved": sum(len(v) for v in self.covers.values()),
            }

    def put(self, template: str, geom: Geom, fw: int, fh: int) -> bool:
        """Remember how this line was boxed here. Never overwrites: the first block at a
        place is the block. Returns True if stored."""
        rec = geom.to_rel(fw, fh)
        with self._lock:
            recs = self.covers.setdefault(template, [])
            self.covers.move_to_end(template)
            if any(same_place(old["cover"], rec["cover"]) for old in recs):
                return False
            recs.append(rec)
            del recs[:-MAX_PER_LINE]
            while len(self.covers) > MAX_LINES:
                self.covers.popitem(last=False)
            self.dirty = True
        return True

    # ---- dialogue layouts ----------------------------------------------------------------
    def dialogue_font(self, layout: str) -> dict | None:
        with self._lock:
            return self.dialogue.get(layout)

    def set_dialogue_font(self, layout: str, rec: dict) -> dict:
        """The first font measured for a dialogue layout is its font for good."""
        with self._lock:
            if layout not in self.dialogue:
                self.dialogue[layout] = rec
                self.dirty = True
            return self.dialogue[layout]

    def set_dialogue_name(self, layout: str, rec: dict) -> dict:
        """The first speaker-name box measured for a layout is its box for good."""
        with self._lock:
            lay = self.dialogue.setdefault(layout, {})
            if "name" not in lay:
                lay["name"] = rec
                self.dirty = True
            return lay["name"]

    def set_dialogue_field(self, layout: str, key: str, value) -> dict:
        """A further fixed value of a dialogue layout (set once, never changed)."""
        with self._lock:
            lay = self.dialogue.setdefault(layout, {})
            if key not in lay:
                lay[key] = value
                self.dirty = True
            return lay

    def name_colours(self, name: str, colours: list | None = None) -> list | None:
        """The colours of a speaker's name ([fill, outline]): the first ones read are its
        colours for good. Without *colours*, only looks them up."""
        with self._lock:
            if name not in self.names and colours is not None:
                self.names[name] = colours
                while len(self.names) > MAX_NAMES:
                    self.names.popitem(last=False)
                self.dirty = True
            elif name in self.names:
                self.names.move_to_end(name)
            return self.names.get(name)


def place(rec: dict, extent_cover: Rect, lines: list[str], fw: int, fh: int) -> Geom:
    """A cached block for text measured at *extent_cover*: exactly as remembered when the
    text is where it was (a pixel or two of measuring noise), moved along with the text
    when it has shifted a little, never reshaped."""
    g = Geom.from_rel(rec, lines, fw, fh)
    cx, cy, cw, ch = g.cover
    ex, ey, ew, eh = extent_cover
    if g.align == "left":
        dx = ex - cx
    elif g.align == "right":
        dx = (ex + ew) - (cx + cw)
    else:
        dx = int(round((ex + ew / 2) - (cx + cw / 2)))
    dy = int(round((ey + eh / 2) - (cy + ch / 2)))
    if abs(dx) <= EXACT_PX and abs(dy) <= EXACT_PX:
        return g
    return g.moved(dx, dy)
