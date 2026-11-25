"""Menus come back from memory: text covered before is recognized by its pixels the moment
it is on screen again, without waiting for the OCR.

Menus are visited over and over (tabs clicked back and forth, the bag opened again): their
labels are always the same text in the same place. Every menu cover placed is remembered
with its anchor (the stroke pixels of its Japanese, :mod:`hud.anchors`); story text
(dialogue, narration, choices) never is: a sentence is rarely seen twice at the same place,
and remembering it would push the menus out. Every 50 ms a few of the stroke pixels of every
remembered cover are looked up in the newest frame at once (one array lookup, whatever the
number of covers remembered); the few covers whose text may be there are checked strictly:
the strokes are at the same pixels with the same values, every character's own strokes
too, and nothing new stands beside the text (a longer line that starts the same, "Details
1" -> "Details 12", is not the remembered one). A cover comes back once it matched on
:attr:`Recall.passes_required` frames in a row (two in the balanced mode: a list scrolling
past a remembered row must not flash its translation).

The OCR still reads the area: it confirms the cover, or, reading another line there, takes
the place over (memory never outranks a reading). The pixel watch removes recalled covers
like any other.

What was learned is kept for the next session (``recall.npz`` next to the cover cache),
least recently used first, so the oldest go when the memory is full. A remembered cover
only ever matches its own pixels at the same window size: after a game update or another
window size it simply never comes back.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hud.anchors import Anchor, make_anchor, recognizes
from hud.board import Board, Cover
from hud.covercache import VERSION as CACHE_VERSION
from hud.ink import Rect, clip_rect, crop_gray, derived, grey, polarity_masks, thin
from hud.layout import Geom, intersects

log = logging.getLogger("hud")

SIDE_GAIN = 6  # stroke pixels beside the text beyond what was there: something new stands
SIG_SCALE = 4  # the change gate looks at the screen at a quarter of its size
SIG_DELTA = 3.0  # grey levels of change (on that copy) that wake a remembered cover up
PASSES = 2  # frames in a row a remembered cover must match before it comes back
PROBE = 64  # stroke pixels looked at first: most remembered covers are not on screen
PROBE_DELTA = 36  # their mean change beyond which a cover is plainly not there
MAX_MEMORY = 4000  # covers remembered (the least recently used are forgotten first)
FORMAT = 3  # of the saved memory (with the cover cache's version: its geometry rules)
INDEX_EVERY_S = 2.0  # covers learned since are looked for after a rebuild at most this late
STRICT_MAX = 6  # remembered covers checked strictly per frame (a white flash wakes many)
_PACKS = {  # kind -> dtype of the packed arrays
    "g": np.uint8,  # anchor grey crops
    "s": np.bool_,  # anchor stroke masks
    "r": np.int16,  # stroke grey values
    "n": np.int16,  # ...minus the background
    "p": np.int64,  # probe pixels (flat indices)
    "q": np.int16,  # their grey values
    "c": np.bool_,  # per character: its share of the strokes
}


@dataclass
class Memory:
    template: str
    text: str
    geom: Geom
    anchor: Anchor
    board: str
    light: bool | None
    cells: list[tuple[int, int]] | None
    sides: tuple[int, int]  # text-polarity stroke pixels just left and right of the text
    frame_size: tuple[int, int]
    sig: np.ndarray | None = None  # what the place looked like when last checked
    passes: int = 0
    probe: np.ndarray | None = None  # a few of its stroke pixels (flat indices)...
    probe_vals: np.ndarray | None = None  # ...and their grey values


class Recall:
    def __init__(self) -> None:
        # by line, least recently used first
        self.memory: OrderedDict[str, list[Memory]] = OrderedDict()
        self._learned: set[str] = set()  # ids of the covers already looked at
        self._lock = threading.Lock()
        self.count = 0
        # the probe pixels of every remembered cover, for one lookup per frame
        self._index_for: tuple[int, int] | None = None  # frame size the index is for
        self._order: list[Memory] = []
        self._idx = np.zeros(0, np.int64)  # flat frame indices
        self._vals = np.zeros(0, np.int16)
        self._starts = np.zeros(0, np.int64)
        self._counts = np.zeros(0, np.float32)
        self._pending: dict[int, Memory] = {}  # covers that passed on the frame before
        self._stale = False  # covers learned since the index was built
        self._built_at = -1e9
        self._dirty = False  # learned since the last save
        self.passes_required = PASSES
        self.checks = 0
        self.hits = 0

    def learn(self, boards: list[Board], frame: np.ndarray) -> None:
        """Remember the menu covers placed since the last call, with what stands beside
        them."""
        fh, fw = frame.shape[:2]
        for b in boards:
            for c in b.snapshot():
                if c.id in self._learned:
                    continue
                self._learned.add(c.id)
                if c.story:
                    continue  # menus only
                mem = Memory(
                    c.template, c.text, c.geom, c.anchor, b.name, c.light, c.cells,
                    sides(frame, c.geom.cover, c.light), (fw, fh),
                )  # fmt: skip
                flat = np.flatnonzero(c.anchor.strokes)
                pick = np.linspace(0, len(flat) - 1, min(PROBE, len(flat))).astype(int)
                mem.probe, mem.probe_vals = flat[pick], c.anchor.raw_vals[pick]
                with self._lock:
                    self._keep(mem)

    def _keep(self, mem: Memory) -> None:
        """Remember *mem* as the newest picture of its line at its place (under the lock)."""
        same = self.memory.setdefault(mem.template, [])
        self.memory.move_to_end(mem.template)
        for i, old in enumerate(same):
            if old.board == mem.board and intersects(old.geom.cover, mem.geom.cover):
                same[i] = mem
                break
        else:
            same.append(mem)
            self.count += 1
        while self.count > MAX_MEMORY and self.memory:
            _template, gone = self.memory.popitem(last=False)
            self.count -= len(gone)
        self._stale = True  # the lookup is built again (soon: not every time)
        self._dirty = True

    def _index(self, fw: int, fh: int) -> None:
        with self._lock:
            entries = [
                m
                for ms in self.memory.values()
                for m in ms
                if m.frame_size == (fw, fh) and m.probe is not None and len(m.probe)
            ]
        idx, vals, starts, counts = [], [], [], []
        n = 0
        for m in entries:
            x, y, _w, _h = clip_rect(m.anchor.rect, fw, fh)
            rows, cols = np.divmod(m.probe, m.anchor.gray.shape[1])
            idx.append((y + rows).astype(np.int64) * fw + (x + cols))
            vals.append(m.probe_vals)
            starts.append(n)
            counts.append(len(m.probe))
            n += len(m.probe)
        self._order = entries
        self._idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        self._vals = np.concatenate(vals) if vals else np.zeros(0, np.int16)
        self._starts = np.array(starts, np.int64)
        self._counts = np.array(counts, np.float32)
        self._index_for = (fw, fh)
        self._stale = False
        kept = {id(m) for m in entries}
        for key in [k for k in self._pending if k not in kept]:
            self._pending.pop(key).passes = 0

    def bring_back(
        self,
        frame: np.ndarray,
        now: float,
        boards: list[Board],
        keep_out: list[Rect],
        allow: Callable[[Memory], bool] | None = None,
    ) -> int:
        """Place the remembered covers whose text is on screen again. *keep_out*: areas
        no cover may be recalled into (the glass of a running dialogue scene); *allow*
        turns down memories the player switched off."""
        fh, fw = frame.shape[:2]
        if self._index_for != (fw, fh) or (self._stale and now - self._built_at >= INDEX_EVERY_S):
            self._index(fw, fh)
            self._built_at = now
        if not self._order:
            return 0
        by_name = {b.name: b for b in boards}
        shown: dict[tuple[str, str], list[Rect]] = {}
        for b in boards:
            for c in b.snapshot():
                shown.setdefault((b.name, c.template), []).append(c.geom.cover)
        small = derived(frame, "recall-small", lambda f: _small(grey(f)))
        placed = strict = 0
        for m in self._near(frame):
            board = by_name.get(m.board)
            if board is None or (allow is not None and not allow(m)):
                continue
            if not self._worth_a_look(m, small, shown.get((m.board, m.template), ()), keep_out):
                continue
            if strict >= STRICT_MAX:  # the rest are looked at on the next frames
                m.sig = None
                continue
            strict += 1
            if self._matches(frame, m) and self._place(m, board, boards, frame, now):
                placed += 1
        return placed

    def _near(self, frame: np.ndarray) -> list[Memory]:
        """The remembered covers whose probe pixels are about as they were (one array
        lookup for all of them), the likeliest first. One that passed on the frame before
        but is not near now starts over."""
        d = np.abs(grey(frame).ravel()[self._idx].astype(np.int16) - self._vals)
        means = np.add.reduceat(d.astype(np.int32), self._starts) / self._counts
        near = np.flatnonzero(means <= PROBE_DELTA)
        out = [self._order[i] for i in near[np.argsort(means[near], kind="stable")]]
        ids = {id(m) for m in out}
        for key in [k for k in self._pending if k not in ids]:
            self._pending.pop(key).passes = 0
        return out

    def _worth_a_look(
        self, m: Memory, small: np.ndarray, shown: list[Rect] | tuple, keep_out: list[Rect]
    ) -> bool:
        """The cheap gates before the strict check: not shown already, something changed
        at its place since it was last looked at, not where no cover may go."""
        if any(intersects(r, m.geom.cover) for r in shown):
            m.sig = None  # on screen: looked at again once it goes
            self._drop(m)
            return False
        sig = _sig(small, m.geom.cover)
        if sig is None:
            return False
        if m.passes == 0 and m.sig is not None and m.sig.shape == sig.shape:
            if float(np.abs(sig - m.sig).mean()) < SIG_DELTA:
                return False  # nothing changed there since it was last checked
        m.sig = sig
        if any(intersects(m.geom.box, k) for k in keep_out):
            self._drop(m)
            return False
        return True

    def _matches(self, frame: np.ndarray, m: Memory) -> bool:
        """The strict check, frame after frame: True once *m*'s text has been recognized
        on :attr:`passes_required` frames in a row."""
        self.checks += 1
        if not recognizes(frame, m.anchor) or _crowded(frame, m):
            self._drop(m)
            return False
        m.passes += 1
        if m.passes < self.passes_required:
            self._pending[id(m)] = m
            return False
        self._drop(m)
        return True

    def _place(self, m: Memory, board: Board, boards: list[Board], frame, now: float) -> bool:
        anchor = make_anchor(frame, m.geom.cover, m.light, m.cells) or m.anchor
        cover = Cover(
            m.template, m.text, m.geom, anchor, placed=now, cached=True,
            light=m.light, cells=m.cells, recalled=True,
        )  # fmt: skip
        if not board.add_if_free(cover, [b for b in boards if b is not board]):
            return False
        self._learned.add(cover.id)
        with self._lock:
            if m.template in self.memory:
                self.memory.move_to_end(m.template)  # used: kept longest
        self.hits += 1
        log.info("recalled: %s at %s (%s)", m.text[:40], m.geom.box, m.board)
        return True

    def diagnostics(self) -> dict[str, int]:
        return {"remembered": self.count, "checks": self.checks, "hits": self.hits}

    def _drop(self, m: Memory) -> None:
        m.passes = 0
        self._pending.pop(id(m), None)

    # ---- across sessions -----------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Keep what was learned for the next session (only when something was)."""
        with self._lock:
            if not self._dirty:
                return
            entries = [m for ms in self.memory.values() for m in ms]
            self._dirty = False
        meta = []
        packs: dict[str, list[np.ndarray]] = {k: [] for k in _PACKS}
        for m in entries:
            a = m.anchor
            fw, fh = m.frame_size
            parts = {
                "g": a.gray, "s": a.strokes, "r": a.raw_vals, "n": a.norm_vals,
                "p": m.probe, "q": m.probe_vals,
                "c": np.stack(a.cells) if a.cells else None,
            }  # fmt: skip
            shapes = {}
            for k, arr in parts.items():
                if arr is None:
                    continue
                shapes[k] = list(arr.shape)
                packs[k].append(np.ascontiguousarray(arr, dtype=_PACKS[k]).ravel())
            meta.append(
                {
                    "template": m.template,
                    "text": m.text,
                    "board": m.board,
                    "light": m.light,
                    "cells": [list(c) for c in m.cells] if m.cells else None,
                    "sides": list(m.sides),
                    "frame": [fw, fh],
                    "geom": m.geom.to_rel(fw, fh),
                    "lines": m.geom.lines,
                    "rect": [int(v) for v in a.rect],
                    "anchor_light": a.light,
                    "local": float(a.local),
                    "shapes": shapes,
                }
            )
        arrays = {
            k: np.concatenate(parts) if parts else np.zeros(0, dtype)
            for (k, dtype), parts in zip(_PACKS.items(), packs.values(), strict=True)
        }
        doc = {"version": f"{CACHE_VERSION}.{FORMAT}", "entries": meta}
        arrays["meta"] = np.array(json.dumps(doc, ensure_ascii=False))
        tmp = path.with_name(path.stem + ".tmp.npz")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tmp, **arrays)
            tmp.replace(path)
        except OSError as exc:
            log.warning("could not save %s: %s", path, exc)
            self._dirty = True
            return
        log.info("recall memory: %d covers saved to %s", len(entries), path)

    def load(self, path: Path) -> None:
        """What an earlier session learned. A damaged file is set aside (and the HUD starts
        without it): it is only a memory."""
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                doc = json.loads(str(data["meta"]))
                if doc.get("version") != f"{CACHE_VERSION}.{FORMAT}":
                    log.info("recall memory %s is from another HUD version: not used", path)
                    return
                flat = {k: data[k] for k in _PACKS}
            memory, count = self._unpack(doc["entries"], flat)
        except Exception as exc:  # noqa: BLE001 - any damage: start without the memory
            log.warning("recall memory %s is damaged (%s): set aside", path, exc)
            try:
                path.replace(path.with_name(path.stem + ".bad.npz"))
            except OSError:
                pass
            return
        with self._lock:
            self.memory, self.count = memory, count
            self._index_for = None
        log.info("recall memory: %d covers from %s", count, path)

    @staticmethod
    def _unpack(entries: list[dict], flat: dict[str, np.ndarray]):
        offsets = dict.fromkeys(flat, 0)

        def take(kind: str, shape) -> np.ndarray | None:
            if shape is None:
                return None
            n = int(np.prod(shape))
            arr = flat[kind][offsets[kind] : offsets[kind] + n]
            if arr.size != n:
                raise ValueError(f"{kind}: {arr.size} values where {n} were saved")
            offsets[kind] += n
            return arr.reshape(shape)

        memory: OrderedDict[str, list[Memory]] = OrderedDict()
        count = 0
        for d in entries:
            fw, fh = d["frame"]
            sh = d["shapes"]
            cells = take("c", sh.get("c"))
            anchor = Anchor(
                tuple(d["rect"]), take("g", sh["g"]), take("s", sh["s"]), take("r", sh["r"]),
                take("n", sh["n"]), d["anchor_light"], d["local"],
                list(cells) if cells is not None else None,
            )  # fmt: skip
            m = Memory(
                d["template"], d["text"], Geom.from_rel(d["geom"], d["lines"], fw, fh),
                anchor, d["board"], d["light"],
                [tuple(c) for c in d["cells"]] if d["cells"] else None,
                tuple(d["sides"]), (fw, fh),
            )  # fmt: skip
            m.probe, m.probe_vals = take("p", sh["p"]), take("q", sh["q"])
            memory.setdefault(m.template, []).append(m)
            count += 1
        return memory, count


def sides(frame: np.ndarray, rect: Rect, light: bool | None) -> tuple[int, int]:
    """Text-polarity strokes in strips a glyph wide left and right of *rect*."""
    x, y, w, h = rect
    out = []
    for part in ((x - h, y, h, h), (x + w, y, h, h)):
        g = crop_gray(frame, part)
        if g is None:
            out.append(0)
            continue
        lm, dm = polarity_masks(g)
        if light is None:
            m = thin(lm) | thin(dm)
        else:
            m = thin(lm if light else dm)
        out.append(int(m.sum()))
    return out[0], out[1]


def _crowded(frame: np.ndarray, m: Memory) -> bool:
    """Does something stand beside the text now that did not when it was remembered (the
    line goes on: a longer text that starts the same)?"""
    left, right = sides(frame, m.geom.cover, m.light)
    ol, orr = m.sides
    return left > ol + max(SIDE_GAIN, ol // 2) or right > orr + max(SIDE_GAIN, orr // 2)


def _small(g: np.ndarray) -> np.ndarray:
    h, w = g.shape[:2]
    size = (max(1, w // SIG_SCALE), max(1, h // SIG_SCALE))
    return cv2.resize(g, size, interpolation=cv2.INTER_AREA)


def _sig(small: np.ndarray, rect: Rect) -> np.ndarray | None:
    x, y, w, h = (int(v) // SIG_SCALE for v in rect)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(small.shape[1], x + max(1, w) + 1), min(small.shape[0], y + max(1, h) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return small[y0:y1, x0:x1].astype(np.int16)
