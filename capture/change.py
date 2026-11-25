"""What changed on screen, cheaply: frames compared at a small size in grey.

The OCR only reads where something changed, which is what keeps it fast: most frames are
identical to the last one read from the OCR's point of view.

:class:`ChangeTracker` keeps a small grey picture of the screen as it looked when each part
was last read. The HUD marks just the areas it read (:meth:`ChangeTracker.mark_read`), so
an area stays changed until it is read; the capture tool reads every changed region at once
and compares consecutive frames (:meth:`ChangeTracker.update`). Areas the HUD's pixel watch
freed (a cover went) are handed over from its thread (:meth:`ChangeTracker.free`).
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from capture.profile import Region

Rect = tuple[int, int, int, int]
SMALL_W = 320  # width of the change map
CHANGE_THRESHOLD = 24  # grey levels on the change map that count as changed
NOISE_KERNEL = 2  # a changed speck smaller than this many map pixels is compression noise
REST_SHARE = 0.002  # a picture with less of its map changed than this is at rest (a cursor
# blinking, a sparkle: not a list scrolling)


class ChangeTracker:
    def __init__(self, width: int = SMALL_W, threshold: int = CHANGE_THRESHOLD) -> None:
        self.width = width
        self.threshold = threshold
        self.ref: np.ndarray | None = None  # the screen as each part looked when last read
        self.unread: np.ndarray | None = None  # parts not read since the reference was made
        self._freed: list[Rect] = []
        self._lock = threading.Lock()
        self._forget = False  # asked for from another thread: everything counts as changed
        self._last: tuple[np.ndarray, np.ndarray] | None = None  # (frame, its small copy)
        self._looked: np.ndarray | None = None  # the small copy at the last at_rest()

    def small(self, frame: np.ndarray) -> np.ndarray:
        """The frame in grey at the map's width (made once per frame)."""
        if self._last is not None and self._last[0] is frame:
            return self._last[1]
        h, w = frame.shape[:2]
        sh = max(1, int(round(h * self.width / w)))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (self.width, sh), interpolation=cv2.INTER_AREA)
        self._last = (frame, small)
        return small

    def reset(self) -> None:
        """Everything counts as changed from the next look on (any thread: the reference is
        dropped by the reading thread itself, which alone reads and writes it)."""
        with self._lock:
            self._forget = True

    def changed(self, frame: np.ndarray) -> np.ndarray:
        """Per map pixel: has it changed since it was last read?"""
        cur = self.small(frame)
        with self._lock:
            if self._forget:
                self.ref, self._forget = None, False
        if self.ref is None or self.ref.shape != cur.shape or self.unread is None:
            return np.ones(cur.shape, dtype=bool)
        return self._differ(cur, self.ref) | self.unread

    def at_rest(self, frame: np.ndarray) -> bool:
        """Is the picture as it was at the last look (a menu that stopped scrolling)? Each
        call is a look; the first one (at this size) is at rest: nothing says it moves."""
        cur = self.small(frame)
        before, self._looked = self._looked, cur
        if before is None or before.shape != cur.shape:
            return True
        return before is cur or float(self._differ(cur, before).mean()) < REST_SHARE

    def _differ(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = (cv2.absdiff(a, b) > self.threshold).astype(np.uint8)
        kernel = np.ones((NOISE_KERNEL, NOISE_KERNEL), np.uint8)
        return cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel) > 0

    def mark_read(self, frame: np.ndarray, rects: list[Rect] | None, keep: list[Rect] = ()) -> None:
        """The pixels of *frame* in *rects* (None: everything) are what was just read;
        *keep* rects stay as they were: read by someone else, or left for later (until they
        are read, they count as changed)."""
        cur = self.small(frame)
        fh, fw = frame.shape[:2]
        sx, sy = cur.shape[1] / fw, cur.shape[0] / fh
        if self.ref is None or self.ref.shape != cur.shape or self.unread is None:
            self.ref = cur.copy()
            self.unread = np.ones(cur.shape, dtype=bool)  # nothing read against it yet
        old, old_unread = self.ref.copy(), self.unread.copy()
        if rects is None:
            self.ref = cur.copy()
            self.unread[:] = False
        else:
            for x, y, w, h in rects:
                ys, xs = _map_slices(x, y, w, h, sx, sy)
                self.ref[ys, xs] = cur[ys, xs]
                self.unread[ys, xs] = False
        for x, y, w, h in keep:
            ys, xs = _map_slices(x, y, w, h, sx, sy)
            self.ref[ys, xs] = old[ys, xs]
            self.unread[ys, xs] = old_unread[ys, xs]

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """What changed since the last frame given here (the capture tool reads every
        changed region of every frame); None for the first frame."""
        first = self.ref is None
        changed = None if first else self.changed(frame)
        self.mark_read(frame, None)
        return changed

    @staticmethod
    def changed_fraction(changed: np.ndarray | None, region: Region | None = None) -> float:
        """The share of the map (in *region*) that changed; everything when unknown."""
        if changed is None:
            return 1.0
        if region is not None:
            changed = region.crop(changed)
        if changed.size == 0:
            return 0.0
        return float(changed.mean())

    # ---- areas the pixel watch freed (another thread) ------------------------------------
    def free(self, rects: list[Rect]) -> None:
        if rects:
            with self._lock:
                self._freed.extend(rects)

    def has_freed(self) -> bool:
        with self._lock:
            return bool(self._freed)

    def take_freed(self) -> list[Rect]:
        with self._lock:
            out, self._freed = self._freed, []
            return out


def _map_slices(x: int, y: int, w: int, h: int, sx: float, sy: float) -> tuple[slice, slice]:
    """The rows and columns of the change map that cover a frame rect."""
    x0, y0 = max(0, int(x * sx)), max(0, int(y * sy))
    x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
    return slice(y0, y1), slice(x0, x1)

