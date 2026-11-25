"""Frames of the game's window, captured on a thread of their own.

The platform's grabber does the capturing (hud.platform): on Windows, Desktop Duplication
of the monitor the window is on (mss across monitors); on X11, the game window's own
pixels. Either way the overlay and the app's panel are never in a frame, which keeps the
HUD from reading its own English.

Capturing runs on its own thread and the HUD takes the newest frame: recovering from a
lost display (a UAC prompt, the lock screen, sleep, a monitor switched off) can wait
seconds, sometimes for good, and must never hold the GUI thread. A grabber that keeps
failing is made anew. The area captured is the whole game window; a smaller part of it
(the picture between black bars) is a crop of the same frame, one object per frame, so an
unchanged screen is the same frame to the HUD; every new frame raises :attr:`seq`.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from hud.platform import Grabber, Rect, current

log = logging.getLogger(__name__)

FPS = 30.0  # frames a second while nothing asks for more
FAST_FPS = 60.0  # ...while frosted glass is shown (it is a blur of the scene, made per frame)
RECREATE_AFTER = 30  # failed grabs in a row before the grabber is made anew
IDLE_S = 0.25  # how long the thread naps while there is nothing to capture
LOG_FAILURE_EVERY = 100  # a failing grab is logged this rarely


def _inside(r: Rect, area: Rect) -> bool:
    x, y, w, h = r
    ax, ay, aw, ah = area
    return ax <= x and ay <= y and x + w <= ax + aw and y + h <= ay + ah


class ScreenSource:
    """The newest frame of the game window's area, for any thread to take."""

    def __init__(self, backend: str = "auto", start: bool = True) -> None:
        self.want = backend  # auto | dxcam | mss (Windows); auto (X11)
        self.backend = "none"  # what grabbed last
        self.fps = FPS
        self.fail_count = 0  # failed grabs in a row
        self.seq = 0  # one more for every new frame
        self._area: Rect | None = None  # what the thread captures
        self._target: int | None = None  # the game window (systems that read it itself)
        self._frame: np.ndarray | None = None
        self._frame_area: Rect | None = None
        self._ok_at = time.monotonic()  # the last grab that worked (a frame, or no change)
        self._crops: dict[Rect, tuple[int, np.ndarray]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
            self._thread.start()

    # ---- the HUD's side (any thread) -----------------------------------------------------
    def set_area(self, rect: Rect | None, target: int | None = None) -> None:
        """What to capture: the game window's client area (None: nothing) and the window
        (its id, hud.platform.WindowRef.id)."""
        if rect is not None and (rect[2] <= 0 or rect[3] <= 0):
            rect = None
        with self._lock:
            if rect != self._area or target != self._target:
                self._area, self._target = rect, target
                self._wake.set()

    def grab(self, rect: Rect) -> np.ndarray | None:
        """The newest frame of *rect* (screen pixels): the whole area or a part of it. A
        rect outside the area becomes the area: its frames come from the next round on."""
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return None
        with self._lock:
            if self._area is None or not _inside(rect, self._area):
                self._area = rect
                self._wake.set()
            frame, farea, seq = self._frame, self._frame_area, self.seq
            if frame is None or farea is None or not _inside(rect, farea):
                return None
            if rect == farea:
                return frame
            cached = self._crops.get(rect)
            if cached is not None and cached[0] == seq:
                return cached[1]
        fx, fy = farea[0], farea[1]
        crop = np.ascontiguousarray(frame[y - fy : y - fy + h, x - fx : x - fx + w])
        with self._lock:
            if self.seq == seq:
                self._crops[rect] = (seq, crop)
        return crop

    def age(self) -> float:
        """Seconds since the last grab that worked: the screen may be unchanged, but if
        this grows, nothing can be captured (a lost display, a window off every screen)."""
        return time.monotonic() - self._ok_at

    def close(self) -> None:
        self._halt.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---- the capture thread ------------------------------------------------------------
    def _run(self) -> None:
        grabber: Grabber = current().grabber(self.want)
        try:
            while not self._halt.is_set():
                with self._lock:
                    area, target = self._area, self._target
                if area is None:
                    self._wake.wait(IDLE_S)
                    self._wake.clear()
                    continue
                t0 = time.perf_counter()
                self._grab_once(grabber, area, target)
                rest = 1.0 / max(1.0, self.fps) - (time.perf_counter() - t0)
                if rest > 0:
                    self._wake.wait(rest)
                    self._wake.clear()
        finally:
            grabber.close()

    def _grab_once(self, grabber: Grabber, area: Rect, target: int | None) -> None:
        try:
            frame, ok = grabber.grab(area, target)
        except Exception as exc:  # noqa: BLE001 - counted, logged, retried
            frame, ok = None, False
            if self.fail_count % LOG_FAILURE_EVERY == 0:
                log.warning("capture failed (%s): %s", grabber.backend, exc)
        self.backend = grabber.backend
        if not ok:
            self.fail_count += 1
            if self.fail_count % RECREATE_AFTER == 0:
                log.info("capture keeps failing: the grabber is made anew")
                grabber.reset()
            return
        self.fail_count = 0
        self._ok_at = time.monotonic()
        if frame is None:
            return  # nothing changed since the last grab
        with self._lock:
            if area == self._area:
                self._frame, self._frame_area = frame, area
                self.seq += 1
                self._crops.clear()
