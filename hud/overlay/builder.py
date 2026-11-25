"""Backdrops are built off the GUI thread: a dialogue area over a moving scene takes 5-20
ms, and the window's 50 ms tick must never wait for one."""

from __future__ import annotations

import logging
import threading

import numpy as np

from hud.items import OverlayItem
from hud.overlay.paint import build_layers

log = logging.getLogger("hud")


class BackdropBuilder(threading.Thread):
    """Builds backdrops. One request per item is kept, the newest: a frame that came in
    while the last one was being built replaces the older request."""

    def __init__(self) -> None:
        super().__init__(name="backdrops", daemon=True)
        self._cv = threading.Condition()
        self._jobs: dict[str, tuple[OverlayItem, np.ndarray | None]] = {}
        self._done: dict[str, tuple[OverlayItem, tuple | None]] = {}
        self._busy = False
        self._stopped = False

    def request(self, ident: str, item: OverlayItem, frame: np.ndarray | None) -> None:
        with self._cv:
            self._jobs[ident] = (item, frame)
            self._cv.notify()

    def forget(self, ident: str) -> None:
        with self._cv:
            self._jobs.pop(ident, None)
            self._done.pop(ident, None)

    def take(self) -> dict[str, tuple[OverlayItem, tuple | None]]:
        with self._cv:
            done, self._done = self._done, {}
            return done

    def idle(self) -> bool:
        with self._cv:
            return not self._jobs and not self._busy and not self._done

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify()

    def run(self) -> None:
        while True:
            with self._cv:
                while not self._jobs and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    return
                ident = next(iter(self._jobs))
                item, frame = self._jobs.pop(ident)
                self._busy = True
            try:
                built = build_layers(frame, item)
            except Exception:  # noqa: BLE001 - a bad frame must not end the thread
                log.exception("backdrop of %s (%s) failed", ident, " ".join(item.lines)[:40])
                built = None
            with self._cv:
                self._busy = False
                self._done[ident] = (item, built)
