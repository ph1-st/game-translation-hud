"""Grabbing an area of the screen with mss (any system). It sees whatever is on the screen,
so it serves where the overlay is hidden from screen capture (Windows) or where no game
window is captured at all (offline tools)."""

from __future__ import annotations

import numpy as np

from hud.platform.base import Grabber, Rect


class MssGrabber(Grabber):
    backend = "mss"

    def __init__(self) -> None:
        self._sct = None

    def grab(self, area: Rect, target: int | None = None) -> tuple[np.ndarray | None, bool]:
        if self._sct is None:
            import mss

            self._sct = mss.mss()
        x, y, w, h = area
        shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        return np.asarray(shot)[:, :, :3].copy(), True  # BGRA -> BGR

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # noqa: BLE001 - closing a broken capture is best effort
                pass
            self._sct = None
