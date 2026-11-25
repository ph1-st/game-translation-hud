"""Driving the HUD core without a window (tests and the offline review scripts).

* :class:`RecordingEngine` wraps the real OCR and keeps every call's result, keyed by the
  region it read; :class:`RecordedEngine` plays such a record back. A fixture is a frame
  plus that record, so a test runs the production path (lookup, extents, layout, cache,
  dialogue) on exactly what the OCR returned when the fixture was made, with no model.
* :func:`compose` paints overlay items onto a frame (offscreen Qt), for previews and
  replays, where no overlay window exists; the HUD is invisible to screen capture by
  design, so this is how its output is reviewed. (A live recording blends the window's
  own pixels instead: tools/recorder.py.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from capture.ocr import OcrLine, OcrResult
from capture.profile import Region

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "frames"


def region_key(region: Region | None) -> str:
    if region is None:
        return "full"
    return ",".join(f"{v:.4f}" for v in (region.x, region.y, region.w, region.h))


class RecordingEngine:
    """The real engine, with every result kept."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.device = engine.device
        self.calls: dict[str, list[dict]] = {}

    def warm_up(self) -> None:
        self.engine.warm_up()

    def read(
        self, frame: np.ndarray, region: Region | None = None, min_conf: float = 0.0,
        fine: bool = False,
    ) -> OcrResult:
        res = self.engine.read(frame, region=region, min_conf=min_conf, fine=fine)
        self.calls[region_key(region)] = [ln.as_dict() for ln in res.lines]
        return res


class RecordedEngine:
    """OCR results played back from a record; a region that was never read is an error
    (the fixture has to be made again with tools/make_fixtures.py)."""

    device = "recorded"

    def __init__(self, calls: dict[str, list[dict]]) -> None:
        self.calls = calls

    def warm_up(self) -> None:
        pass

    def read(
        self, frame: np.ndarray, region: Region | None = None, min_conf: float = 0.0,
        fine: bool = False,
    ) -> OcrResult:
        key = region_key(region)
        if key not in self.calls:
            raise KeyError(f"no recorded OCR for region {key}; rebuild the fixture")
        lines = [OcrLine.from_dict(d) for d in self.calls[key]]
        return OcrResult([ln for ln in lines if ln.conf >= min_conf], 0.0)


def load_fixture(game: str, name: str) -> tuple[np.ndarray, dict]:
    import cv2

    folder = FIXTURES / game
    frame = cv2.imread(str(folder / f"{name}.png"))
    if frame is None:
        raise FileNotFoundError(folder / f"{name}.png")
    with open(folder / f"{name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    return frame, meta


class Animator:
    """What the window animates, for a replay: items fading in (and out, when they say
    so) and English typing out, step by step in the replay's time."""

    def __init__(self) -> None:
        self.anims: dict[str, dict] = {}  # identity -> item, alpha, share, leaving
        self.last: float | None = None

    def step(self, items: list, now: float) -> list[tuple]:
        """(item, fade, share) of what shows at *now*, in the order it is painted."""
        from hud.overlay import FADE_IN_S, reveal_target

        dt = 0.0 if self.last is None else max(0.0, now - self.last)
        self.last = now
        wanted = {it.identity(): it for it in items}
        for ident, a in list(self.anims.items()):
            if ident not in wanted:
                if a["item"].fade_out > 0 and a["alpha"] > 0:
                    a["leaving"] = True
                else:
                    del self.anims[ident]
        for ident, it in wanted.items():
            a = self.anims.get(ident)
            if a is None:
                share = 0.0 if it.reveal is not None else 1.0
                a = self.anims[ident] = {"alpha": 0.0, "share": share, "fresh": True}
            a["item"], a["leaving"] = it, False
        out = []
        for ident, a in list(self.anims.items()):
            it = a["item"]
            if a["leaving"]:
                a["alpha"] -= dt / it.fade_out
                if a["alpha"] <= 0:
                    del self.anims[ident]
                    continue
            elif a.pop("fresh", False):
                pass  # it starts to fade in from the frame after it came
            else:
                a["alpha"] = min(1.0, a["alpha"] + dt / (it.fade_in or FADE_IN_S))
            a["share"] = max(a["share"], reveal_target(it, now))
            out.append((it, a["alpha"], a["share"]))
        return out


def compose(
    frame: np.ndarray,
    items: list,
    now: float | None = None,
    animator: Animator | None = None,
    backdrop=None,
) -> np.ndarray:
    """The frame with the overlay items painted on it (BGR), their backdrops built from
    the same frame (or by *backdrop(item)*: from an older one, as the window has them).
    With an *animator* (and *now*), items fade and English types out as in the window;
    without, everything shows as it ends up. Needs a QGuiApplication."""
    import cv2
    from PySide6.QtGui import QImage, QPainter

    from hud.overlay import build_backdrop, paint_all, qimage, render_text, revealed

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    shown = (
        animator.step(items, now)
        if animator is not None and now is not None
        else [(it, 1.0, 1.0) for it in items]
    )
    entries = []
    for it, fade, share in shown:
        built = backdrop(it) if backdrop is not None else build_backdrop(frame, it)
        drawn = None
        if it.reveal is not None and share < 1.0:
            tx = revealed(render_text(it), it, share)
            drawn = (tx[0], tx[1], qimage(tx[2])) if tx is not None else None
        entries.append((it, built, fade, drawn))
    p = QPainter(img)
    paint_all(p, entries)
    p.end()
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    out = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(h, img.bytesPerLine())
    return cv2.cvtColor(out[:, : w * 3].reshape(h, w, 3), cv2.COLOR_RGB2BGR)
