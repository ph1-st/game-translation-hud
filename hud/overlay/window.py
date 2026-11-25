"""The overlay window: frameless, translucent, click-through, always on top, and excluded
from screen capture. It draws :class:`hud.items.OverlayItem` items in window coordinates
exactly as the layout decided them (box, font size, line breaks): nothing is fitted,
wrapped or resized here. New items fade in; removed items vanish at once (a cover goes the
moment its Japanese goes) or fade out when they say so.

Backdrops made from the screen (painted-over Japanese, frosted glass) are built from the
newest captured frame (:meth:`OverlayWindow.set_frame`) and rebuilt when the pixels under
them change, so a scene moving behind a key hint is painted over as it is, never as it
was. They are built on a thread of their own (:mod:`hud.overlay.builder`); an item whose
Japanese is not painted over yet is not shown yet.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from hud import backdrop
from hud.items import OverlayItem, reveal_target
from hud.overlay.builder import BackdropBuilder
from hud.overlay.paint import PROVISIONAL_ALPHA, Layer, paint_all, qimage, render_text
from hud.overlay.reveal import revealed
from hud.platform import current as current_platform
from hud.screens import screen_at, to_qt

FADE_IN_S = 0.12
REBUILD_DELTA = 2.0  # mean grey change under an erase that rebuilds its backdrop
ANIMATE_MS = 16  # fades and typing text are drawn this often
COLLECT_MS = 8  # finished backdrops are picked up this often while any are due


def _made_from_frame(item: OverlayItem) -> bool:
    """Is the item's backdrop made from the screen (it paints over the Japanese, or is a
    pane of frosted glass)? Then it is not shown before it is built, and is built again
    when the pixels under it change."""
    return item.erase is not None or item.frost is not None


def _watched(item: OverlayItem) -> tuple[int, int, int, int]:
    """The pixels a backdrop is made from: what it paints over and, for a label, what lies
    under its text (stray marks, the pill it may extend); a pane of glass, its area."""
    if item.erase is None:
        assert item.frost is not None
        return item.frost
    x, y, w, h = item.erase
    if item.style != "label":
        return item.erase
    rx, ry, rw, rh = item.rect
    x0, y0 = min(x, rx), min(y, ry)
    return x0, y0, max(x + w, rx + rw) - x0, max(y + h, ry + rh) - y0


@dataclass
class _Anim:
    item: OverlayItem
    alpha: float
    built: tuple[int, int, QImage] | None = None
    sig: np.ndarray | None = None  # the pixels under the backdrop in the last frame sent
    fresh: bool = False  # built at least once
    built_px: Layer | None = None  # the backdrop's pixels...
    text_px: Layer | None = None  # ...and the text's, drawn once...
    text_img: tuple[int, int, QImage] | None = None  # ...for the window too
    text_full: Layer | None = None  # the text before it typed out
    share: float = 1.0  # how much of the text shows (it may type out with the Japanese)
    leaving: bool = False  # gone from the item set, fading out

    def show_text(self) -> None:
        """The text as far as it typed out, for the window and the recorder."""
        self.text_px = revealed(self.text_full, self.item, self.share)
        tx = self.text_px
        self.text_img = (tx[0], tx[1], qimage(tx[2])) if tx is not None else None


class OverlayWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | current_platform().overlay_flags(),
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._anims: dict[str, _Anim] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()  # the item set, read by the recorder's thread
        self.pixels_version = 0  # bumped whenever what the window shows changes
        self._frame: np.ndarray | None = None
        self.capture_excluded = False
        self.placed_for: tuple | None = None  # (rect, scale) it was last placed for
        self.enabled = True
        self._last_tick = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(ANIMATE_MS)
        self._timer.timeout.connect(self._animate)
        self._builder = BackdropBuilder()
        self._builder.start()
        self._collector = QTimer(self)
        self._collector.setInterval(COLLECT_MS)
        self._collector.timeout.connect(self._collect)

    # ---- lifecycle --------------------------------------------------------------------
    def show_over(self, rect: tuple[int, int, int, int]) -> None:
        """Lie over *rect* (the game's client area in screen pixels). With display scaling
        (125 %, 150 %) Qt places windows in scaled units: the window goes where those put it,
        and everything is painted at the screen's scale, so a frame pixel stays a pixel."""
        screen = screen_at(rect)
        scale = float(screen.devicePixelRatio()) if screen is not None else 1.0
        self.placed_for = (tuple(rect), scale)
        self.setGeometry(to_qt(rect, screen))
        if not self.isVisible():
            self.show()
        self.capture_excluded = current_platform().prepare_overlay(self)
        self.raise_()

    def misplaced(self, rect: tuple[int, int, int, int]) -> bool:
        """Is the window placed for another rect or another scaling of its screen (the
        player changed the display scaling, or the game moved to another monitor)?"""
        screen = screen_at(rect)
        d = float(screen.devicePixelRatio()) if screen is not None else 1.0
        return self.placed_for != (tuple(rect), d)

    def set_items(self, items: list[OverlayItem]) -> None:
        """Replace the item set. An item that keeps its identity is left exactly as it is
        drawn, a new one fades in, a missing one is gone at once (or fades out, when it
        says so)."""
        wanted: dict[str, OverlayItem] = {}
        for it in items:
            wanted.setdefault(it.identity(), it)
        # the text of new items is drawn once, into an image the window paints from
        # (repainting paths every frame is what costs), and before the lock is taken: the
        # recorder's thread reads the item set many times a second
        texts = {}
        for ident, it in wanted.items():
            anim = self._anims.get(ident)
            if anim is None or anim.item != it:
                texts[ident] = render_text(it)
        changed = False
        with self._lock:
            for ident in list(self._anims):
                if ident not in wanted:
                    anim = self._anims[ident]
                    if anim.item.fade_out > 0 and anim.alpha > 0:
                        anim.leaving = True  # it fades out (the timer takes it away)
                    else:
                        self._forget(ident)
                    changed = True
            for ident, it in wanted.items():
                anim = self._anims.get(ident)
                if anim is None:
                    anim = self._anims[ident] = _Anim(it, 0.0)
                    anim.share = 0.0 if it.reveal is not None else 1.0
                    self._order.append(ident)
                elif anim.leaving or anim.item != it:
                    anim.leaving = False  # back before it was gone: it fades in again
                    anim.item = it
                else:
                    continue
                if ident in texts:
                    anim.text_full = texts[ident]
                    anim.show_text()
                self._rebuild(ident, anim)
                changed = True
        if changed:
            if not self._timer.isActive():
                self._last_tick = time.monotonic()
                self._timer.start()
            self._repaint()

    def _forget(self, ident: str) -> None:
        del self._anims[ident]
        self._order.remove(ident)
        self._builder.forget(ident)

    def set_frame(self, frame: np.ndarray) -> None:
        """The newest captured frame: backdrops whose pixels changed are built again."""
        first = self._frame is None
        self._frame = frame
        for ident, anim in self._anims.items():
            if anim.leaving:
                continue  # fading out as it was
            if not _made_from_frame(anim.item):
                if first and not anim.fresh:
                    self._rebuild(ident, anim)
                continue
            self._refresh(ident, anim, frame)
        self._collect()

    def has_glass(self) -> bool:
        return any(a.item.frost is not None for a in self._anims.values())

    def refresh_glass(self, frame: np.ndarray) -> None:
        """A newer frame for the panes of frosted glass only: they follow the scene behind
        them closely (a blur of it), more often than the window's tick."""
        self._frame = frame
        for ident, anim in self._anims.items():
            if anim.item.frost is not None and not anim.leaving:
                self._refresh(ident, anim, frame)

    def _refresh(self, ident: str, anim: _Anim, frame: np.ndarray) -> None:
        sig = backdrop.signature(frame, _watched(anim.item))
        if (
            sig is not None
            and anim.sig is not None
            and sig.shape == anim.sig.shape
            and float(np.abs(sig - anim.sig).mean()) < REBUILD_DELTA
        ):
            return
        self._rebuild(ident, anim)

    def _rebuild(self, ident: str, anim: _Anim) -> None:
        it = anim.item
        if _made_from_frame(it):
            if self._frame is None:
                return  # built with the first frame
            anim.sig = backdrop.signature(self._frame, _watched(it))
        self._builder.request(ident, it, self._frame if _made_from_frame(it) else None)
        if not self._collector.isActive():
            self._collector.start()

    def _collect(self) -> None:
        """Finished backdrops replace the ones drawn (for items still shown as they were)."""
        changed = False
        with self._lock:
            for ident, (item, built) in self._builder.take().items():
                anim = self._anims.get(ident)
                if anim is None or anim.item != item:
                    continue  # gone, or changed since: its own request is on the way
                anim.built, anim.built_px = built if built is not None else (None, None)
                anim.fresh = True
                changed = True
        if self._builder.idle():
            self._collector.stop()
        if changed and self.enabled:
            self._repaint()

    def close_builder(self) -> None:
        self._builder.stop()

    def clear(self) -> None:
        self.set_items([])

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self._repaint()

    def _animate(self) -> None:
        now = time.monotonic()  # (the clock of the core's readings: a reveal's start)
        dt = now - self._last_tick
        self._last_tick = now
        busy = False
        with self._lock:
            for ident, anim in list(self._anims.items()):
                if anim.leaving:
                    anim.alpha -= dt / anim.item.fade_out
                    if anim.alpha <= 0.0:
                        self._forget(ident)
                    busy = True
                    continue
                if anim.alpha < 1.0:
                    if anim.fresh or not _made_from_frame(anim.item):
                        # (a backdrop fades in from its first picture, not before)
                        anim.alpha = min(1.0, anim.alpha + dt / (anim.item.fade_in or FADE_IN_S))
                    busy = busy or anim.alpha < 1.0
                if anim.share < 1.0:
                    share = max(anim.share, reveal_target(anim.item, now))
                    if share > anim.share:
                        anim.share = share
                        anim.show_text()
                    busy = busy or anim.share < 1.0
        if not busy:
            self._timer.stop()
        self._repaint()

    # ---- painting -----------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        if not self.enabled:
            return
        p = QPainter(self)
        d = float(self.devicePixelRatioF()) or 1.0  # (the screen it is on now)
        if abs(d - 1.0) > 1e-3:
            p.scale(1.0 / d, 1.0 / d)  # items are in frame (screen) pixels
        self.render_items(p)
        p.end()

    def _repaint(self) -> None:
        self.pixels_version += 1
        self.update()

    def snapshot(self) -> tuple[int, list[tuple[Layer, float]]]:
        """(version, what the window shows right now as layers of premultiplied RGBA pixels
        with their fade, in the order the window paints them), for the recorder's thread."""
        version = self.pixels_version
        if not self.enabled:
            return version, []
        with self._lock:
            anims = [self._anims[i] for i in self._order if i in self._anims]
            shown = [a for a in anims if a.alpha > 0 and (not _made_from_frame(a.item) or a.fresh)]
            out = []
            for styles in (("panel",), ("label",), ("text",)):
                for a in shown:
                    if a.item.style in styles and a.built_px is not None:
                        out.append((a.built_px, a.alpha))
            for a in shown:
                if a.item.style in ("label", "text") and a.text_px is not None:
                    faint = PROVISIONAL_ALPHA if a.item.provisional else 1.0
                    out.append((a.text_px, a.alpha * faint))
            return version, out

    def render_items(self, p: QPainter) -> None:
        """Paint the current items (with their fade state) on any painter."""
        entries = []
        for ident in self._order:
            anim = self._anims.get(ident)
            if anim is None or (_made_from_frame(anim.item) and not anim.fresh):
                continue  # not built yet: the Japanese would show through
            entries.append((anim.item, anim.built, anim.alpha, anim.text_img))
        paint_all(p, entries)
