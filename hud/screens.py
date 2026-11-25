"""Screen pixels (what the system's window positions and the capture use) to Qt's units,
per monitor.

With display scaling (125 %, 150 %) Qt places windows in scaled units. Qt 6 keeps each
screen's top-left corner where it is in screen pixels and scales the rest of it, so a point
maps around the top-left of its own screen: dividing absolute coordinates by the scale is
only right on a monitor that starts at (0, 0). (On X11 Qt usually runs unscaled, and both
units agree.)"""

from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication, QScreen

Rect = tuple[int, int, int, int]


def screen_at(rect: Rect) -> QScreen | None:
    """The screen that holds the centre of *rect* (screen pixels)."""
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    for scr in QGuiApplication.screens():
        g, d = scr.geometry(), scr.devicePixelRatio()
        if g.x() <= cx < g.x() + g.width() * d and g.y() <= cy < g.y() + g.height() * d:
            return scr
    return QGuiApplication.primaryScreen()


def to_qt(rect: Rect, screen: QScreen | None) -> QRect:
    """*rect* (screen pixels) in Qt's units on *screen*."""
    x, y, w, h = rect
    if screen is None:
        return QRect(x, y, w, h)
    g, d = screen.geometry(), screen.devicePixelRatio()
    ox, oy = g.x(), g.y()
    return QRect(round(ox + (x - ox) / d), round(oy + (y - oy) / d), round(w / d), round(h / d))
