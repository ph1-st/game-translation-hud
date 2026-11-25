"""The operating system, behind one interface (:class:`~hud.platform.base.Platform`).

    from hud.platform import current, find_game_window
    window = find_game_window(profile.window)

Windows uses Win32 (hud.platform.windows); Linux uses X11 (hud.platform.x11), also on a
Wayland session, where the app runs through XWayland: :func:`use_x11_for_qt` must run
before the QApplication is made. Elsewhere no game window is ever found.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from hud.platform.base import (
    NOT_GAMES,
    PART,
    SHARED_PROGRAMS,
    WHOLE,
    Grabber,
    Hotkeys,
    Platform,
    Rect,
    WindowRef,
    best_window,
)

if TYPE_CHECKING:
    from capture.profile import WindowMatch

__all__ = [
    "NOT_GAMES", "PART", "SHARED_PROGRAMS", "WHOLE", "Grabber", "Hotkeys", "Platform",
    "Rect", "WindowRef", "best_window", "current", "find_game_window", "use_x11_for_qt",
]  # fmt: skip

_current: Platform | None = None


def current() -> Platform:
    """This system's platform (made once)."""
    global _current
    if _current is None:
        if sys.platform == "win32":
            from hud.platform.windows import WindowsPlatform

            _current = WindowsPlatform()
        elif sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
            from hud.platform.x11 import X11Platform

            _current = X11Platform()
        else:
            from hud.platform.base import NoPlatform

            _current = NoPlatform()
    return _current


def find_game_window(match: WindowMatch) -> WindowRef | None:
    """The game's window for a profile's window match, or None."""
    return current().find(match)


def use_x11_for_qt() -> None:
    """On Linux, Qt runs on X11 (XWayland on a Wayland session): the overlay must be placed
    and stacked by the app, and only X11 lets a window do that. Call before QApplication."""
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
