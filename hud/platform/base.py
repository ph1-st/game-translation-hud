"""The interface between the HUD and the operating system.

The HUD needs five things from the system, and each system does them differently:

* the game's window: listed, matched against the profile, its client area on the screen
* frames of that window that never contain the overlay (the HUD must not read its own
  English): Windows hides the overlay from screen capture; on X11 the frames come from the
  game window's own contents
* the overlay itself: on top, click-through, out of the frames
* global hotkeys
* whether the game runs in exclusive full screen (where no overlay can show)

:class:`Platform` is that interface; ``hud.platform.current()`` gives the one for this
system. Matching a window against a profile is the same everywhere (:func:`best_window`).
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    from capture.profile import WindowMatch

Rect = tuple[int, int, int, int]  # x, y, width, height in screen pixels

WHOLE, PART = 2, 1  # how well a window matched: the whole title or the program; a part

# Programs whose windows are named after what they show (a page, a folder, a file, a
# chat): such a window counts only when the player picked it by its whole title, never by
# a part of it. Windows file names and Linux program / WM_CLASS names.
NOT_GAMES = frozenset({
    # Windows
    "chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "brave.exe", "vivaldi.exe",
    "iexplore.exe", "arc.exe", "explorer.exe", "notepad.exe", "notepad++.exe", "code.exe",
    "discord.exe", "slack.exe", "telegram.exe", "teams.exe", "ms-teams.exe",
    "steamwebhelper.exe", "obs64.exe", "winword.exe", "excel.exe", "acrord32.exe",
    "acrobat.exe", "sumatrapdf.exe", "windowsterminal.exe", "cmd.exe", "powershell.exe",
    "pwsh.exe", "conhost.exe", "openconsole.exe",
    # Linux
    "chrome", "google-chrome", "chromium", "chromium-browser", "firefox", "firefox-esr",
    "navigator", "brave", "brave-browser", "vivaldi-bin", "opera", "nautilus", "dolphin",
    "thunar", "nemo", "pcmanfm", "caja", "code", "codium", "gedit", "kate", "mousepad",
    "discord", "slack", "telegram-desktop", "teams", "obs", "libreoffice", "soffice.bin",
    "evince", "okular", "gnome-terminal-server", "konsole", "xterm", "alacritty", "kitty",
    "wezterm-gui", "tilix", "xfce4-terminal", "steamwebhelper",
})  # fmt: skip
# Programs that show many games or pages: a window picked from one is remembered by its
# title (its program would match every window it has).
SHARED_PROGRAMS = NOT_GAMES | {
    "hd-player.exe", "bluestacks.exe", "dnplayer.exe", "ldplayer.exe", "nox.exe",
    "noxvmhandle.exe", "memu.exe", "mumuplayer.exe", "mumunxdevice.exe",
    "applicationframehost.exe", "vlc.exe", "mpc-hc64.exe", "mpc-be64.exe",
    "potplayermini64.exe", "microsoft.media.player.exe", "python.exe", "pythonw.exe",
    "java.exe", "javaw.exe",
    "vlc", "mpv", "celluloid", "totem", "python", "python3", "java", "wine", "wine64",
    "wine64-preloader", "wine-preloader", "waydroid",
}  # fmt: skip


@dataclass
class WindowRef:
    """A top-level window: its handle (HWND on Windows, the window id on X11), its title,
    its process and program, and its client area on the screen."""

    id: int
    title: str
    pid: int
    program: str  # the program's file name in lower case ("game.exe"); "" when unknown
    client: Rect
    score: int = 0  # how well it matched (best_window): WHOLE or PART


def best_window(windows: list[WindowRef], match: WindowMatch, foreground: int = 0,
                own_pid: int | None = None) -> WindowRef | None:
    """The game's window among *windows*: one whose title matches a pattern of *match*
    (case-insensitive regular expressions) or whose program is one of its process names.
    A window of a program in :data:`NOT_GAMES` counts only when its whole title is one the
    player picked (``match.exact_titles``, any case): a page or folder named after the
    game never does. This app's own windows never count. The best match wins: the whole
    title or the program (:data:`WHOLE`) over a part of the title, then the window in
    front, then the largest."""
    pats = [re.compile(p, re.IGNORECASE) for p in match.title_patterns]
    procs = {p.lower() for p in match.process_names}
    picked = {t.casefold() for t in match.exact_titles}
    me = os.getpid() if own_pid is None else own_pid
    best: WindowRef | None = None
    best_key: tuple = ()
    for w in windows:
        if w.pid == me:
            continue
        whole = any(p.fullmatch(w.title) for p in pats) or w.title.casefold() in picked
        part = whole or any(p.search(w.title) for p in pats)
        by_program = bool(procs) and w.program in procs
        if not part and not by_program:
            continue
        if w.program in NOT_GAMES and not by_program and w.title.casefold() not in picked:
            continue  # a page, folder or chat named after the game
        score = WHOLE if whole or by_program else PART
        key = (score, w.id == foreground, w.client[2] * w.client[3])
        if best is None or key > best_key:
            best, best_key = w, key
            w.score = score
    return best


class Grabber(ABC):
    """Takes frames on the capture thread (hud.screen_source). Every call comes from that
    thread, so a grabber may keep thread-bound resources (a camera, an X connection)."""

    backend = "none"  # what grabbed last, for diagnostics

    @abstractmethod
    def grab(self, area: Rect, target: int | None) -> tuple[np.ndarray | None, bool]:
        """The pixels of *area* (screen pixels) as a BGR array, and whether the grab
        worked; (None, True) when nothing changed since the last grab. *target*: the game
        window's id, for systems that capture the window itself."""

    def reset(self) -> None:  # noqa: B027 - an optional hook
        """Make every resource anew on the next grab (a display that was lost)."""

    def close(self) -> None:  # noqa: B027 - an optional hook
        """Release everything (the capture thread ends)."""


class Hotkeys(ABC):
    """Global hotkeys: a spec ("ctrl+shift+F8", hud.hotkey) runs a callback on the GUI
    thread, whichever window has the focus."""

    def install(self, app: object) -> None:  # noqa: B027 - an optional hook
        """Hook into the application's event loop (before the first register)."""

    @abstractmethod
    def register(self, spec: str, callback: Callable[[], None]) -> bool:
        """Take the key combination; False when it cannot be had (another program has it,
        or the spec names an unknown key)."""

    @abstractmethod
    def unregister_all(self) -> None:
        """Give back every key combination this object took (it may register again)."""

    def uninstall(self) -> None:  # noqa: B027 - an optional hook
        """Unhook from the event loop (the object is done)."""


class Platform(ABC):
    """One operating system's way of doing what the HUD needs (module docstring)."""

    name = "none"
    keys_overlay = False  # can key_overlay give a window a colour key?
    exclude_error = 0  # the system's error code when it last refused exclude_from_capture

    @abstractmethod
    def windows(self) -> list[WindowRef]:
        """Every visible top-level window with a title."""

    def find(self, match: WindowMatch) -> WindowRef | None:
        """The game's window for *match* (:func:`best_window`)."""
        return best_window(self.windows(), match, self.foreground())

    @abstractmethod
    def client_rect(self, window_id: int) -> Rect | None:
        """The window's client area in screen pixels; None once the window is gone."""

    @abstractmethod
    def foreground(self) -> int:
        """The id of the window that has the focus (0: none or unknown)."""

    @abstractmethod
    def grabber(self, want: str = "auto") -> Grabber:
        """A frame grabber whose frames never contain this app's windows (for the capture
        thread). *want*: a backend asked for by name ("auto" picks)."""

    @abstractmethod
    def exclude_from_capture(self, widget: QWidget) -> bool:
        """Keep *widget*'s window out of the HUD's frames; False when the system refused
        (then the HUD would read its own English)."""

    def exclude_detail(self) -> str:
        """Why exclude_from_capture last said no, for the log and the report."""
        return f"error {self.exclude_error}"

    def key_overlay(self, widget: QWidget, key: tuple[int, int, int], opaque: bool) -> bool:
        """Give *widget*'s window a colour key: its pixels in *key* (RGB) show what lies
        under it; *opaque* False: the whole window is see-through for now. False when
        this system cannot (keys_overlay) or refused."""
        return False

    def overlay_flags(self) -> Qt.WindowType:
        """Window flags the overlay adds to its own (frameless, on top, click-through)."""
        from PySide6.QtCore import Qt

        return Qt.WindowType(0)

    def prepare_overlay(self, widget: QWidget) -> bool:
        """Make the shown overlay window what it must be; True when frames exclude it."""
        return self.exclude_from_capture(widget)

    @abstractmethod
    def hotkeys(self) -> Hotkeys:
        """A new set of global hotkeys."""

    def hotkey_available(self, spec: str) -> bool:
        """Could *spec* be taken right now (no program, this one included, has it)?"""
        keys = self.hotkeys()
        try:
            return keys.register(spec, lambda: None)
        finally:
            keys.unregister_all()

    def exclusive_fullscreen(self, window: WindowRef | None) -> bool:
        """Does the game run in exclusive full screen, where no overlay can show? Only a
        hint: asked about the game's window, and only while it is the foreground one."""
        return False

    def ocr_devices(self) -> list[str]:
        """The OCR devices to try for "auto", best first (capture.ocr)."""
        from capture.ocr import preferred_devices

        return preferred_devices()


class NoPlatform(Platform):
    """A system without a supported window system: no game window is ever found (the
    offline tools and the tests still run)."""

    def windows(self) -> list[WindowRef]:
        return []

    def client_rect(self, window_id: int) -> Rect | None:
        return None

    def foreground(self) -> int:
        return 0

    def grabber(self, want: str = "auto") -> Grabber:
        from hud.platform.screen import MssGrabber

        return MssGrabber()

    def exclude_from_capture(self, widget: QWidget) -> bool:
        return False

    def hotkeys(self) -> Hotkeys:
        return _NoHotkeys()


class _NoHotkeys(Hotkeys):
    def register(self, spec: str, callback: Callable[[], None]) -> bool:
        return False

    def unregister_all(self) -> None:
        pass
