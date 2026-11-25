"""Windows: Win32 calls through ctypes (every one declared), Desktop Duplication capture
(dxcam, mss as fallback), RegisterHotKey.

The overlay is kept out of every frame with ``SetWindowDisplayAffinity(hwnd,
WDA_EXCLUDEFROMCAPTURE)`` (Windows 10 2004 or newer): screen capture then shows what lies
under it, so the HUD never reads its own English.
"""

from __future__ import annotations

import ctypes
import logging
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication

from hud.platform.base import Grabber, Hotkeys, Platform, Rect, WindowRef
from hud.platform.screen import MssGrabber

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11
QUNS_RUNNING_D3D_FULL_SCREEN = 3  # SHQueryUserNotificationState: a D3D game owns the screen
WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x0001, 0x0002, 0x0004, 0x0008
MOD_NOREPEAT = 0x4000
DXCAM_RETRY_S = 30.0  # after dxcam failed to set up, mss captures this long before a retry

# virtual key codes of the key names hotkey specs use (hud.hotkey)
VK = {
    "SPACE": 0x20, "TAB": 0x09, "ESC": 0x1B, "PAUSE": 0x13, "SCROLL": 0x91,
    "HOME": 0x24, "END": 0x23, "INSERT": 0x2D, "DELETE": 0x2E, "PAGEUP": 0x21,
    "PAGEDOWN": 0x22, "LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28,
    **{f"F{i}": 0x6F + i for i in range(1, 25)},
    **{f"NUM{i}": 0x60 + i for i in range(10)},
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
}  # fmt: skip
_MODS = {"ctrl": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}

_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _declare() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32")
    sig = {
        "SetWindowDisplayAffinity": ([wintypes.HWND, wintypes.DWORD], wintypes.BOOL),
        "EnumWindows": ([_WNDENUMPROC, wintypes.LPARAM], wintypes.BOOL),
        "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
        "IsIconic": ([wintypes.HWND], wintypes.BOOL),
        "IsWindow": ([wintypes.HWND], wintypes.BOOL),
        "GetWindowTextLengthW": ([wintypes.HWND], ctypes.c_int),
        "GetWindowTextW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        "GetWindowThreadProcessId": ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)],
                                     wintypes.DWORD),
        "GetClientRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        "ClientToScreen": ([wintypes.HWND, ctypes.POINTER(wintypes.POINT)], wintypes.BOOL),
        "GetForegroundWindow": ([], wintypes.HWND),
        "RegisterHotKey": ([wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT],
                           wintypes.BOOL),
        "UnregisterHotKey": ([wintypes.HWND, ctypes.c_int], wintypes.BOOL),
    }  # fmt: skip
    for name, (args, res) in sig.items():
        fn = getattr(user32, name)
        fn.argtypes, fn.restype = args, res
    query = shell32.SHQueryUserNotificationState
    query.argtypes, query.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_long
    return user32, shell32


_user32, _shell32 = _declare()


def _client_rect(hwnd: int) -> Rect | None:
    rc = wintypes.RECT()
    pt = wintypes.POINT(0, 0)
    if not _user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    if not _user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return pt.x, pt.y, rc.right - rc.left, rc.bottom - rc.top


def _program_names(pids: set[int]) -> dict[int, str]:
    """Each process's program file name in lower case ("" when it cannot be told)."""
    import psutil

    out: dict[int, str] = {}
    for pid in pids:
        try:
            out[pid] = (psutil.Process(pid).name() or "").lower()
        except psutil.Error:  # gone, or not ours to look at
            out[pid] = ""
    return out


class WindowsPlatform(Platform):
    name = "windows"

    def windows(self) -> list[WindowRef]:
        found: list[tuple[int, str, int, Rect]] = []

        def visit(hwnd: int, _lparam: int) -> bool:
            if not _user32.IsWindowVisible(hwnd) or _user32.IsIconic(hwnd):
                return True
            n = _user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            _user32.GetWindowTextW(hwnd, buf, n + 1)
            pid = wintypes.DWORD()
            _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            rect = _client_rect(hwnd)
            if rect is not None and rect[2] > 0 and rect[3] > 0:
                found.append((int(hwnd), buf.value, int(pid.value), rect))
            return True

        _user32.EnumWindows(_WNDENUMPROC(visit), 0)
        names = _program_names({pid for _h, _t, pid, _r in found})
        return [WindowRef(h, t, pid, names.get(pid, ""), r) for h, t, pid, r in found]

    def client_rect(self, window_id: int) -> Rect | None:
        if not _user32.IsWindow(window_id):
            return None
        return _client_rect(window_id)

    def foreground(self) -> int:
        return int(_user32.GetForegroundWindow() or 0)

    def grabber(self, want: str = "auto") -> Grabber:
        return MssGrabber() if want == "mss" else DesktopGrabber()

    def exclude_from_capture(self, widget: QWidget) -> bool:
        return bool(_user32.SetWindowDisplayAffinity(int(widget.winId()),
                                                     WDA_EXCLUDEFROMCAPTURE))

    def hotkeys(self) -> Hotkeys:
        return WinHotkeys()

    def exclusive_fullscreen(self, window: WindowRef | None) -> bool:
        if window is None or self.foreground() != window.id:
            return False
        state = ctypes.c_int(0)
        if _shell32.SHQueryUserNotificationState(ctypes.byref(state)) != 0:
            return False
        return state.value == QUNS_RUNNING_D3D_FULL_SCREEN


class DesktopGrabber(Grabber):
    """Desktop Duplication (dxcam) of the monitor that holds the whole area, in that
    output's own coordinates; mss for an area across monitors, or while dxcam fails."""

    def __init__(self) -> None:
        self.backend = "none"
        self._dxcam = None  # the module, once imported
        self._dx_failed_at = -1e9
        self._cams: dict[tuple[int, int], object] = {}
        self._mss = MssGrabber()

    def grab(self, area: Rect, target: int | None = None) -> tuple[np.ndarray | None, bool]:
        if time.monotonic() - self._dx_failed_at > DXCAM_RETRY_S:
            output = self._output_of(area)
            if output is not None:
                return self._grab_dx(area, *output)
        self.backend = "mss"
        return self._mss.grab(area)

    def _outputs(self) -> list[list[object]] | None:
        """dxcam's outputs per adapter. dxcam 0.3 has no public way to tell each output's
        place on the desktop, so its factory is read here, and only here; without it mss
        captures."""
        if self._dxcam is None:
            try:
                import dxcam
            except Exception as exc:  # noqa: BLE001 - no dxcam: mss captures
                log.warning("dxcam unavailable (%s): mss captures", exc)
                self._dx_failed_at = time.monotonic()
                return None
            self._dxcam = dxcam
        factory = getattr(self._dxcam, "__factory", None)
        return getattr(factory, "outputs", None)

    def _output_of(self, area: Rect) -> tuple[int, int, int, int] | None:
        """(adapter, output, left, top) of the monitor that holds all of *area*, or None
        (it spans monitors, or dxcam cannot tell)."""
        outputs = self._outputs()
        if not outputs:
            return None
        x, y, w, h = area
        for refresh in (False, True):
            for d, per_adapter in enumerate(outputs):
                for o, output in enumerate(per_adapter):
                    try:
                        if refresh:
                            output.update_desc()  # (a display changed since)
                        c = output.desc.DesktopCoordinates
                    except Exception:  # noqa: BLE001 - an output that went away
                        continue
                    if c.left <= x and c.top <= y and x + w <= c.right and y + h <= c.bottom:
                        return d, o, c.left, c.top
        return None

    def _grab_dx(self, area: Rect, d: int, o: int, left: int, top: int):
        cam = self._cams.get((d, o))
        if cam is None:
            try:
                cam = self._dxcam.create(device_idx=d, output_idx=o, output_color="BGR")
            except Exception as exc:  # noqa: BLE001 - mss captures for a while
                log.warning("dxcam could not capture output %d.%d (%s): mss captures", d, o, exc)
                self._dx_failed_at = time.monotonic()
                self.backend = "mss"
                return self._mss.grab(area)
            self._cams[(d, o)] = cam
        x, y, w, h = area
        frame = cam.grab(region=(x - left, y - top, x - left + w, y - top + h))
        self.backend = "dxcam"
        return frame, True  # None: nothing changed on the screen since the last grab

    def reset(self) -> None:
        for cam in self._cams.values():
            try:
                cam.release()
            except Exception:  # noqa: BLE001 - a camera of a lost display
                pass
        self._cams.clear()
        self._mss.reset()

    def close(self) -> None:
        self.reset()


def parse_spec(spec: str) -> tuple[int, int]:
    """'ctrl+shift+F8' -> (RegisterHotKey modifiers, virtual key code); ValueError for an
    unknown key or a spec without one."""
    from hud.hotkey import split_spec

    mods, key = split_spec(spec)
    if key not in VK:
        raise ValueError(f"unknown key in hotkey: {key}")
    flags = MOD_NOREPEAT
    for m in mods:
        flags |= _MODS[m]
    return flags, VK[key]


class _HotkeyMessages(QAbstractNativeEventFilter):
    """Hands WM_HOTKEY messages of this thread to the hotkeys that registered them."""

    def __init__(self, callbacks: dict[int, Callable[[], None]]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def nativeEventFilter(self, event_type, message):  # noqa: N802 - Qt API
        msg = wintypes.MSG.from_address(int(message))
        if msg.message == WM_HOTKEY:
            callback = self.callbacks.get(int(msg.wParam))
            if callback is not None:
                callback()
                return True, 0
        return False, 0


class WinHotkeys(Hotkeys):
    """RegisterHotKey on this thread, its WM_HOTKEY messages caught by a Qt native event
    filter. Hotkey ids are numbered across every instance: the panel and the overlay
    each have theirs, and WM_HOTKEY names only the id."""

    _next_id = 0

    def __init__(self) -> None:
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._filter: _HotkeyMessages | None = None
        self._user32 = _user32

    def install(self, app: object) -> None:
        self._filter = _HotkeyMessages(self._callbacks)
        app.installNativeEventFilter(self._filter)  # type: ignore[attr-defined]

    def register(self, spec: str, callback: Callable[[], None]) -> bool:
        try:
            mods, vk = parse_spec(spec)
        except ValueError:
            return False
        WinHotkeys._next_id += 1
        hk_id = WinHotkeys._next_id
        if not self._user32.RegisterHotKey(None, hk_id, mods, vk):
            return False
        self._callbacks[hk_id] = callback
        return True

    def unregister_all(self) -> None:
        for hk_id in list(self._callbacks):
            self._user32.UnregisterHotKey(None, hk_id)
        self._callbacks.clear()

    def uninstall(self) -> None:
        if self._filter is not None:
            app = QCoreApplication.instance()
            if app is not None:
                app.removeNativeEventFilter(self._filter)
            self._filter = None
