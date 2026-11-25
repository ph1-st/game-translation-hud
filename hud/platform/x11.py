"""X11 (and XWayland on a Wayland session): Xlib through ctypes.

X11 has no way to hide a window from screen capture, so the HUD captures the game's window
itself instead of the screen: with the Composite extension the window keeps its own
pixels in an off-screen pixmap (``XCompositeRedirectWindow``, automatic: the screen shows
it as before), and the frame is read from that pixmap (``XShmGetImage``; ``XGetImage``
without shared memory). Whatever lies above the game, the overlay included, is never in
the frame.

The overlay is an override-redirect window (no window manager decides where it goes or
whether it stays on top) with an empty input shape (clicks fall through to the game; Qt
does that for ``WindowTransparentForInput``). Global hotkeys are ``XGrabKey`` grabs on the
root window, read by a thread of their own.

Every connection belongs to one thread: the GUI's (window list, focus), the capture
thread's and the hotkey thread's. X errors are trapped per connection, never fatal.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import select
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, Signal

from hud.platform.base import Grabber, Hotkeys, Platform, Rect, WindowRef

if TYPE_CHECKING:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

# ---- Xlib through ctypes --------------------------------------------------------------
c_Display = ctypes.c_void_p
XID = ctypes.c_ulong

SUCCESS = 0
IS_VIEWABLE = 2  # XWindowAttributes.map_state
INPUT_ONLY = 2  # XWindowAttributes.class
Z_PIXMAP = 2
ALL_PLANES = ctypes.c_ulong(-1).value
REDIRECT_AUTOMATIC = 0  # CompositeRedirectAutomatic: the screen still shows the window
KEY_PRESS = 2
GRAB_MODE_ASYNC = 1
XA_ATOM, XA_CARDINAL, XA_STRING, XA_WINDOW, XA_WM_NAME, XA_WM_CLASS = 4, 6, 31, 33, 39, 67
ANY_PROPERTY_TYPE = 0
SHIFT_MASK, LOCK_MASK, CONTROL_MASK, MOD1_MASK, MOD2_MASK, MOD4_MASK = 1, 2, 4, 8, 16, 64
IPC_PRIVATE, IPC_CREAT, IPC_RMID = 0, 0o1000, 0
PROPERTY_LONGS = 1 << 16  # longest property read (in 32-bit units)
HOTKEY_POLL_S = 0.2  # how often the hotkey thread looks whether it should end


class XErrorEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("display", c_Display), ("resourceid", XID),
        ("serial", ctypes.c_ulong), ("error_code", ctypes.c_ubyte),
        ("request_code", ctypes.c_ubyte), ("minor_code", ctypes.c_ubyte),
    ]  # fmt: skip


class XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int), ("y", ctypes.c_int), ("width", ctypes.c_int),
        ("height", ctypes.c_int), ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p), ("root", XID), ("c_class", ctypes.c_int),
        ("bit_gravity", ctypes.c_int), ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int), ("colormap", XID),
        ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long), ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long), ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]  # fmt: skip


class XImage(ctypes.Structure):
    pass


class _XImageFuncs(ctypes.Structure):
    _fields_ = [
        ("create_image", ctypes.c_void_p),
        ("destroy_image", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(XImage))),
        ("get_pixel", ctypes.c_void_p), ("put_pixel", ctypes.c_void_p),
        ("sub_image", ctypes.c_void_p), ("add_pixel", ctypes.c_void_p),
    ]  # fmt: skip


XImage._fields_ = [
    ("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int),
    ("format", ctypes.c_int), ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
    ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int),
    ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int), ("bytes_per_line", ctypes.c_int),
    ("bits_per_pixel", ctypes.c_int), ("red_mask", ctypes.c_ulong),
    ("green_mask", ctypes.c_ulong), ("blue_mask", ctypes.c_ulong),
    ("obdata", ctypes.c_void_p), ("f", _XImageFuncs),
]  # fmt: skip


class XShmSegmentInfo(ctypes.Structure):
    _fields_ = [("shmseg", ctypes.c_ulong), ("shmid", ctypes.c_int),
                ("shmaddr", ctypes.c_void_p), ("readOnly", ctypes.c_int)]


class XKeyEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
        ("display", c_Display), ("window", XID), ("root", XID), ("subwindow", XID),
        ("time", ctypes.c_ulong), ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("x_root", ctypes.c_int), ("y_root", ctypes.c_int), ("state", ctypes.c_uint),
        ("keycode", ctypes.c_uint), ("same_screen", ctypes.c_int),
    ]  # fmt: skip


class XEvent(ctypes.Union):
    _fields_ = [("type", ctypes.c_int), ("xkey", XKeyEvent), ("pad", ctypes.c_long * 24)]


_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, c_Display, ctypes.POINTER(XErrorEvent))


def _load(name: str, soname: str) -> ctypes.CDLL:
    return ctypes.CDLL(ctypes.util.find_library(name) or soname)


class _Xlib:
    """The Xlib, Xext (MIT-SHM) and Xcomposite functions used here, declared once."""

    def __init__(self) -> None:
        x = _load("X11", "libX11.so.6")
        ext = _load("Xext", "libXext.so.6")
        comp = _load("Xcomposite", "libXcomposite.so.1")
        libc = ctypes.CDLL(None, use_errno=True)
        p = ctypes.POINTER
        u, i, ul = ctypes.c_uint, ctypes.c_int, ctypes.c_ulong
        sigs = {
            (x, "XInitThreads"): ([], i),
            (x, "XOpenDisplay"): ([ctypes.c_char_p], c_Display),
            (x, "XCloseDisplay"): ([c_Display], i),
            (x, "XDefaultRootWindow"): ([c_Display], XID),
            (x, "XConnectionNumber"): ([c_Display], i),
            (x, "XSync"): ([c_Display, i], i),
            (x, "XFlush"): ([c_Display], i),
            (x, "XPending"): ([c_Display], i),
            (x, "XNextEvent"): ([c_Display, p(XEvent)], i),
            (x, "XSetErrorHandler"): ([_ERROR_HANDLER], ctypes.c_void_p),
            (x, "XInternAtom"): ([c_Display, ctypes.c_char_p, i], XID),
            (x, "XGetWindowProperty"): (
                [c_Display, XID, XID, ctypes.c_long, ctypes.c_long, i, XID, p(XID), p(i),
                 p(ul), p(ul), p(ctypes.c_void_p)], i),
            (x, "XFree"): ([ctypes.c_void_p], i),
            (x, "XQueryTree"): ([c_Display, XID, p(XID), p(XID), p(ctypes.c_void_p), p(u)], i),
            (x, "XGetWindowAttributes"): ([c_Display, XID, p(XWindowAttributes)], i),
            (x, "XTranslateCoordinates"): (
                [c_Display, XID, XID, i, i, p(i), p(i), p(XID)], i),
            (x, "XGetInputFocus"): ([c_Display, p(XID), p(i)], i),
            (x, "XGetImage"): ([c_Display, XID, i, i, u, u, ul, i], p(XImage)),
            (x, "XFreePixmap"): ([c_Display, XID], i),
            (x, "XStringToKeysym"): ([ctypes.c_char_p], XID),
            (x, "XKeysymToKeycode"): ([c_Display, XID], ctypes.c_ubyte),
            (x, "XGrabKey"): ([c_Display, i, u, XID, i, i, i], i),
            (x, "XUngrabKey"): ([c_Display, i, u, XID], i),
            (ext, "XShmQueryExtension"): ([c_Display], i),
            (ext, "XShmCreateImage"): (
                [c_Display, ctypes.c_void_p, u, i, ctypes.c_char_p, p(XShmSegmentInfo), u, u],
                p(XImage)),
            (ext, "XShmAttach"): ([c_Display, p(XShmSegmentInfo)], i),
            (ext, "XShmDetach"): ([c_Display, p(XShmSegmentInfo)], i),
            (ext, "XShmGetImage"): ([c_Display, XID, p(XImage), i, i, ul], i),
            (comp, "XCompositeQueryExtension"): ([c_Display, p(i), p(i)], i),
            (comp, "XCompositeRedirectWindow"): ([c_Display, XID, i], None),
            (comp, "XCompositeNameWindowPixmap"): ([c_Display, XID], XID),
            (libc, "shmget"): ([i, ctypes.c_size_t, i], i),
            (libc, "shmat"): ([i, ctypes.c_void_p, i], ctypes.c_void_p),
            (libc, "shmdt"): ([ctypes.c_void_p], i),
            (libc, "shmctl"): ([i, i, ctypes.c_void_p], i),
        }  # fmt: skip
        for (lib, name), (args, res) in sigs.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, res
            setattr(self, name, fn)


_lib: _Xlib | None = None
_errors: dict[int, int] = {}  # display pointer -> the first X error code since the trap
_errors_lock = threading.Lock()


@_ERROR_HANDLER
def _on_error(display: int, event: ctypes.POINTER(XErrorEvent)) -> int:  # type: ignore[valid-type]
    """Record an X error for the connection that caused it (Xlib's default handler would
    end the process)."""
    with _errors_lock:
        _errors.setdefault(int(display or 0), int(event.contents.error_code))
    return 0


def xlib() -> _Xlib:
    """The Xlib functions, loaded on first use (threads enabled, errors trapped)."""
    global _lib
    if _lib is None:
        lib = _Xlib()
        lib.XInitThreads()
        lib.XSetErrorHandler(_on_error)
        _lib = lib
    return _lib


class XError(Exception):
    """An X request failed (the window went away, a key is taken...)."""

    def __init__(self, code: int) -> None:
        super().__init__(f"X error {code}")
        self.code = code


@contextmanager
def trap(display: int) -> Iterator[None]:
    """Run X requests; raise XError if the server refused one of them."""
    lib = xlib()
    lib.XSync(display, 0)
    with _errors_lock:
        _errors.pop(display, None)
    yield
    lib.XSync(display, 0)
    with _errors_lock:
        code = _errors.pop(display, None)
    if code is not None:
        raise XError(code)


def open_display(name: str | None = None) -> int:
    """A connection to the X server ($DISPLAY by default); OSError when there is none."""
    disp = xlib().XOpenDisplay(name.encode() if name else None)
    if not disp:
        raise OSError(f"cannot open the X display {name or os.environ.get('DISPLAY', '')!r}")
    return int(disp)


# ---- reading windows ---------------------------------------------------------------------
class _Atoms:
    def __init__(self, display: int) -> None:
        lib = xlib()
        for name in ("_NET_CLIENT_LIST", "_NET_ACTIVE_WINDOW", "_NET_WM_NAME", "_NET_WM_PID",
                     "UTF8_STRING"):  # fmt: skip
            setattr(self, name.strip("_"), lib.XInternAtom(display, name.encode(), 0))


def get_property(display: int, window: int, prop: int, kind: int) -> tuple[int, bytes]:
    """(format, raw bytes) of a window property; (0, b"") when it has none. Items of
    format 32 come as C longs (8 bytes each on 64-bit systems)."""
    lib = xlib()
    actual_type, actual_format = XID(), ctypes.c_int()
    nitems, after = ctypes.c_ulong(), ctypes.c_ulong()
    data = ctypes.c_void_p()
    status = lib.XGetWindowProperty(
        display, window, prop, 0, PROPERTY_LONGS, 0, kind, ctypes.byref(actual_type),
        ctypes.byref(actual_format), ctypes.byref(nitems), ctypes.byref(after),
        ctypes.byref(data),
    )  # fmt: skip
    if status != SUCCESS or not data.value:
        return 0, b""
    try:
        fmt = actual_format.value
        size = {8: 1, 16: ctypes.sizeof(ctypes.c_short), 32: ctypes.sizeof(ctypes.c_long)}
        n = nitems.value * size.get(fmt, 0)
        return fmt, ctypes.string_at(data.value, n)
    finally:
        lib.XFree(data)


def _longs(raw: bytes) -> list[int]:
    n = len(raw) // ctypes.sizeof(ctypes.c_ulong)
    return list((ctypes.c_ulong * n).from_buffer_copy(raw[: n * ctypes.sizeof(ctypes.c_ulong)]))


def _text(raw: bytes) -> str:
    raw = raw.split(b"\0")[0]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def window_title(display: int, window: int, atoms: _Atoms) -> str:
    _fmt, raw = get_property(display, window, atoms.NET_WM_NAME, atoms.UTF8_STRING)
    if not raw:
        _fmt, raw = get_property(display, window, XA_WM_NAME, ANY_PROPERTY_TYPE)
    return _text(raw)


def window_class(display: int, window: int) -> tuple[str, str]:
    """WM_CLASS: (instance, class), e.g. ("azurpromiliajp.exe", "azurpromiliajp.exe") for
    a Wine program, ("steam_app_123", ...) for one Proton runs."""
    _fmt, raw = get_property(display, window, XA_WM_CLASS, XA_STRING)
    parts = raw.split(b"\0")
    return (_text(parts[0]) if parts else "", _text(parts[1]) if len(parts) > 1 else "")


def program_of(pid: int, wm_class: tuple[str, str]) -> str:
    """The window's program in lower case: the .exe Wine names in WM_CLASS, else the
    process's program (a Windows path on Wine's command line is cut at its backslash)."""
    for name in wm_class:
        if name.lower().endswith(".exe"):
            return name.lower()
    if pid > 0:
        try:
            import psutil

            proc = psutil.Process(pid)
            cmd = proc.cmdline()
            if cmd:
                first = cmd[0].replace("\\", "/").rsplit("/", 1)[-1]
                if first.lower().endswith(".exe"):
                    return first.lower()
            return (proc.name() or "").lower()
        except psutil.Error:
            pass
    return (wm_class[0] or "").lower()


def client_rect(display: int, window: int) -> Rect | None:
    """The window's area on the screen (its own pixels, no frame); None once it is gone or
    not on the screen."""
    lib = xlib()
    attrs = XWindowAttributes()
    if not lib.XGetWindowAttributes(display, window, ctypes.byref(attrs)):
        return None
    if attrs.map_state != IS_VIEWABLE or attrs.c_class == INPUT_ONLY:
        return None
    x, y, child = ctypes.c_int(), ctypes.c_int(), XID()
    if not lib.XTranslateCoordinates(display, window, attrs.root, 0, 0, ctypes.byref(x),
                                     ctypes.byref(y), ctypes.byref(child)):
        return None
    return x.value, y.value, attrs.width, attrs.height


def top_windows(display: int, atoms: _Atoms) -> list[int]:
    """The client windows the window manager lists (_NET_CLIENT_LIST); without one, the
    root window's children that have a title."""
    root = xlib().XDefaultRootWindow(display)
    _fmt, raw = get_property(display, root, atoms.NET_CLIENT_LIST, XA_WINDOW)
    if raw:
        return _longs(raw)
    lib = xlib()
    root_ret, parent = XID(), XID()
    children, count = ctypes.c_void_p(), ctypes.c_uint()
    if not lib.XQueryTree(display, root, ctypes.byref(root_ret), ctypes.byref(parent),
                          ctypes.byref(children), ctypes.byref(count)):
        return []
    try:
        ids = list((XID * count.value).from_address(children.value)) if count.value else []
    finally:
        if children.value:
            lib.XFree(children)
    return [w for w in ids if window_title(display, w, atoms)]


# ---- the platform -----------------------------------------------------------------------
class X11Platform(Platform):
    name = "x11"

    def __init__(self, display_name: str | None = None) -> None:
        self._display_name = display_name
        self._display = 0
        self._atoms: _Atoms | None = None
        self._lock = threading.Lock()

    def _conn(self) -> tuple[int, _Atoms]:
        if not self._display:
            self._display = open_display(self._display_name)
            self._atoms = _Atoms(self._display)
        assert self._atoms is not None
        return self._display, self._atoms

    def windows(self) -> list[WindowRef]:
        out: list[WindowRef] = []
        with self._lock:
            display, atoms = self._conn()
            for w in top_windows(display, atoms):
                try:
                    with trap(display):
                        rect = client_rect(display, w)
                        if rect is None or rect[2] <= 0 or rect[3] <= 0:
                            continue
                        title = window_title(display, w, atoms)
                        if not title:
                            continue
                        _fmt, raw = get_property(display, w, atoms.NET_WM_PID, XA_CARDINAL)
                        pids = _longs(raw)
                        cls = window_class(display, w)
                except XError:  # the window went away meanwhile
                    continue
                pid = pids[0] if pids else 0
                out.append(WindowRef(w, title, pid, program_of(pid, cls), rect))
        return out

    def client_rect(self, window_id: int) -> Rect | None:
        with self._lock:
            display, _atoms = self._conn()
            try:
                with trap(display):
                    return client_rect(display, window_id)
            except XError:
                return None

    def foreground(self) -> int:
        with self._lock:
            display, atoms = self._conn()
            root = xlib().XDefaultRootWindow(display)
            _fmt, raw = get_property(display, root, atoms.NET_ACTIVE_WINDOW, XA_WINDOW)
            ids = _longs(raw)
            if ids:
                return ids[0]
            focus, revert = XID(), ctypes.c_int()
            xlib().XGetInputFocus(display, ctypes.byref(focus), ctypes.byref(revert))
            return int(focus.value)

    def grabber(self, want: str = "auto") -> Grabber:
        return WindowGrabber(self._display_name)

    def exclude_from_capture(self, widget: QWidget) -> bool:
        return True  # frames are the game window's own pixels: no other window is in them

    def overlay_flags(self) -> Qt.WindowType:
        from PySide6.QtCore import Qt

        return Qt.WindowType.X11BypassWindowManagerHint

    def prepare_overlay(self, widget: QWidget) -> bool:
        """Override-redirect (placed and stacked where the HUD says, above a full-screen
        game too); set before the window was first shown, else shown again once."""
        from PySide6.QtCore import Qt

        flag = Qt.WindowType.X11BypassWindowManagerHint
        if not widget.windowFlags() & flag:
            visible = widget.isVisible()
            widget.setWindowFlag(flag, True)  # (re-creates the native window, hidden)
            if visible:
                widget.show()
        return True

    def hotkeys(self) -> Hotkeys:
        return X11Hotkeys(self._display_name)


# ---- frames of the game window -----------------------------------------------------------
class WindowGrabber(Grabber):
    """The game window's own pixels (Composite), on the capture thread's connection."""

    backend = "xcomposite"

    def __init__(self, display_name: str | None = None) -> None:
        self._display_name = display_name
        self._display = 0
        self._redirected: set[int] = set()
        self._shm: _ShmImage | None = None
        self._last: np.ndarray | None = None

    def _conn(self) -> int:
        if not self._display:
            display = open_display(self._display_name)
            ev, err = ctypes.c_int(), ctypes.c_int()
            if not xlib().XCompositeQueryExtension(display, ctypes.byref(ev),
                                                   ctypes.byref(err)):
                xlib().XCloseDisplay(display)
                raise OSError("the X server has no Composite extension")
            self._display = display
        return self._display

    def grab(self, area: Rect, target: int | None) -> tuple[np.ndarray | None, bool]:
        if not target:
            return None, False  # (the screen would show the overlay: only windows are read)
        display = self._conn()
        lib = xlib()
        try:
            with trap(display):
                where = client_rect(display, target)
                if where is None:
                    return None, False
                if target not in self._redirected:
                    lib.XCompositeRedirectWindow(display, target, REDIRECT_AUTOMATIC)
                    self._redirected.add(target)
                wx, wy, ww, wh = where
                x, y, w, h = area[0] - wx, area[1] - wy, area[2], area[3]
                if x < 0 or y < 0 or x + w > ww or y + h > wh or w <= 0 or h <= 0:
                    return None, False  # (the area left the window: the HUD finds it again)
                pixmap = lib.XCompositeNameWindowPixmap(display, target)
            try:
                frame = self._read(display, pixmap, target, x, y, w, h)
            finally:
                lib.XFreePixmap(display, pixmap)
        except XError as e:
            log.debug("window capture failed: %s", e)
            return None, False
        if frame is None:
            return None, False
        if self._last is not None and self._last.shape == frame.shape and np.array_equal(
            self._last, frame
        ):
            return None, True  # nothing changed since the last grab
        self._last = frame
        return frame, True

    def _read(self, display: int, drawable: int, window: int, x: int, y: int, w: int,
              h: int) -> np.ndarray | None:  # fmt: skip
        attrs = XWindowAttributes()
        xlib().XGetWindowAttributes(display, window, ctypes.byref(attrs))
        if self._shm is not None and not self._shm.fits(w, h, attrs.depth):
            self._shm.close()
            self._shm = None
        if self._shm is None and xlib().XShmQueryExtension(display):
            try:
                self._shm = _ShmImage(display, attrs.visual, attrs.depth, w, h)
            except OSError as e:
                log.info("no shared memory for capture (%s): XGetImage reads", e)
        with trap(display):
            if self._shm is not None:
                return self._shm.grab(drawable, x, y)
            return _get_image(display, drawable, x, y, w, h)

    def reset(self) -> None:
        self.close()

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._display:
            xlib().XCloseDisplay(self._display)
            self._display = 0
        self._redirected.clear()
        self._last = None


def _to_bgr(image: XImage) -> np.ndarray:
    """A 24- or 32-bit ZPixmap image as a BGR array (copied)."""
    if image.bits_per_pixel != 32:
        raise XError(-1)  # (only TrueColor 24/32 bit visuals: what games and desktops use)
    w, h, bpl = image.width, image.height, image.bytes_per_line
    buf = (ctypes.c_ubyte * (bpl * h)).from_address(image.data)
    rows = np.frombuffer(buf, np.uint8).reshape(h, bpl)
    return rows[:, : w * 4].reshape(h, w, 4)[:, :, :3].copy()


def _get_image(display: int, drawable: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    ptr = xlib().XGetImage(display, drawable, x, y, w, h, ALL_PLANES, Z_PIXMAP)
    if not ptr:
        raise XError(-1)
    try:
        return _to_bgr(ptr.contents)
    finally:
        ptr.contents.f.destroy_image(ptr)


class _ShmImage:
    """An XImage in shared memory, filled by the server without copying through the
    socket (MIT-SHM); one size at a time."""

    def __init__(self, display: int, visual: int, depth: int, w: int, h: int) -> None:
        lib = xlib()
        self.display, self.w, self.h, self.depth = display, w, h, depth
        self.info = XShmSegmentInfo()
        self.image = lib.XShmCreateImage(display, visual, depth, Z_PIXMAP, None,
                                         ctypes.byref(self.info), w, h)
        if not self.image:
            raise OSError("XShmCreateImage failed")
        size = self.image.contents.bytes_per_line * h
        self.info.shmid = lib.shmget(IPC_PRIVATE, size, IPC_CREAT | 0o600)
        if self.info.shmid < 0:
            self._free_image()
            raise OSError(ctypes.get_errno(), "shmget failed")
        addr = lib.shmat(self.info.shmid, None, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            lib.shmctl(self.info.shmid, IPC_RMID, None)
            self._free_image()
            raise OSError(ctypes.get_errno(), "shmat failed")
        self.info.shmaddr = addr
        self.image.contents.data = addr
        self.info.readOnly = 0
        try:
            with trap(display):
                lib.XShmAttach(display, ctypes.byref(self.info))
        except XError as e:  # (a remote X server cannot share memory)
            lib.shmdt(addr)
            lib.shmctl(self.info.shmid, IPC_RMID, None)
            self._free_image()
            raise OSError(f"XShmAttach failed ({e})") from e
        lib.shmctl(self.info.shmid, IPC_RMID, None)  # (freed once both sides detach)

    def fits(self, w: int, h: int, depth: int) -> bool:
        return (w, h, depth) == (self.w, self.h, self.depth)

    def grab(self, drawable: int, x: int, y: int) -> np.ndarray:
        if not xlib().XShmGetImage(self.display, drawable, self.image, x, y, ALL_PLANES):
            raise XError(-1)
        return _to_bgr(self.image.contents)

    def _free_image(self) -> None:
        if self.image:
            self.image.contents.data = None  # (the data is shared memory, not malloc'd)
            self.image.contents.f.destroy_image(self.image)
            self.image = None

    def close(self) -> None:
        lib = xlib()
        if self.image:
            lib.XShmDetach(self.display, ctypes.byref(self.info))
            lib.XSync(self.display, 0)
            lib.shmdt(self.info.shmaddr)
            self._free_image()


# ---- hotkeys ---------------------------------------------------------------------------
KEYSYMS = {
    "SPACE": "space", "TAB": "Tab", "ESC": "Escape", "PAUSE": "Pause",
    "SCROLL": "Scroll_Lock", "HOME": "Home", "END": "End", "INSERT": "Insert",
    "DELETE": "Delete", "PAGEUP": "Prior", "PAGEDOWN": "Next", "LEFT": "Left",
    "UP": "Up", "RIGHT": "Right", "DOWN": "Down",
    **{f"F{i}": f"F{i}" for i in range(1, 25)},
    **{f"NUM{i}": f"KP_{i}" for i in range(10)},
    **{chr(c): chr(c).lower() for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): str(d) for d in range(10)},
}  # fmt: skip
_MODS = {"ctrl": CONTROL_MASK, "alt": MOD1_MASK, "shift": SHIFT_MASK, "win": MOD4_MASK}
# a grab names exact modifier states: Caps Lock and Num Lock must not stop a hotkey
_LOCK_VARIANTS = (0, LOCK_MASK, MOD2_MASK, LOCK_MASK | MOD2_MASK)


class _Relay(QObject):
    """Carries a key press's callback from the hotkey thread to the GUI thread."""

    pressed = Signal(object)


class X11Hotkeys(Hotkeys):
    """XGrabKey on the root window, on a connection of its own read by a thread; a key
    press runs its callback on the GUI thread (a queued Qt signal)."""

    def __init__(self, display_name: str | None = None) -> None:
        self._display_name = display_name
        self._display = 0
        self._root = 0
        self._lock = threading.Lock()
        self._grabs: dict[tuple[int, int], Callable[[], None]] = {}  # (keycode, mods) -> cb
        self._relay = _Relay()  # (made on the GUI thread: the signal is delivered there)
        self._relay.pressed.connect(self._fire)
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def _conn(self) -> int:
        if not self._display:
            self._display = open_display(self._display_name)
            self._root = xlib().XDefaultRootWindow(self._display)
        return self._display

    def register(self, spec: str, callback: Callable[[], None]) -> bool:
        from hud.hotkey import split_spec

        try:
            mods, key = split_spec(spec)
        except ValueError:
            return False
        if key not in KEYSYMS:
            return False
        lib = xlib()
        with self._lock:
            try:
                display = self._conn()
            except OSError:
                return False
            keysym = lib.XStringToKeysym(KEYSYMS[key].encode())
            keycode = lib.XKeysymToKeycode(display, keysym) if keysym else 0
            if not keycode:
                return False
            mask = 0
            for m in mods:
                mask |= _MODS[m]
            done: list[int] = []
            try:
                for lock in _LOCK_VARIANTS:
                    with trap(display):
                        lib.XGrabKey(display, keycode, mask | lock, self._root, 1,
                                     GRAB_MODE_ASYNC, GRAB_MODE_ASYNC)
                    done.append(lock)
            except XError:  # another program has it
                for lock in done:
                    lib.XUngrabKey(display, keycode, mask | lock, self._root)
                lib.XSync(display, 0)
                return False
            self._grabs[(keycode, mask)] = callback
        self._start()
        return True

    def _start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._halt.clear()
            self._thread = threading.Thread(target=self._read, name="hotkeys", daemon=True)
            self._thread.start()

    def _read(self) -> None:
        """Wait for key presses (select on the connection; nothing blocks inside Xlib, so
        a register from the GUI thread never waits long for the lock)."""
        lib = xlib()
        event = XEvent()
        fd = lib.XConnectionNumber(self._display)
        relevant = ~(LOCK_MASK | MOD2_MASK)
        while not self._halt.is_set():
            ready, _w, _x = select.select([fd], [], [], HOTKEY_POLL_S)
            if not ready:
                continue
            with self._lock:
                while self._display and lib.XPending(self._display):
                    lib.XNextEvent(self._display, ctypes.byref(event))
                    if event.type != KEY_PRESS:
                        continue
                    k = event.xkey
                    callback = self._grabs.get((int(k.keycode), int(k.state) & relevant & 0xFF))
                    if callback is not None:
                        self._relay.pressed.emit(callback)

    @staticmethod
    def _fire(callback: Callable[[], None]) -> None:
        callback()

    def unregister_all(self) -> None:
        with self._lock:
            if self._display:
                lib = xlib()
                for keycode, mask in self._grabs:
                    for lock in _LOCK_VARIANTS:
                        lib.XUngrabKey(self._display, keycode, mask | lock, self._root)
                lib.XSync(self._display, 0)
            self._grabs.clear()

    def uninstall(self) -> None:
        self.unregister_all()
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * HOTKEY_POLL_S + 1)
        with self._lock:
            if self._display:
                xlib().XCloseDisplay(self._display)
                self._display = 0
