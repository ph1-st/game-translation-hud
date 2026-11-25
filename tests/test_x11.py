"""The X11 platform against a real X server (run under Xvfb: the tests are marked x11 and
skipped without $DISPLAY). Windows are made with Xlib directly: a "game" window with known
pixels, and an override-redirect "overlay" window over it."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.x11

RED, GREEN = 0xFF0000, 0x00FF00  # (TrueColor pixel values: 0xRRGGBB)
CW_OVERRIDE_REDIRECT = 1 << 9
PROP_MODE_REPLACE = 0
SHAPE_INPUT = 2


class XSetWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("background_pixmap", ctypes.c_ulong), ("background_pixel", ctypes.c_ulong),
        ("border_pixmap", ctypes.c_ulong), ("border_pixel", ctypes.c_ulong),
        ("bit_gravity", ctypes.c_int), ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int), ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong), ("save_under", ctypes.c_int),
        ("event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int), ("colormap", ctypes.c_ulong),
        ("cursor", ctypes.c_ulong),
    ]  # fmt: skip


@pytest.fixture(scope="module")
def x():
    """Xlib with the calls the tests need to make windows, on a connection of their own."""
    from hud.platform import x11

    lib = x11.xlib()
    raw = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
    ul, i, vp = ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p
    sigs = {
        "XCreateSimpleWindow": ([vp, ul, i, i, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                 ul, ul], ul),
        "XMapRaised": ([vp, ul], i), "XDestroyWindow": ([vp, ul], i),
        "XStoreName": ([vp, ul, ctypes.c_char_p], i),
        "XChangeProperty": ([vp, ul, ul, ul, i, i, ctypes.c_char_p, i], i),
        "XChangeWindowAttributes": ([vp, ul, ul, ctypes.POINTER(XSetWindowAttributes)], i),
        "XGetImage": ([vp, ul, i, i, ctypes.c_uint, ctypes.c_uint, ul, i], vp),
    }  # fmt: skip
    for name, (args, res) in sigs.items():
        fn = getattr(raw, name)
        fn.argtypes, fn.restype = args, res
    display = x11.open_display()
    yield type("X", (), {"lib": lib, "raw": raw, "display": display,
                         "root": lib.XDefaultRootWindow(display), "x11": x11})  # fmt: skip
    lib.XCloseDisplay(display)


def make_window(x, title: str, rect, pixel: int, override: bool = False,
                wm_class: bytes = b"") -> int:  # fmt: skip
    rx, ry, w, h = rect
    win = x.raw.XCreateSimpleWindow(x.display, x.root, rx, ry, w, h, 0, 0, pixel)
    if override:
        attrs = XSetWindowAttributes(override_redirect=1)
        x.raw.XChangeWindowAttributes(x.display, win, CW_OVERRIDE_REDIRECT, ctypes.byref(attrs))
    x.raw.XStoreName(x.display, win, title.encode())
    if wm_class:
        from hud.platform.x11 import XA_STRING, XA_WM_CLASS

        x.raw.XChangeProperty(x.display, win, XA_WM_CLASS, XA_STRING, 8, PROP_MODE_REPLACE,
                              wm_class, len(wm_class))  # fmt: skip
    x.raw.XMapRaised(x.display, win)
    x.lib.XSync(x.display, 0)
    time.sleep(0.2)  # (the server paints the backgrounds)
    return win


def test_the_game_window_is_found_by_its_title_and_wine_program(x):
    from capture.profile import WindowMatch
    from hud.platform.x11 import X11Platform

    win = make_window(x, "AzurPromiliaJP", (50, 60, 320, 240), RED,
                      wm_class=b"azurpromiliajp.exe\0azurpromiliajp.exe\0")  # fmt: skip
    try:
        plat = X11Platform()
        found = {w.id: w for w in plat.windows()}
        assert win in found
        ref = found[win]
        assert ref.title == "AzurPromiliaJP" and ref.program == "azurpromiliajp.exe"
        assert ref.client == (50, 60, 320, 240)
        assert plat.find(WindowMatch(process_names=["AzurPromiliaJP.exe"])).id == win
        assert plat.find(WindowMatch(title_patterns=[r"Azur\s*Promilia"])).id == win
        assert plat.client_rect(win) == (50, 60, 320, 240)
    finally:
        x.raw.XDestroyWindow(x.display, win)
        x.lib.XSync(x.display, 0)
    assert X11Platform().client_rect(win) is None  # gone: never an X error


def test_a_minimized_window_still_mapped_is_not_on_the_screen(x):
    """KWin keeps a minimized window mapped (for thumbnails): its state says it is hidden."""
    from hud.platform import x11

    win = make_window(x, "AzurPromiliaJP", (50, 60, 320, 240), RED)
    atom = x.lib.XInternAtom(x.display, b"_NET_WM_STATE", 0)
    hidden = (ctypes.c_ulong * 1)(x.lib.XInternAtom(x.display, b"_NET_WM_STATE_HIDDEN", 0))
    wm_state = x.lib.XInternAtom(x.display, b"WM_STATE", 0)
    iconic = (ctypes.c_ulong * 2)(x11.ICONIC_STATE, 0)

    def set_prop(prop, kind, values, n):  # noqa: ANN001, ANN202
        x.raw.XChangeProperty(x.display, win, prop, kind, 32, PROP_MODE_REPLACE,
                              ctypes.cast(values, ctypes.c_char_p), n)  # fmt: skip
        x.lib.XSync(x.display, 0)

    try:
        plat = x11.X11Platform()
        assert plat.client_rect(win) == (50, 60, 320, 240)
        set_prop(atom, x11.XA_ATOM, hidden, 1)  # minimized (EWMH)
        assert plat.client_rect(win) is None
        assert win not in {w.id for w in plat.windows()}
        set_prop(atom, x11.XA_ATOM, None, 0)  # restored
        assert plat.client_rect(win) == (50, 60, 320, 240)
        set_prop(wm_state, wm_state, iconic, 2)  # minimized (ICCCM)
        assert plat.client_rect(win) is None
    finally:
        x.raw.XDestroyWindow(x.display, win)
        x.lib.XSync(x.display, 0)


def test_the_window_manager_list_is_used_when_there_is_one(x):
    from hud.platform import x11

    win = make_window(x, "Listed", (0, 0, 100, 100), RED)
    atom = x.lib.XInternAtom(x.display, b"_NET_CLIENT_LIST", 0)
    try:
        ids = (ctypes.c_ulong * 1)(win)
        x.raw.XChangeProperty(x.display, x.root, atom, x11.XA_WINDOW, 32, PROP_MODE_REPLACE,
                              ctypes.cast(ids, ctypes.c_char_p), 1)  # fmt: skip
        x.lib.XSync(x.display, 0)
        assert x11.top_windows(x.display, x11._Atoms(x.display)) == [win]
    finally:
        x.raw.XChangeProperty(x.display, x.root, atom, x11.XA_WINDOW, 32, PROP_MODE_REPLACE,
                              None, 0)  # fmt: skip
        x.raw.XDestroyWindow(x.display, win)
        x.lib.XSync(x.display, 0)


def screen_pixels(x, rect) -> np.ndarray:
    """What the screen shows there (the root window): the overlay included."""
    from hud.platform import x11

    ptr = x.raw.XGetImage(x.display, x.root, *rect, x11.ALL_PLANES, x11.Z_PIXMAP)
    image = ctypes.cast(ptr, ctypes.POINTER(x11.XImage))
    try:
        return x11._to_bgr(image.contents)
    finally:
        image.contents.f.destroy_image(image)


def test_the_capture_shows_the_game_never_the_overlay_over_it(x):
    from hud.platform.x11 import WindowGrabber

    game_rect = (100, 100, 400, 300)
    game = make_window(x, "Game", game_rect, RED)
    grabber, fresh = WindowGrabber(), WindowGrabber()
    overlay = 0
    try:
        frame, ok = grabber.grab(game_rect, game)  # (redirected from the first grab on)
        assert ok and frame.shape == (300, 400, 3) and (frame == (0, 0, 255)).all()
        overlay = make_window(x, "Overlay", (80, 80, 500, 400), GREEN, override=True)
        assert tuple(screen_pixels(x, (200, 200, 4, 4))[0, 0]) == (0, 255, 0)  # covered
        assert grabber.grab(game_rect, game) == (None, True)  # the game itself is unchanged
        under, ok = fresh.grab(game_rect, game)  # read anew, with the overlay on top
        assert ok and (under == (0, 0, 255)).all()  # the game's red, never the overlay's green
        part, ok = fresh.grab((150, 120, 50, 40), game)  # a part of the window
        assert ok and part.shape == (40, 50, 3)
        assert fresh.grab((0, 0, 50, 50), game) == (None, False)  # outside the window
        no_shm = WindowGrabber()  # an X server that cannot share memory: XGetImage reads
        lib = x.x11.xlib()
        real, lib.XShmQueryExtension = lib.XShmQueryExtension, lambda display: 0
        try:
            plain, ok = no_shm.grab(game_rect, game)
        finally:
            lib.XShmQueryExtension = real
            no_shm.close()
        assert ok and no_shm._shm is None and (plain == (0, 0, 255)).all()
    finally:
        grabber.close()
        fresh.close()
        for w in (game, overlay):
            if w:
                x.raw.XDestroyWindow(x.display, w)
        x.lib.XSync(x.display, 0)


def test_hotkeys_are_grabbed_and_fire_on_the_gui_thread(x, qt_metrics):
    from hud.platform.x11 import X11Hotkeys

    fired: list[str] = []
    a, b = X11Hotkeys(), X11Hotkeys()
    try:
        assert a.register("ctrl+shift+F8", lambda: fired.append("a"))
        assert not b.register("ctrl+shift+F8", lambda: fired.append("b"))  # taken
        assert not a.register("ctrl+NOSUCHKEY", lambda: None)
        tst = ctypes.util.find_library("Xtst")
        if tst:  # press it, as a person would
            xtst = ctypes.CDLL(tst)
            xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
                                               ctypes.c_ulong]  # fmt: skip
            keys = [x.lib.XKeysymToKeycode(x.display, x.lib.XStringToKeysym(n))
                    for n in (b"Control_L", b"Shift_L", b"F8")]  # fmt: skip
            for k in keys:
                xtst.XTestFakeKeyEvent(x.display, k, 1, 0)
            for k in reversed(keys):
                xtst.XTestFakeKeyEvent(x.display, k, 0, 0)
            x.lib.XSync(x.display, 0)
            deadline = time.monotonic() + 3.0
            while not fired and time.monotonic() < deadline:
                qt_metrics.processEvents()
                time.sleep(0.02)
            assert fired == ["a"]
        a.unregister_all()
        assert b.register("ctrl+shift+F8", lambda: None)  # given back: another may take it
    finally:
        a.uninstall()
        b.uninstall()


@pytest.mark.skipif(os.environ.get("QT_QPA_PLATFORM") != "xcb", reason="Qt on X11 only")
def test_the_overlay_is_override_redirect_and_click_through(x, qt_metrics):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    from hud.platform.x11 import X11Platform, XWindowAttributes

    plat = X11Platform()
    w = QWidget(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                | plat.overlay_flags())  # fmt: skip
    w.setGeometry(10, 10, 200, 100)
    w.show()
    assert plat.prepare_overlay(w)
    qt_metrics.processEvents()
    x.lib.XSync(x.display, 0)
    try:
        attrs = XWindowAttributes()
        assert x.lib.XGetWindowAttributes(x.display, int(w.winId()), ctypes.byref(attrs))
        assert attrs.override_redirect == 1
        ext = ctypes.CDLL(ctypes.util.find_library("Xext") or "libXext.so.6")
        ext.XShapeGetRectangles.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_int),
                                            ctypes.POINTER(ctypes.c_int)]  # fmt: skip
        ext.XShapeGetRectangles.restype = ctypes.c_void_p
        count, order = ctypes.c_int(-1), ctypes.c_int()
        rects = ext.XShapeGetRectangles(x.display, int(w.winId()), SHAPE_INPUT,
                                        ctypes.byref(count), ctypes.byref(order))  # fmt: skip
        if rects:
            x.lib.XFree(rects)
        assert count.value == 0  # no input region: every click goes to the game
    finally:
        w.close()
