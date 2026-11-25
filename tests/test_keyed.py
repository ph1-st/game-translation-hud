"""The overlay with a colour key (Windows 10 keeps only such a window out of the capture):
where it draws, the frame blended with its layers exactly as the desktop would blend the
translucent window; everywhere else the key colour, which shows the live game."""

from __future__ import annotations

import numpy as np

from hud.overlay import OverlayItem, render_text
from hud.overlay.keyed import KEY, SHOWN, KeyedComposer, overlapping
from tools.recorder import blend_layers


def _layers(rng: np.random.Generator) -> list:
    ramp = np.zeros((20, 30, 4), np.uint8)
    ramp[:, :, 3] = np.linspace(0, 255, 30).astype(np.uint8)  # clear to opaque, left to right
    ramp[:, :, 0] = ramp[:, :, 3]  # (premultiplied red)
    solid = np.zeros((10, 10, 4), np.uint8)
    solid[...] = (0, 0, 255, 255)
    soft = np.zeros((8, 8, 4), np.uint8)
    soft[...] = (0, 60, 0, 128)
    return [((10, 10, ramp), 1.0), ((25, 15, solid), 0.5), ((60, 40, soft), 1.0)]


def test_groups_are_the_layers_that_overlap():
    boxes = [(0, 0, 10, 10), (30, 30, 40, 40), (5, 5, 15, 15), (14, 14, 31, 31)]
    assert overlapping(boxes) == [[0, 1, 2, 3]]  # (the fourth links the second to the rest)
    assert overlapping([(0, 0, 10, 10), (10, 0, 20, 10)]) == [[0], [1]]  # touching is apart


def test_tiles_blend_as_the_desktop_would_and_leave_the_rest_clear():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (60, 80, 3)).astype(np.uint8)
    layers = _layers(rng)
    tiles = KeyedComposer().tiles(frame, layers)
    assert [(x, y, t.shape[:2]) for x, y, t in tiles] == [(10, 10, (20, 30)), (60, 40, (8, 8))]
    desktop = blend_layers(frame, layers)
    drawn = np.zeros(frame.shape[:2], bool)
    for (x, y, rgba), fade in layers:
        h, w = rgba.shape[:2]
        drawn[y : y + h, x : x + w] |= np.rint(rgba[:, :, 3] * fade) >= SHOWN
    for x, y, tile in tiles:
        h, w = tile.shape[:2]
        mine = drawn[y : y + h, x : x + w]
        assert (tile[:, :, 3] == np.where(mine, 255, 0)).all()
        assert (tile[mine][:, :3] == desktop[y : y + h, x : x + w][mine]).all()
        assert not tile[~mine].any()  # (clear: the window's key shows the game there)
    assert not drawn[10:30, 10].any() and drawn[10:30, 11].all()  # the ramp's clear edge


def test_an_overlay_pixel_in_the_key_colour_is_nudged_off_it():
    frame = np.zeros((4, 4, 3), np.uint8)
    magenta = np.zeros((4, 4, 4), np.uint8)
    magenta[...] = (*KEY, 255)
    ((_x, _y, tile),) = KeyedComposer().tiles(frame, [((0, 0, magenta), 1.0)])
    bgr = tile[:, :, :3]
    assert (tile[:, :, 3] == 255).all()
    keyed = (bgr[:, :, 2] == KEY[0]) & (bgr[:, :, 1] == KEY[1]) & (bgr[:, :, 0] == KEY[2])
    assert not keyed.any()


def _shown(window, w: int, h: int) -> np.ndarray:
    """What the window paints, as RGB."""
    from PySide6.QtGui import QImage

    img = window.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
    return arr[:h, : w * 3].reshape(h, w, 3).copy()


def test_the_keyed_window_paints_the_key_where_it_is_clear():
    import shiboken6

    from hud.overlay import OverlayWindow

    rng = np.random.default_rng(5)
    frame = rng.integers(20, 230, (90, 320, 3)).astype(np.uint8)
    item = OverlayItem(
        (20, 30, 280, 24), lines=["Texture Quality"], font_px=18, row_top=34, row_h=16,
        align="left", style="label", fg=(240, 240, 240), edge=(30, 30, 30),
    )  # fmt: skip
    text = render_text(item)
    window = OverlayWindow(colour_key=True)
    try:
        window.resize(320, 90)
        window._frame = frame
        window.snapshot = lambda: (window.pixels_version, [(text, 1.0)])
        got = _shown(window, 320, 90)
        desktop = blend_layers(frame, [(text, 1.0)])[:, :, ::-1]  # (RGB)
        key = (got == KEY).all(axis=2)
        x, y, rgba = text
        drawn = np.zeros(frame.shape[:2], bool)
        drawn[y : y + rgba.shape[0], x : x + rgba.shape[1]] = rgba[:, :, 3] >= SHOWN
        assert (key == ~drawn).all()  # the key exactly where the overlay is clear
        assert (got[drawn] == desktop[drawn]).all()  # elsewhere: as the desktop would blend
        # a newer frame, the same under the text: the tiles stay as they were made
        tiles = window.keyed_tiles()
        same = frame.copy()
        same[0:5, 0:5] = 0
        window._frame = same
        window._scene_moved(same)
        assert window.keyed_tiles() is tiles
        # the scene under the text moved: they are made again, on the new scene
        moved = same.copy()
        moved[40:50, 30:60] = 255 - moved[40:50, 30:60]
        window._frame = moved
        window._scene_moved(moved)
        assert window.keyed_tiles() is not tiles
        window.set_enabled(False)  # off: nothing but the key
        assert (_shown(window, 320, 90) == KEY).all()
    finally:
        window.close_builder()
        shiboken6.delete(window)


def test_tests_and_tools_draw_translucent():
    import shiboken6

    from hud.overlay import OverlayWindow
    from hud.overlay.window import translucent_refused

    assert not translucent_refused()  # (offscreen: no window to ask about)
    window = OverlayWindow()
    try:
        assert not window.colour_key
    finally:
        window.close_builder()
        shiboken6.delete(window)
