"""The game's picture in its window: a 16:9 picture between black bars (full screen on a
21:9 monitor) is found, and the HUD reads and draws within it."""

from __future__ import annotations

import numpy as np

from hud.picture import UNKNOWN, find_picture

ASPECT = 1.7778


def scene(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8)


def pillarboxed(sw: int = 2560, sh: int = 1080) -> np.ndarray:
    frame = np.zeros((sh, sw, 3), np.uint8)
    pw = round(sh * ASPECT)
    x = (sw - pw) // 2
    frame[:, x : x + pw] = scene(sh, pw)
    return frame


def test_a_picture_between_bars_is_found():
    frame = pillarboxed()
    assert find_picture(frame, ASPECT) == (320, 0, 1920, 1080)
    frame[1000:1060, :] = 200  # a video player's controls across the bottom
    frame[5:40, 40:600] = 230  # ...and its title
    assert find_picture(frame, ASPECT) == (320, 0, 1920, 1080)
    tall = np.zeros((1200, 1600, 3), np.uint8)  # letterbox: bars above and below
    tall[150:1050] = scene(900, 1600)
    assert find_picture(tall, ASPECT) == (0, 150, 1600, 900)


def test_no_bars_no_picture():
    assert find_picture(scene(1080, 1920), ASPECT) is None  # the picture's own shape
    assert find_picture(scene(1080, 2560), ASPECT) is None  # a wide game fills the window
    frame = pillarboxed()
    frame[400:600, 30:200] = 120  # something in a bar: it is no bar
    assert find_picture(frame, ASPECT) is None
    assert find_picture(np.zeros((1080, 2560, 3), np.uint8), ASPECT) == UNKNOWN  # all black


def test_the_picture_is_a_crop_of_the_newest_frame_one_object_per_frame():
    from hud.screen_source import ScreenSource

    src = ScreenSource(start=False)  # (no capture thread: frames handed in by the test)
    whole = pillarboxed()
    src.set_area((0, 0, 2560, 1080))
    src._frame, src._frame_area, src.seq = whole, (0, 0, 2560, 1080), 1
    assert src.grab((0, 0, 2560, 1080)) is whole
    part = src.grab((320, 0, 1920, 1080))
    assert part.shape == (1080, 1920, 3) and np.array_equal(part, whole[:, 320:2240])
    assert src.grab((320, 0, 1920, 1080)) is part  # the same frame: no new object
    src.seq = 2  # a new frame came
    src._frame = whole.copy()
    assert src.grab((320, 0, 1920, 1080)) is not part
    assert src.grab((3000, 0, 100, 100)) is None  # outside: it becomes what is captured
    assert src._area == (3000, 0, 100, 100)
