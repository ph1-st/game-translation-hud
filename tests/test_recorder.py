"""The recording shows what the player sees: the overlay's own pixels on the game frame."""

from __future__ import annotations

import numpy as np

from hud.overlay import OverlayItem, paint_text, render_text
from tools.recorder import blend_layers


def test_the_recording_shows_the_english_as_the_window_paints_it():
    from PySide6.QtGui import QImage, QPainter

    rng = np.random.default_rng(1)
    frame = rng.integers(40, 200, (120, 400, 3)).astype(np.uint8)
    item = OverlayItem(
        (40, 40, 300, 24), lines=["Chapter 1: Charles Village"], font_px=18, row_top=44,
        row_h=16, align="left", style="label", fg=(205, 186, 80), edge=(30, 30, 30),
    )  # fmt: skip
    recorded = blend_layers(frame, [(render_text(item), 1.0)])
    # the window: the same item painted straight onto the frame by Qt
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    img = QImage(rgb.data, 400, 120, 1200, QImage.Format.Format_RGB888).copy()
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    paint_text(p, item, 1.0)
    p.end()
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(120, img.bytesPerLine())[:, :1200]
    painted = arr.reshape(120, 400, 3)[:, :, ::-1]
    diff = np.abs(recorded.astype(np.int16) - painted.astype(np.int16))
    assert diff.mean() < 0.5 and np.percentile(diff, 99.9) <= 6
    assert np.abs(recorded.astype(np.int16) - frame).max() > 100  # the English is there


def test_a_fading_layer_is_recorded_faded():
    frame = np.full((10, 10, 3), 100, np.uint8)
    white = np.zeros((10, 10, 4), np.uint8)
    white[...] = (255, 255, 255, 255)  # opaque white, premultiplied
    half = blend_layers(frame, [((0, 0, white), 0.5)])
    assert abs(int(half[5, 5, 0]) - 178) <= 1  # 0.5 * 255 + 0.5 * 100
