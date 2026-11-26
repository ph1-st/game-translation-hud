"""Painting overlay items: the backdrops (:mod:`hud.backdrop`) and the text, the same way
for the window, the recorder and offline previews.

The text is drawn as outlined shapes in the colours of the Japanese it replaces: the
outline of the Japanese when it has one, else (light text) a soft dark halo that keeps it
readable on the glow. Right-to-left text is shaped and ordered by Qt (``addText`` lays
out bidirectional text) and starts at the right where the game's starts at the left.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPainterPath, QPen

from hud import backdrop
from hud.items import OverlayItem, painted_align
from hud.layout import PAD_X, cap_centre_baseline
from hud.looks import is_light
from hud.textwidth import hud_font

EDGE_W = 0.08  # outline width (each side of a stroke) per pixel of font size
EDGE_MIN, EDGE_MAX = 0.8, 2.2
PLAIN_FILL, PLAIN_EDGE = (255, 255, 255), (0, 0, 0)  # the player's plain white text...
PLAIN_EDGE_W = 0.8  # ...its outline this share of a game outline (a crisp edge, not a smudge)
HALO = (18, 20, 26)  # the soft shadow under light text whose Japanese has no outline
HALO_ALPHA = 0.16
HALO_WIDTH = 1.3  # ...this much wider than an outline would be
PROVISIONAL_ALPHA = 0.75  # a line translated live is shown this faint
MIN_FONT = 6

Layer = tuple[int, int, np.ndarray]  # x, y, premultiplied RGBA pixels


def qimage(rgba: np.ndarray) -> QImage:
    h, w = rgba.shape[:2]
    data = np.ascontiguousarray(rgba)
    return QImage(data.data, w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied).copy()


def paint_backdrop(p: QPainter, built: tuple[int, int, QImage] | None, alpha: float) -> None:
    if built is None:
        return
    x, y, img = built
    p.save()
    p.setOpacity(alpha)
    p.drawImage(QPointF(x, y), img)
    p.restore()


def text_lines(item: OverlayItem) -> list[tuple[float, float, float, float]]:
    """(left, right, top, bottom) of each line of *item* as painted, in frame pixels."""
    x, _y, w, _h = item.rect
    fm = QFontMetricsF(hud_font(max(MIN_FONT, item.font_px), item.stretch))
    align = painted_align(item)
    out = []
    for i, line in enumerate(item.lines):
        tw = fm.horizontalAdvance(line)
        if align == "left":
            tx = x + PAD_X
        elif align == "right":
            tx = x + w - PAD_X - tw
        else:
            tx = x + (w - tw) / 2.0
        top = item.row_top + i * item.pitch
        out.append((tx, tx + tw, top, top + item.row_h))
    return out


def paint_text(p: QPainter, item: OverlayItem, alpha: float = 1.0) -> None:
    """The lines of *item*, as outlined shapes."""
    if not item.lines:
        return
    font = hud_font(max(MIN_FONT, item.font_px), item.stretch)
    a = alpha * (PROVISIONAL_ALPHA if item.provisional else 1.0)
    fg, edge = (PLAIN_FILL, PLAIN_EDGE) if item.plain else (item.fg, item.edge)
    fill = QColor(*fg, int(255 * a))
    ew = min(EDGE_MAX, max(EDGE_MIN, EDGE_W * item.font_px))
    if item.plain:
        ew *= PLAIN_EDGE_W
    for line, (left, _right, top, _bottom) in zip(item.lines, text_lines(item), strict=True):
        path = QPainterPath()
        path.addText(QPointF(left, cap_centre_baseline(top, item.row_h, item.font_px)), font, line)
        if edge is not None:
            pen = QPen(QColor(*edge, int(255 * a)), 2.0 * ew)
        elif is_light(fg):
            pen = QPen(QColor(*HALO, int(255 * a * HALO_ALPHA)), 2.0 * HALO_WIDTH * ew)
        else:
            pen = None
        if pen is not None:
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.strokePath(path, pen)
        p.fillPath(path, fill)


def paint_solid(p: QPainter, item: OverlayItem, alpha: float = 1.0) -> None:
    x, y, w, h = item.rect
    p.fillRect(QRectF(x, y, w, h), QColor(*item.fg, int(255 * alpha)))


def build_layers(frame: np.ndarray | None, item: OverlayItem):  # noqa: ANN201
    """((x, y, QImage), (x, y, premultiplied RGBA)) of an item's backdrop; None when it
    has none."""
    if item.style == "solid":
        return None
    got = backdrop.build(frame, item, item.shine)
    if got is None:
        return None
    x, y, rgba = got
    return (x, y, qimage(rgba)), (x, y, rgba)


def build_backdrop(frame: np.ndarray | None, item: OverlayItem) -> tuple[int, int, QImage] | None:
    got = build_layers(frame, item)
    return got[0] if got is not None else None


def render_text(item: OverlayItem) -> Layer | None:
    """The text of *item* as premultiplied RGBA pixels, painted exactly as the window
    paints it (the window paints from it; the recorder puts it onto captured frames)."""
    if not item.lines:
        return None
    x, _y, w, _h = item.rect
    m = max(8, item.font_px)
    x0, y0 = x - m, int(item.row_top) - m
    cw = w + 2 * m
    ch = int((len(item.lines) - 1) * item.pitch + item.row_h) + 2 * m
    if cw < 2 or ch < 2:
        return None
    img = QImage(cw, ch, QImage.Format.Format_RGBA8888_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.translate(-x0, -y0)
    paint_text(p, item, 1.0)
    p.end()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=img.bytesPerLine() * ch)
    rgba = arr.reshape(ch, img.bytesPerLine())[:, : cw * 4].reshape(ch, cw, 4).copy()
    return x0, y0, rgba


def paint_all(p: QPainter, entries: list[tuple]) -> None:
    """Backdrops first (glass, then labels, then dialogue lines), then all text, so no
    text ends up under another item's backdrop. An entry is (item, backdrop, fade[, text
    already drawn into an image])."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    for styles in (("solid",), ("panel",), ("label",), ("text",)):
        for item, built, alpha, *_ in entries:
            if item.style not in styles or alpha <= 0.0:
                continue
            if item.style == "solid":
                paint_solid(p, item, alpha)
            else:
                paint_backdrop(p, built, alpha)
    for item, _built, alpha, *drawn in entries:
        if item.style not in ("label", "text") or alpha <= 0.0:
            continue
        if drawn and drawn[0] is not None:
            faint = PROVISIONAL_ALPHA if item.provisional else 1.0
            paint_backdrop(p, drawn[0], alpha * faint)
        else:
            paint_text(p, item, alpha)
