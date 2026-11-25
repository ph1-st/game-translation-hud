"""Choosing a game's dialogue area (set-region mode): the game window is dimmed and the
player drags a box over the place where the game's text appears. The area is kept as
fractions of the game's picture (inside black bars, if the picture has them), like every
region of a profile.

The picker is excluded from screen capture (the overlay never reads it) and stays on top
of the game until the player is done (Enter or the ✔ button) or cancels (Esc)."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QPushButton, QWidget

from hud.platform import current
from hud.screens import screen_at, to_qt
from i18n import _

Rect = tuple[int, int, int, int]
MIN_PX = 12  # a smaller drag is a click, not an area


class RegionPicker(QWidget):
    done = Signal(object)  # (x, y, w, h) as fractions of the picture, or None

    def __init__(self, window: Rect, picture: Rect, current: tuple | None, prompt: str) -> None:
        """*window*: the game window's client area, *picture*: the game's picture in it
        (both in physical screen pixels); *current*: the area chosen before, if any."""
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.prompt = prompt
        geo = to_qt(window, screen_at(window))
        self.setGeometry(geo)
        sx = geo.width() / max(1, window[2])
        sy = geo.height() / max(1, window[3])
        self.picture = QRect(
            round((picture[0] - window[0]) * sx), round((picture[1] - window[1]) * sy),
            round(picture[2] * sx), round(picture[3] * sy),
        )  # fmt: skip
        self.sel: QRect | None = None
        if current:
            x, y, w, h = current
            p = self.picture
            self.sel = QRect(round(p.x() + x * p.width()), round(p.y() + y * p.height()),
                             round(w * p.width()), round(h * p.height()))
        self._start: QPoint | None = None
        self._result: tuple | None = None
        self.ok = QPushButton("✔  " + _("Use this area"), self)
        self.ok.setObjectName("primary")
        self.ok.clicked.connect(self.accept)
        self.cancel = QPushButton(_("Cancel"), self)
        self.cancel.clicked.connect(self.close)
        self._place_buttons()

    # ---- the result ----------------------------------------------------------------------
    def relative(self) -> tuple[float, float, float, float] | None:
        if self.sel is None or self.sel.width() < MIN_PX or self.sel.height() < MIN_PX:
            return None
        r = self.sel.intersected(self.picture)
        p = self.picture
        return ((r.x() - p.x()) / p.width(), (r.y() - p.y()) / p.height(),
                r.width() / p.width(), r.height() / p.height())

    def accept(self) -> None:
        self._result = self.relative()
        if self._result is not None:
            self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.done.emit(self._result)
        super().closeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        current().exclude_from_capture(self)  # (the overlay never reads the picker)
        self.activateWindow()
        self.setFocus()

    # ---- drawing and dragging --------------------------------------------------------------
    def _place_buttons(self) -> None:
        self.ok.adjustSize()
        self.cancel.adjustSize()
        y = self.picture.y() + 64
        mid = self.picture.center().x()
        self.ok.move(mid - self.ok.width() - 6, y)
        self.cancel.move(mid + 6, y)
        self.ok.setEnabled(self.relative() is not None)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        shade = QPainterPath()
        shade.addRect(QRectF(self.rect()))
        if self.sel is not None:
            hole = QPainterPath()
            hole.addRect(QRectF(self.sel))
            shade = shade.subtracted(hole)
        p.fillPath(shade, QColor(10, 14, 22, 150))
        if self.sel is not None:
            p.setPen(QPen(QColor(95, 170, 255), 2))
            p.drawRect(self.sel.adjusted(0, 0, -1, -1))
            rel = self.relative()
            if rel is not None:
                p.setPen(QColor(220, 230, 245))
                p.drawText(self.sel.adjusted(6, 4, 0, 0),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                           f"{rel[2] * 100:.0f}% × {rel[3] * 100:.0f}%")
        f = QFont(self.font())
        f.setPixelSize(17)
        f.setBold(True)
        p.setFont(f)
        box = QRect(self.picture.x() + 20, self.picture.y() + 14, self.picture.width() - 40, 44)
        p.setPen(QColor(255, 255, 255))
        p.drawText(box, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.prompt)
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self.sel = QRect(self._start, self._start)
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._start is not None:
            self.sel = QRect(self._start, event.position().toPoint()).normalized()
            self._place_buttons()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._start is not None:
            self.sel = QRect(self._start, event.position().toPoint()).normalized()
            self._start = None
            self._place_buttons()
            self.update()

    def mouseDoubleClickEvent(self, _event) -> None:  # noqa: N802 - Qt API
        self.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()


def picture_of(window, aspect: float | None) -> Rect:  # noqa: ANN001 - hud.platform.WindowRef
    """The game's picture in its window: between black bars when it has some."""
    rect = window.client
    if not aspect:
        return rect
    from app.langdetect import grab
    from hud.picture import find_picture

    frame = grab(window)
    if frame is None:
        return rect
    found = find_picture(frame, aspect)
    if isinstance(found, tuple):
        x, y, w, h = found
        return rect[0] + x, rect[1] + y, w, h
    return rect
