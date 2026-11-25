"""What is left of the panel while it is folded away: a small tab on the screen's edge."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QWidget

from app.theme import COLOURS
from hud.platform import current
from i18n import _

RADIUS = 9.0


class EdgeHandle(QWidget):
    """A small tab on the screen's edge with the overlay's state as a dot. A click unfolds
    the panel, a right click offers the rest. The overlay never reads it (it is excluded
    from screen capture like the panel)."""

    def __init__(self, panel) -> None:  # noqa: ANN001 - app.panel.window.MainWindow
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.panel = panel
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.left = False
        self.hover = False
        self.dot = QColor(COLOURS["off"])
        self.logo = panel.icon.pixmap(QSize(16, 16))

    def dock(self, rect: QRect, left: bool) -> None:
        self.left = left
        if self.geometry() != rect:
            self.setGeometry(rect)
        self.update()

    def set_dot(self, colour: str) -> None:
        c = QColor(colour)
        if c != self.dot:
            self.dot = c
            self.update()

    def showEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        super().showEvent(event)
        current().exclude_from_capture(self)

    def enterEvent(self, _event) -> None:  # noqa: N802, ANN001 - Qt API
        self.hover = True
        self.update()

    def leaveEvent(self, _event) -> None:  # noqa: N802, ANN001 - Qt API
        self.hover = False
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.panel.unfold()

    def contextMenuEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        menu = QMenu(self)
        menu.addAction(_("Unfold the panel"), self.panel.unfold)
        menu.addAction(_("Hide / show translations"), self.panel.toggle_translations)
        menu.addSeparator()
        menu.addAction(_("Close the app"), self.panel.quit)
        menu.exec(event.globalPos())

    def paintEvent(self, _event) -> None:  # noqa: N802, ANN001 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        # rounded on the inner side: the outer corners lie beyond the screen's edge
        r = r.adjusted(-RADIUS, 0, 0, 0) if self.left else r.adjusted(0, 0, RADIUS, 0)
        path = QPainterPath()
        path.addRoundedRect(r, RADIUS, RADIUS)
        p.fillPath(path, QColor(17, 21, 31, 235 if self.hover else 175))
        p.setPen(QPen(QColor(255, 255, 255, 70 if self.hover else 40), 1.0))
        p.drawPath(path)
        w, h = self.width(), self.height()
        p.drawPixmap(QPoint(w // 2 - 8, 8), self.logo)
        p.setPen(QColor(230, 235, 245, 240 if self.hover else 170))
        f = p.font()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "›" if self.left else "‹")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.dot)
        p.drawEllipse(QPointF(w / 2, h - 12), 3.5, 3.5)
