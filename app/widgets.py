"""Small pieces the app's windows share: a background task with progress, and buttons,
boxes and labels in the app's look."""

from __future__ import annotations

import html
import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.theme import COLOURS

log = logging.getLogger("app")


class Task(QObject):
    """Runs *fn(task, ...)* off the GUI thread; *fn* may call task.report(a, b) and look at
    task.cancel. The thread is a daemon: closing the app never waits for a download or a
    translation (what they finished is saved as they go). The signals arrive on the GUI
    thread: only there are widgets touched and settings saved."""

    progress = Signal(object, object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs) -> None:  # noqa: ANN001
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name="task", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def running(self) -> bool:
        return self._thread.is_alive()

    def report(self, a, b=None) -> None:  # noqa: ANN001
        try:
            self.progress.emit(a, b)
        except RuntimeError:  # the window is gone (the app is closing): nobody to tell
            pass

    def _run(self) -> None:
        try:
            result = self.fn(self, *self.args, **self.kwargs)
        except Exception as e:  # noqa: BLE001 - shown to the player
            log.exception("task failed")
            self._emit(self.failed, str(e) or e.__class__.__name__)
            return
        self._emit(self.done, result)

    @staticmethod
    def _emit(signal, value) -> None:  # noqa: ANN001
        try:
            signal.emit(value)
        except RuntimeError:  # the window is gone (the app is closing)
            pass


def note(text: str = "") -> QLabel:
    """A quiet, wrapping line of text (links open in the browser)."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName("note")
    lab.setOpenExternalLinks(True)
    return lab


def button(text: str, tip: str = "", primary: bool = False) -> QPushButton:
    b = QPushButton(text)
    if tip:
        b.setToolTip(tip)
    if primary:
        b.setObjectName("primary")
    return b


def link_button(text: str, tip: str = "") -> QToolButton:
    """A borderless button that looks like a link."""
    b = QToolButton()
    b.setObjectName("linkbtn")
    b.setText(text)
    if tip:
        b.setToolTip(tip)
    return b


def card(title: str = "") -> tuple[QFrame, QVBoxLayout]:
    """A rounded box with a small title: the sections of the app's windows."""
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 12)
    lay.setSpacing(8)
    if title:
        head = QLabel(title.upper())
        head.setObjectName("section")
        lay.addWidget(head)
    return frame, lay


def combo() -> QComboBox:
    """A combo box that fits a narrow panel (a long entry is cut, not the panel)."""
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(8)
    return c


def scrolled(inner: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidget(inner)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # the window shows through (setWidget gave the page a background of its own)
    area.viewport().setAutoFillBackground(False)
    inner.setAutoFillBackground(False)
    return area


def dot(state: str) -> str:
    """A coloured ● for rich text: ok, wait, off, problem, live."""
    return f"<span style='color:{COLOURS[state]}'>●</span>"


def tinted(text: str, state: str) -> str:
    """*text* (escaped) in a state's colour, for rich text."""
    return f"<span style='color:{COLOURS[state]}'>{html.escape(text)}</span>"


def path_text(path: Path | str) -> str:
    """A file path for a word-wrapped label: it may break after any folder."""
    return html.escape(str(path)).replace("\\", "\\​").replace("/", "/​")

