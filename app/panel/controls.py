"""What the panel holds while the HUD runs: start and stop, the game window, the cards, and
(folded away) diagnostics and the rest of the settings."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from app.dialogs import pick_window
from app.panel.cards import (
    DiagnosticsCard,
    DialogueCard,
    LiveCard,
    MoreCard,
    ReaderCard,
    StreamCard,
    TranslateCard,
)
from app.widgets import button, link_button, note, scrolled, tinted
from i18n import _


class Fold(QWidget):
    """A ▸ title that shows or hides what is under it."""

    def __init__(self, title: str, inner: QWidget) -> None:
        super().__init__()
        self.title, self.inner = title, inner
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toggle = link_button("")
        self.toggle.setCheckable(True)
        self.toggle.toggled.connect(self._shown)
        lay.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(inner)
        self._shown(False)

    def set_title(self, title: str) -> None:
        self.title = title
        self._shown(self.toggle.isChecked())

    def open(self, on: bool = True) -> None:
        self.toggle.setChecked(on)

    def _shown(self, on: bool) -> None:
        self.inner.setVisible(on)
        self.toggle.setText(("▾  " if on else "▸  ") + self.title)


class HudControls(QWidget):
    def __init__(self, win) -> None:  # noqa: ANN001 - app.panel.window.MainWindow
        super().__init__()
        self.win = win
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 2, 6, 6)
        lay.setSpacing(10)
        brow = QHBoxLayout()
        self.start_btn = button("▶  " + _("Start"), primary=True)
        self.start_btn.setMinimumHeight(38)
        self.start_btn.clicked.connect(win.toggle_session)
        self.show_btn = button(_("Hide translations"), _("Same as the hotkey"))
        self.show_btn.setMinimumHeight(38)
        self.show_btn.clicked.connect(win.toggle_translations)
        brow.addWidget(self.start_btn, 1)
        brow.addWidget(self.show_btn, 1)
        lay.addLayout(brow)
        wrow = QHBoxLayout()
        self.window_label = note()
        choose = link_button(_("Choose window…"))
        choose.clicked.connect(self._choose_window)
        wrow.addWidget(self.window_label, 1)
        wrow.addWidget(choose, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(wrow)
        self.translate = TranslateCard(win)
        self.dialogue = DialogueCard(win)
        self.live = LiveCard(win)
        self.reader = ReaderCard(win)
        self.diagnostics = DiagnosticsCard(win)
        self.more = MoreCard(win)
        self.stream = StreamCard(win)
        for c in (self.translate, self.dialogue, self.live, self.reader):
            lay.addWidget(c)
        self.diag_fold = Fold(_("Diagnostics"), self.diagnostics)
        self.diag_fold.setVisible(False)
        lay.addWidget(self.diag_fold)
        self.stream_fold = Fold(_("Streaming (OBS)"), self.stream)
        lay.addWidget(self.stream_fold)
        lay.addWidget(Fold(_("More"), self.more))
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(inner))

    def refresh(self) -> None:
        for c in (self.translate, self.dialogue, self.live, self.reader, self.more,
                  self.stream):
            c.refresh()
        self.refresh_window()
        self.set_running(self.win.session.running, self.win.session.busy)

    def refresh_window(self) -> None:
        g = self.win.settings.current_game
        if g is None:
            self.window_label.setText("")
            return
        lib = self.win.library
        if not lib.has_window(g):
            self.window_label.setText(_("No game window chosen"))
            return
        hud = self.win.session.hud
        found = hud.window if hud is not None else lib.find_window(g)
        self.window_label.setText(tinted("✔", "ok") + " " + html.escape(found.title)
                                  if found is not None else _("Waiting for the game window"))

    def set_running(self, running: bool, busy: bool = False) -> None:
        if busy:
            self.start_btn.setText(_("Starting…"))
            self.start_btn.setEnabled(False)
        else:
            self.start_btn.setEnabled(True)
            self.start_btn.setText("■  " + _("Stop") if running else "▶  " + _("Start"))
        self.show_btn.setEnabled(running)
        hud = self.win.session.hud
        hidden = hud is not None and not hud.overlay.enabled
        self.show_btn.setText(_("Show translations") if hidden else _("Hide translations"))

    def update_diagnostics(self, data: dict | None, problem: str = "") -> None:
        visible = bool(data and data.get("window")) or bool(problem)
        self.diag_fold.setVisible(visible)
        if not visible:
            return
        if self.diagnostics.update_from(data, problem):
            self.diag_fold.open()  # (a new problem opens it)
        ocr = (data or {}).get("ocr") if isinstance((data or {}).get("ocr"), dict) else {}
        device = str(ocr.get("device") or "").lower()
        label = {"cpu": _("CPU"), "": _("starting")}.get(device, _("GPU"))
        self.diag_fold.set_title(_("Diagnostics") + f" · {label}")

    def _choose_window(self) -> None:
        g = self.win.settings.current_game
        if g is None:
            return
        choice = pick_window(self)
        if choice is not None:
            g.window_title, g.process_name = choice.title, choice.own_program
            self.win.settings.save()
            self.refresh()
            self.win.restart_overlay(start=True)
