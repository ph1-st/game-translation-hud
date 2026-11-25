"""The app's dialogs: choosing the game's window, adding a game, a game's settings,
reviewing the names and terms a database is made with, and a license to accept."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from app import ONE_GAME
from app.config import GameEntry
from app.library import GENERIC, Library
from app.widgets import Task, button
from app.windows import choices
from i18n import _
from translate.languages import SOURCE_NATIVE, SOURCES


def _languages_box(selected: str = "") -> QComboBox:
    box = QComboBox()
    for code, name in SOURCES.items():
        box.addItem(f"{SOURCE_NATIVE[code]} — {_(name)}", code)
    box.setCurrentIndex(max(0, box.findData(selected)))
    return box


def _ok_cancel(dialog: QDialog, ok_text: str = "") -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                           | QDialogButtonBox.StandardButton.Cancel)
    if ok_text:
        box.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    return box


class LicenseDialog(QDialog):
    """A license to read and accept before something is installed: what it is about
    (*intro*, rich text), the license's own *text*, and Accept or Cancel."""

    def __init__(self, title: str, intro: str, text: str, accept: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lead = QLabel(intro)
        lead.setWordWrap(True)
        lead.setOpenExternalLinks(True)
        lay.addWidget(lead)
        body = QPlainTextEdit(text)
        body.setReadOnly(True)
        body.setMinimumSize(560, 300)
        lay.addWidget(body, 1)
        lay.addWidget(_ok_cancel(self, accept))


class WindowPicker(QDialog):
    """Choose the game's window from the windows open now."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle(_("Choose the game window"))
        self.resize(600, 400)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(_("Start the game, then pick its window:")))
        self.list = QListWidget()
        lay.addWidget(self.list)
        row = QHBoxLayout()
        refresh = button(_("Refresh"))
        refresh.clicked.connect(self.fill)
        row.addWidget(refresh)
        row.addStretch(1)
        row.addWidget(_ok_cancel(self))
        lay.addLayout(row)
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        self.fill()

    def fill(self) -> None:
        self.list.clear()
        for c in choices():
            item = QListWidgetItem(c.label())
            item.setData(Qt.ItemDataRole.UserRole, c)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def choice(self):  # noqa: ANN201 - app.windows.Choice | None
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None


def pick_window(parent: QWidget):  # noqa: ANN201 - app.windows.Choice | None
    """The window the player picks, or None."""
    d = WindowPicker(parent)
    try:
        return d.choice() if d.exec() else None
    finally:
        d.deleteLater()


class AddGameDialog(QDialog):
    """A game for the library: its name, preset, window and language."""

    def __init__(self, library: Library, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Add a game"))
        self.setMinimumWidth(460)
        self.picked = None
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText(_("My game"))
        form.addRow(_("Name"), self.name)
        self.profile = QComboBox()
        for pid, title, dlg in library.profiles():
            self.profile.addItem(title + (f"  ({_('preset')})" if dlg else ""), pid)
        self.profile.setCurrentIndex(self.profile.count() - 1)
        form.addRow(_("Preset"), self.profile)
        row = QHBoxLayout()
        self.window = QLabel("—")
        self.window.setObjectName("note")
        pick = button(_("Choose…"), _("Pick the game's window (the game must be open)"))
        pick.clicked.connect(self.pick)
        row.addWidget(self.window, 1)
        row.addWidget(pick)
        form.addRow(_("Window"), row)
        self.lang = _languages_box()
        form.addRow(_("Text in"), self.lang)
        lay.addLayout(form)
        lay.addWidget(_ok_cancel(self, _("Add")))

    def pick(self) -> None:
        choice = pick_window(self)
        if choice is not None:
            self.picked = choice
            self.window.setText(choice.label())
            if not self.name.text().strip():
                self.name.setText(choice.title)

    def add(self, library: Library) -> GameEntry:
        """The game, added to *library*."""
        name = self.name.text().strip() or (self.picked.title if self.picked else _("My game"))
        game = library.add_game(
            name, profile=self.profile.currentData() or GENERIC,
            window_title=self.picked.title if self.picked else "",
            process_name=self.picked.own_program if self.picked else "",
        )  # fmt: skip
        game.source_lang = self.lang.currentData() or ""
        library.settings.save()
        return game


class GameDialog(QDialog):
    """A game's settings: name, window, the player's name, the language the game shows;
    remove it."""

    def __init__(self, library: Library, game: GameEntry, parent: QWidget | None = None,
                 ocr_device: str = "auto") -> None:
        super().__init__(parent)
        self.library, self.game, self.ocr_device = library, game, ocr_device
        self.removed = False
        self.task: Task | None = None
        self.setWindowTitle(game.name)
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(game.name)
        form.addRow(_("Name"), self.name)
        wrow = QHBoxLayout()
        self.window = QLineEdit(game.window_title)
        own = [] if game.window_title else library.profile(game).window.title_patterns
        self.window.setPlaceholderText(" / ".join(p.replace("\\s*", " ").replace("\\", "")
                                                  for p in own) or _("the game window's title"))
        pick = button(_("Choose…"), _("Pick the game's window (the game must be open)"))
        pick.clicked.connect(self.pick)
        wrow.addWidget(self.window, 1)
        wrow.addWidget(pick)
        form.addRow(_("Window"), wrow)
        self.process = game.process_name
        self.player = QLineEdit(game.player_name)
        self.player.setPlaceholderText(_("your character's name (optional)"))
        form.addRow(_("Player"), self.player)
        lrow = QHBoxLayout()
        self.lang = _languages_box(library.source_lang(game))
        self.detect_btn = button(_("Detect"), _("Read the game's window to find its language"))
        self.detect_btn.clicked.connect(self.detect)
        lrow.addWidget(self.lang, 1)
        lrow.addWidget(self.detect_btn)
        form.addRow(_("Text in"), lrow)
        if ONE_GAME:
            form.setRowVisible(lrow, False)
        lay.addLayout(form)
        row = QHBoxLayout()
        if not ONE_GAME:
            remove = button(_("Remove game…"))
            remove.setObjectName("danger")
            remove.clicked.connect(self.remove)
            row.addWidget(remove)
        row.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.save)
        box.rejected.connect(self.reject)
        row.addWidget(box)
        lay.addLayout(row)

    def pick(self) -> None:
        choice = pick_window(self)
        if choice is not None:
            self.window.setText(choice.title)
            self.process = choice.own_program

    def save(self) -> None:
        g = self.game
        g.name = self.name.text().strip() or g.name
        title = self.window.text().strip()
        if title != g.window_title:
            g.window_title = title
            g.process_name = self.process if title else ""
        g.player_name = self.player.text().strip()
        if not ONE_GAME:
            g.source_lang = self.lang.currentData() or g.source_lang
        self.library.settings.save()
        self.accept()

    def detect(self) -> None:
        from app.langdetect import detect, grab

        if self.task is not None:
            return
        found = self.library.find_window(self.game)
        frame = grab(found) if found is not None else None
        if frame is None:
            QMessageBox.information(self, _("Detect"), _(
                "Open the game with some text on screen (and choose its window) first."))
            return
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText(_("Reading…"))
        ocr = dict(self.library.profile(self.game).ocr)

        def end() -> None:
            self.task = None
            self.detect_btn.setEnabled(True)
            self.detect_btn.setText(_("Detect"))

        def done(found_lang) -> None:  # noqa: ANN001 - app.langdetect.Found
            end()
            if found_lang.lang is None:
                QMessageBox.information(self, _("Detect"), _(
                    "No Japanese, Chinese or Korean text found on the game's screen."))
                return
            self.lang.setCurrentIndex(max(0, self.lang.findData(found_lang.lang)))

        def failed(why: str) -> None:
            end()
            QMessageBox.warning(self, _("Detect"), why)

        self.task = Task(lambda _t: detect(frame, ocr, self.ocr_device))
        self.task.done.connect(done)
        self.task.failed.connect(failed)
        self.task.start()

    def remove(self) -> None:
        g = self.game
        box = QMessageBox(self)
        box.setWindowTitle(_("Remove game"))
        box.setText(_("Remove {game}?").format(game=g.name))
        keep = box.addButton(_("Remove, keep its files"), QMessageBox.ButtonRole.AcceptRole)
        delete = box.addButton(_("Remove and delete its files"),
                               QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() not in (keep, delete):
            return
        self.library.remove_game(g, delete_files=box.clickedButton() is delete)
        self.removed = True
        self.accept()
