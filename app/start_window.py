"""The window the app opens with: the games, and for the one chosen the language the player
reads and its translation databases (download from a link, use a file, check for a newer
one) and the folder they are kept in. **Play** opens the
HUD (the panel at the side of the screen and the overlay); the panel's ◀ comes back here.

Every file in a game's database folder is checked: the databases in the player's language
are used together, every other file is listed with why it is not. Files dropped into the
folder by hand count as soon as they are there (the folder is looked at every two seconds).
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import i18n
from app import APP_NAME, HELP_URL, ONE_GAME
from app.config import GameEntry
from app.dialogs import AddGameDialog, GameDialog, pick_window
from app.library import GENERIC
from app.theme import COLOURS
from app.widgets import Task, button, card, dot, path_text, tinted
from i18n import _, ngettext
from translate.languages import LANGUAGES, native_name

log = logging.getLogger("app")

RESCAN_MS = 2000  # the database folder is looked at again this often (files dropped in)
UPDATE_CHECK_DELAY_MS = 600  # the quiet look at the link after the window opens


def db_summary(info: dict, lang: str) -> str:
    """One line about a game's databases in the player's language."""
    if not info.get("files"):
        return _("No {language} database yet").format(language=native_name(lang))
    text = ngettext("{n} line in {language}", "{n} lines in {language}",
                    int(info.get("translated") or 0)).format(
        n=f"{int(info.get('translated') or 0):,}", language=native_name(lang))
    if info.get("version"):
        text += " · " + _("version {version}").format(version=info["version"])
    return text


def game_line(info: dict, lang: str) -> str:
    """The short line under a game's name in the list."""
    if not info.get("files"):
        return _("No database yet") + f" · {native_name(lang)}"
    n = int(info.get("translated") or 0)
    return ngettext("{n} line", "{n} lines", n).format(n=f"{n:,}") + f" · {native_name(lang)}"


class GameItem(QWidget):
    """A game in the list: its name and a line about its database."""

    def __init__(self, name: str, line: str, preset: bool) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(1)
        star = f"  <span style='color:{COLOURS['preset']}'>★</span>" if preset else ""
        top = QLabel(html.escape(name) + star)
        top.setObjectName("gamename")
        self.line = QLabel(line)
        self.line.setObjectName("note")
        lay.addWidget(top)
        lay.addWidget(self.line)
        for w in (self, top, self.line):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class StartWindow(QWidget):
    def __init__(self, win) -> None:  # noqa: ANN001 - app.panel.MainWindow
        super().__init__(None, Qt.WindowType.Window)
        self.win = win
        self.task: Task | None = None  # a download or a file moved in
        self.check_task: Task | None = None  # a look at the link for a newer database
        self._checked: set[tuple[str, str]] = set()  # (game, language) looked at this start
        self._seen: tuple = ()  # what the database folder held when last shown
        self._link_for = ""  # the game whose link the field shows
        self.setObjectName("start")
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(win.icon))
        self.resize(940, 620)
        self.setMinimumSize(800, 540)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._sidebar())
        root.addWidget(self._content(), 1)
        self._timer = QTimer(self)
        self._timer.setInterval(RESCAN_MS)
        self._timer.timeout.connect(self._rescan)

    # ---- layout --------------------------------------------------------------------------
    def _sidebar(self) -> QFrame:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(250)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(16, 18, 16, 16)
        lay.setSpacing(10)
        head = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(self.win.icon.pixmap(QSize(30, 30)))
        head.addWidget(logo)
        name = QLabel(APP_NAME)
        name.setObjectName("appname")
        name.setWordWrap(True)
        head.addWidget(name, 1)
        lay.addLayout(head)
        lay.addSpacing(8)
        games = QLabel(_("GAMES"))
        games.setObjectName("section")
        lay.addWidget(games)
        self.list = QListWidget()
        self.list.setObjectName("games")
        self.list.setSpacing(2)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.currentItemChanged.connect(lambda *_a: self._picked())
        lay.addWidget(self.list, 1)
        self.add_btn = button("+  " + _("Add game"))
        self.add_btn.setObjectName("flat")
        self.add_btn.clicked.connect(self.add_game)
        self.add_btn.setVisible(ONE_GAME is None)
        lay.addWidget(self.add_btn)
        self.ui_lang = QComboBox()
        self.ui_lang.setToolTip(_("The app's own language"))
        self.ui_lang.addItem("🌐  " + _("System language"), "auto")
        for code in i18n.available():
            self.ui_lang.addItem("🌐  " + native_name(code), code)
        self.ui_lang.setCurrentIndex(max(0, self.ui_lang.findData(self.win.settings.ui_language)))
        self.ui_lang.activated.connect(lambda _i: self._ui_language())
        lay.addWidget(self.ui_lang)
        return side

    def _content(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(14)
        lay.addLayout(self._header())
        lay.addLayout(self._language_row())
        lay.addWidget(self._get_card())
        lay.addWidget(self._folder_card())
        lay.addStretch(1)
        bottom = QHBoxLayout()
        self.msg = QLabel()
        self.msg.setObjectName("status")
        self.msg.setWordWrap(True)
        bottom.addWidget(self.msg, 1)
        self.play_btn = QPushButton("▶   " + _("Play"))
        self.play_btn.setObjectName("play")
        self.play_btn.setToolTip(_("Open the HUD: the overlay starts and waits for the game"))
        self.play_btn.clicked.connect(self.play)
        bottom.addWidget(self.play_btn, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(bottom)
        return page

    def _header(self) -> QHBoxLayout:
        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("gametitle")
        top.addWidget(self.title)
        self.chip = QLabel(_("PRESET"))
        self.chip.setObjectName("chip")
        top.addWidget(self.chip, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addStretch(1)
        self.gear = QToolButton()
        self.gear.setObjectName("iconbtn")
        self.gear.setText("⚙")
        self.gear.setToolTip(_("Game settings"))
        self.gear.clicked.connect(self.edit_game)
        top.addWidget(self.gear)
        helpb = QToolButton()
        helpb.setObjectName("iconbtn")
        helpb.setText("?")
        helpb.setToolTip(_("Help (opens the guide)"))
        helpb.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(HELP_URL)))
        top.addWidget(helpb)
        return top

    def _language_row(self) -> QHBoxLayout:
        """The language the player reads, and what the game has in it."""
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel(_("Translate into")))
        self.target = QComboBox()
        self.target.setMinimumWidth(230)
        self.target.setToolTip(_("The language you read: the databases in it are used, and "
                                 "live translation translates into it"))
        self.target.activated.connect(lambda _i: self._target_changed())
        row.addWidget(self.target)
        self.summary = QLabel()
        self.summary.setObjectName("summary")
        self.summary.setWordWrap(True)
        row.addWidget(self.summary, 1)
        return row

    def _get_card(self) -> QFrame:
        box, lay = card(_("Get the translation"))
        lrow = QHBoxLayout()
        self.link = QLineEdit()
        self.link.setPlaceholderText(
            _("Paste a download link: a database, or a shared Google Drive folder of them"))
        self.link.returnPressed.connect(self.download)
        self.dl_btn = button(_("Download"), primary=True)
        self.dl_btn.clicked.connect(self.download)
        lrow.addWidget(self.link, 1)
        lrow.addWidget(self.dl_btn)
        lay.addLayout(lrow)
        brow = QHBoxLayout()
        self.file_btn = button(_("Use a file…"), _("Move a database file you have into the "
                                                   "game's folder"))
        self.file_btn.clicked.connect(self.use_file)
        self.check_btn = button(_("Check for updates"),
                                _("Ask the link whether there is a newer database (nothing "
                                  "is downloaded)"))
        self.check_btn.clicked.connect(lambda: self.check_updates(quiet=False))
        for b in (self.file_btn, self.check_btn):
            brow.addWidget(b)
        brow.addStretch(1)
        lay.addLayout(brow)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.notice = QFrame()
        self.notice.setObjectName("notice")
        nl = QHBoxLayout(self.notice)
        nl.setContentsMargins(12, 8, 8, 8)
        self.notice_text = QLabel()
        nl.addWidget(self.notice_text, 1)
        get_new = button(_("Download it"), primary=True)
        get_new.clicked.connect(self.download)
        nl.addWidget(get_new)
        self.notice.hide()
        lay.addWidget(self.notice)
        return box

    def _folder_card(self) -> QFrame:
        box, lay = card(_("Database folder"))
        prow = QHBoxLayout()
        self.path = QLabel()
        self.path.setObjectName("path")
        self.path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path.setWordWrap(True)
        prow.addWidget(self.path, 1)
        opn = button(_("Open"), _("Open the folder: drop database files in, or take them out"))
        opn.clicked.connect(self.open_folder)
        prow.addWidget(opn, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(prow)
        self.files = QLabel()
        self.files.setObjectName("files")
        self.files.setWordWrap(True)
        self.files.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.files)
        return box

    # ---- showing --------------------------------------------------------------------------
    def open(self) -> None:
        """Shown (again): the games as they are, the chosen one's folder looked at."""
        self.fill()
        if not self.isVisible():
            scr = QGuiApplication.primaryScreen().availableGeometry()
            self.move(scr.center() - self.rect().center())
        self.show()
        self.raise_()
        self.activateWindow()
        self._timer.start()
        QTimer.singleShot(UPDATE_CHECK_DELAY_MS, lambda: self.check_updates(quiet=True))

    def hideEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        self._timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        event.ignore()
        if self.win.isVisible() or self.win.handle.isVisible():
            self.hide()  # (the HUD is up: this window only goes)
        else:
            self.win.quit()

    def status(self, text: str) -> None:
        self.msg.setText(text)

    # ---- the list -------------------------------------------------------------------------
    def fill(self) -> None:
        s = self.win.settings
        self.list.blockSignals(True)
        self.list.clear()
        for g in s.games:
            lib = self.win.library
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, g.id)
            w = GameItem(g.name, game_line(lib.info(g), lib.target_lang(g)),
                         g.profile != GENERIC)
            item.setSizeHint(w.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, w)
        self.list.blockSignals(False)
        cur = s.current_game
        rows = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        if rows:
            self.list.setCurrentRow(rows.index(cur.id) if cur is not None and cur.id in rows
                                    else 0)
        self.show_game()

    def game(self) -> GameEntry | None:
        item = self.list.currentItem()
        return self.win.settings.game(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _picked(self) -> None:
        g = self.game()
        if g is not None and g.id != self.win.settings.current:
            self.win.settings.current = g.id
            self.win.settings.save()
        self.notice.hide()
        self.show_game()
        self.win.game_changed()
        QTimer.singleShot(300, lambda: self.check_updates(quiet=True))

    def show_game(self) -> None:
        g = self.game()
        for w in (self.link, self.dl_btn, self.file_btn, self.check_btn, self.gear,
                  self.play_btn, self.target):
            w.setEnabled(g is not None)
        if g is None:
            self.title.setText(_("Add a game"))
            self.chip.hide()
            for lab in (self.summary, self.path, self.files):
                lab.setText("")
            return
        self.title.setText(html.escape(g.name))
        self.chip.setVisible(g.profile != GENERIC)
        if g.id != self._link_for:  # (a link being typed stays until another game is shown)
            self.link.setText(g.db_link)
            self._link_for = g.id
        self._fill_targets(g)
        self._show_files(g)

    def _fill_targets(self, g: GameEntry) -> None:
        """The languages to read: those with a database first (✓), then every other one
        (translated live)."""
        lib = self.win.library
        have = lib.offered_languages(g)
        current = lib.target_lang(g)
        self.target.blockSignals(True)
        self.target.clear()
        for code in have:
            self.target.addItem(f"{native_name(code)}  ✔", code)
        if have:
            self.target.insertSeparator(self.target.count())
        for lang in LANGUAGES:
            if lang.code not in have:
                self.target.addItem(lang.native, lang.code)
        if self.target.findData(current) < 0:
            self.target.addItem(native_name(current), current)
        self.target.setCurrentIndex(self.target.findData(current))
        self.target.blockSignals(False)

    def _show_files(self, g: GameEntry) -> None:
        lib = self.win.library
        folder = lib.databases(g)
        files = folder.scan()
        lang = lib.target_lang(g)
        info = lib.info(g)
        self._seen = tuple((f.path.name, f.ok, f.lines, f.translated) for f in files)
        if info["files"]:
            self.summary.setText(f"{dot('ok')} {html.escape(db_summary(info, lang))}")
        else:
            hint = _("lines are translated live, if live translation is on")
            self.summary.setText(f"{dot('wait')} {html.escape(db_summary(info, lang))} · "
                                 f"{html.escape(hint)}")
        self.path.setText(path_text(lib.db_dir(g)))
        rows = [self._file_row(f, folder.why_unused(f, lang)) for f in files]
        self.files.setText("<br>".join(rows) if rows else tinted(_("Empty"), "off"))
        item = self.list.currentItem()
        w = self.list.itemWidget(item) if item is not None else None
        if isinstance(w, GameItem):
            w.line.setText(game_line(info, lang))

    @staticmethod
    def _file_row(f, why: str) -> str:  # noqa: ANN001 - app.dbfolder.DbFile
        name = html.escape(f.path.name)
        if why:
            return (tinted("⚠", "problem") + f"&nbsp; {name} "
                    + tinted("— " + _("not used: {why}").format(why=why.rstrip(".")), "problem"))
        if f.live:
            what = ngettext("{n} live translation", "{n} live translations",
                            f.translated).format(n=f"{f.translated:,}")
        else:
            what = _("{language} · {done} of {total} lines").format(
                language=native_name(f.lang), done=f"{f.translated:,}", total=f"{f.lines:,}")
            if f.version:
                what += " · " + _("version {version}").format(version=f.version)
        return tinted("✔", "ok") + f"&nbsp; {name} " + tinted("— " + what, "off")

    def _rescan(self) -> None:
        """Files dropped into the folder (or taken out) by hand count at once."""
        g = self.game()
        if g is None or self.task is not None:
            return
        files = self.win.library.databases(g).scan()
        if tuple((f.path.name, f.ok, f.lines, f.translated) for f in files) != self._seen:
            self._show_files(g)

    # ---- choices --------------------------------------------------------------------------
    def _target_changed(self) -> None:
        g = self.game()
        code = self.target.currentData()
        if g is None or not code or code == self.win.library.target_lang(g):
            return
        g.target_lang = code
        self.win.settings.save()
        self.notice.hide()
        self.show_game()
        self.win.game_changed()
        self.check_updates(quiet=True)

    def _ui_language(self) -> None:
        code = self.ui_lang.currentData() or "auto"
        if code == self.win.settings.ui_language:
            return
        self.win.settings.ui_language = code
        self.win.settings.save()
        self.status(_("The app's language changes the next time it starts."))

    # ---- actions --------------------------------------------------------------------------
    def add_game(self) -> None:
        d = AddGameDialog(self.win.library, self)
        if d.exec():
            g = d.add(self.win.library)
            self.fill()
            self.status(_("{game} added.").format(game=g.name))
        d.deleteLater()

    def edit_game(self) -> None:
        g = self.game()
        if g is None or self.task is not None:
            return
        d = GameDialog(self.win.library, g, self, self.win.settings.ocr_device)
        if d.exec():
            self.fill()
            self.status(_("{game} removed.").format(game=g.name) if d.removed else _("Saved."))
        d.deleteLater()

    def download(self) -> None:
        g = self.game()
        if g is None or self.task is not None or self.win.refuse_if_busy():
            return
        g.db_link = self.link.text().strip()
        self.win.settings.save()
        if not g.db_link:
            self.status(_("Paste the download link first."))
            self.link.setFocus()
            return
        self.notice.hide()
        self._busy(True)
        self.bar.setRange(0, 0)
        self.bar.setFormat(_("Connecting…"))
        lib = self.win.library

        def work(task: Task):  # noqa: ANN202
            return lib.sync(g, progress=lambda done, total: task.report(done, total))

        def progress(done, total) -> None:  # noqa: ANN001
            if total:
                self.bar.setRange(0, 1000)
                self.bar.setValue(int(1000 * done / total))
                self.bar.setFormat(_("{done:.1f} of {total:.1f} MB").format(
                    done=done / 1e6, total=total / 1e6))
            else:
                self.bar.setFormat(_("{done:.1f} MB").format(done=done / 1e6))

        def finished(res) -> None:  # noqa: ANN001
            replaced, state, dest = res
            lib.synced(g, state, dest)
            self._busy(False)
            self.show_game()
            n = lib.info(g)["translated"]
            self.status(_("Database downloaded and checked: {n} lines.").format(n=f"{n:,}")
                        if replaced else _("You have the newest database already."))
            self.win.game_changed()

        self._run(work, finished, progress)

    def use_file(self) -> None:
        g = self.game()
        if g is None or self.task is not None or self.win.refuse_if_busy():
            return
        path, _filter = QFileDialog.getOpenFileName(
            self, _("Choose a database file"), str(Path.home() / "Downloads"),
            _("Translation database (*.sqlite *.sqlite3 *.db);;All files (*)"))
        if not path:
            return
        lib = self.win.library
        src = Path(path)
        dest = lib.file_target(g, src)
        if src.resolve() == dest.resolve():
            self.status(_("That file is in the folder already."))
            return
        ask = _("Move {file} into the game's database folder?").format(file=src.name)
        ask += f"\n\n{dest.parent}"
        if dest.exists():
            ask += "\n\n" + _("The {file} there now goes to the backups.").format(file=dest.name)
        if QMessageBox.question(self, _("Use a file"), ask) != QMessageBox.StandardButton.Yes:
            return
        self._busy(True)
        self.bar.setRange(0, 0)
        self.bar.setFormat(_("Checking the file…"))

        def finished(dest_: Path) -> None:
            if lib.take_source_language(g, dest_):
                self.win.settings.save()
            self._busy(False)
            self.show_game()
            self.status(_("{file} is in the folder.").format(file=dest_.name))
            self.win.game_changed()

        def failed(why: str) -> None:
            self._busy(False)
            QMessageBox.warning(self, _("Use a file"), _(
                "{file} is not used: {why}\n\nIt was left where it is.").format(
                file=src.name, why=why))

        self._run(lambda _t: lib.use_database_file(g, src), finished, None, failed)

    def check_updates(self, quiet: bool = False) -> None:
        """Ask the game's link whether it has a database the player has not got (only the
        server's answer or the index is read). *quiet*: the look the window takes by
        itself (once per game, language and start, when the player wants it), which says
        nothing unless there is news."""
        g = self.game()
        if g is None or self.check_task is not None or self.task is not None:
            return
        lang = self.win.library.target_lang(g)
        if quiet and ((g.id, lang) in self._checked or not self.win.settings.auto_sync
                      or not g.db_link.strip() or not self.isVisible()):
            return
        if not quiet:
            g.db_link = self.link.text().strip() or g.db_link
            self.win.settings.save()
            if not g.db_link:
                self.status(_("Paste the download link first."))
                return
            self.status(_("Asking the link…"))
        self._checked.add((g.id, lang))

        def done(answer) -> None:  # noqa: ANN001 - (result, languages the link offers)
            self.check_task = None
            res, offered = answer
            if offered is not None and self.win.library.remember_index(g, offered):
                if self.game() is g:
                    self._fill_targets(g)
            if self.game() is not g:
                return
            news = {"newer": _("A newer {language} database is available."),
                    "available": _("A {language} database is available.")}
            if res in news:
                self.notice_text.setText(news[res].format(language=native_name(lang)))
                self.notice.show()
                self.status("")
            elif not quiet:
                self.status({
                    "same": _("Up to date."),
                    "none": _("The link has no {language} database (yet).").format(
                        language=native_name(lang)),
                }.get(res, _("The link's host doesn't say whether the file changed: "
                             "Download fetches it if it did.")))

        def failed(why: str) -> None:
            self.check_task = None
            if not quiet:
                self.status(_("Could not ask the link: {why}").format(why=why))

        self.check_task = Task(lambda _t: self.win.library.check_update(g))
        self.check_task.done.connect(done)
        self.check_task.failed.connect(failed)
        self.check_task.start()

    def open_folder(self) -> None:
        g = self.game()
        if g is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.win.library.db_dir(g))))

    def play(self) -> None:
        g = self.game()
        if g is None:
            return
        lib = self.win.library
        if not lib.has_window(g):
            choice = pick_window(self)
            if choice is None:
                return
            g.window_title, g.process_name = choice.title, choice.own_program
            self.win.settings.save()
        if not lib.info(g)["files"] and not self.win.settings.live.enabled and (
            QMessageBox.question(self, _("Play"), _(
                "No {language} database yet: only live translation would translate (switch "
                "it on in the HUD). Open the HUD anyway?").format(
                language=native_name(lib.target_lang(g))))
            != QMessageBox.StandardButton.Yes
        ):  # fmt: skip
            return
        self.win.play(g)

    # ---- tasks ----------------------------------------------------------------------------
    def _busy(self, on: bool) -> None:
        self.bar.setVisible(on)
        for w in (self.dl_btn, self.file_btn, self.check_btn, self.play_btn, self.list,
                  self.target, self.gear):
            w.setEnabled(not on)

    def _run(self, work, finished, progress=None, failed=None) -> None:  # noqa: ANN001
        self.task = Task(work)
        if progress is not None:
            self.task.progress.connect(progress)

        def end(result) -> None:  # noqa: ANN001
            self.task = None
            finished(result)

        def fail(why: str) -> None:
            self.task = None
            if failed is not None:
                failed(why)
            else:
                self._busy(False)
                QMessageBox.warning(self, APP_NAME, why)

        self.task.done.connect(end)
        self.task.failed.connect(fail)
        self.task.start()
