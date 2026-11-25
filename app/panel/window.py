"""The HUD's panel window, and the app around it.

The panel lies on one side of the screen, above the game and the overlay, and folds away to
a handle on the screen's edge. The window also runs the app: the start window it opens
with, the overlay session, the tray icon, and closing everything down.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QGuiApplication, QIcon, QScreen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import APP_NAME, HELP_URL
from app.config import GameEntry, Settings
from app.library import Library
from app.panel.controls import HudControls
from app.panel.edge import EdgeHandle
from app.paths import assets_dir
from app.session import HudSession
from app.start_window import StartWindow
from app.theme import COLOURS
from app.widgets import link_button
from hud.app import problem_text
from hud.hotkey import GlobalHotkeys, pretty
from hud.platform import current
from hud.screens import screen_at
from i18n import _

log = logging.getLogger("app")

PANEL_W = 410  # the panel's width (screen units)
PANEL_MARGIN = 8  # its distance from the screen's edges
HANDLE_W, HANDLE_H = 22, 92  # the handle left on the edge while it is folded away
SLIDE_PX = 36  # folding and unfolding: it slides this far while it fades
SLIDE_MS = 170
RESTART_MS = 300  # changes made in a row restart the overlay once
CLOCK_MS = 1000  # the panel looks at the overlay's state this often
FOLDER_EVERY_TICKS = 3  # ...and at the database folder every third time


class MainWindow(QWidget):
    """The app, as part of the HUD: a panel on one side of the screen, above the game and
    the overlay, and interactive (the overlay itself lets every click through to the game).
    The panel folds away to a handle on the edge (its » button, the handle, or a hotkey);
    ✕ closes the app, overlay and all. The overlay never reads the panel."""

    def __init__(self, settings: Settings, desktop: bool = True) -> None:
        """*desktop*: the tray icon and the panel's hotkey too (tests go without both)."""
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.settings = settings
        self.library = Library(settings)
        self.library.ensure_defaults()
        self.session = HudSession(self.library, settings)
        self.session.on_raise = self.keep_on_top
        self.quitting = False
        self.setWindowTitle(APP_NAME)
        self.icon = QIcon(str(assets_dir() / "icon.png"))
        self.setWindowIcon(self.icon)

        frame = QFrame()
        frame.setObjectName("panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 12, 10)
        lay.setSpacing(8)
        lay.addLayout(self._header())
        self.controls = HudControls(self)
        lay.addWidget(self.controls, 1)
        foot = QHBoxLayout()
        self.msg = QLabel("")
        self.msg.setObjectName("status")
        self.msg.setWordWrap(True)
        foot.addWidget(self.msg, 1)
        helpb = link_button(_("Help"))
        helpb.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(HELP_URL)))
        foot.addWidget(helpb, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(foot)

        self.handle = EdgeHandle(self)
        self._anim: QParallelAnimationGroup | None = None
        self._folding = False
        self._screen_seen: QScreen | None = None  # where the game was last
        self._restart = QTimer(self)
        self._restart.setSingleShot(True)
        self._restart.setInterval(RESTART_MS)
        self._restart.timeout.connect(self._do_restart)
        self._ticks = 0
        self._problem: str | None = None  # what keeps the overlay from working, as said
        self._clock = QTimer(self)
        self._clock.setInterval(CLOCK_MS)
        self._clock.timeout.connect(self._tick)
        self._clock.start()

        self.start = StartWindow(self)
        self.session.state.connect(self._session_state)
        self.session.message.connect(self.status)
        self._tray(show=desktop)
        self.hotkeys = GlobalHotkeys()
        self.hotkeys.install(QApplication.instance())
        if desktop and not self.hotkeys.register(settings.panel_hotkey, self.toggle_panel):
            log.warning("could not register hotkey %s", settings.panel_hotkey)
        self.refresh_all()
        self._update_state()
        self.dock()

    def _header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        back = QToolButton()
        back.setObjectName("headbtn")
        back.setText("◀")
        back.setToolTip(_("Games: stop the HUD and choose a game or its database"))
        back.clicked.connect(self.back)
        row.addWidget(back, 0, Qt.AlignmentFlag.AlignTop)
        logo = QLabel()
        logo.setPixmap(self.icon.pixmap(QSize(26, 26)))
        row.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.title = QLabel(APP_NAME)
        self.title.setObjectName("paneltitle")
        col.addWidget(self.title)
        self.state_label = QLabel()
        self.state_label.setObjectName("note")
        col.addWidget(self.state_label)
        row.addLayout(col, 1)
        self.fold_btn = QToolButton()
        self.fold_btn.setObjectName("headbtn")
        self.fold_btn.clicked.connect(self.fold)
        row.addWidget(self.fold_btn, 0, Qt.AlignmentFlag.AlignTop)
        close = QToolButton()
        close.setObjectName("closebtn")
        close.setText("✕")
        close.setToolTip(_("Close the app (the overlay with it)"))
        close.clicked.connect(self.quit)
        row.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        return row

    # ---- the flow: the start window, then the HUD ----------------------------------------
    def launch(self) -> None:
        """The app starts: the games (the start window); Play there opens the HUD."""
        self.start.open()

    def play(self, game: GameEntry) -> None:
        """The HUD for *game*: the panel at the side of the screen, the overlay started
        (it waits for the game window)."""
        self.settings.current = game.id
        self.settings.save()
        self.refresh_all()
        self.start.hide()
        self.deploy()

    def back(self) -> None:
        """◀: the HUD stops and the start window comes back."""
        self._restart.stop()
        self.session.stop(quiet=True)
        self._stop_anim()
        self._folding = False
        self.handle.hide()
        self.hide()
        self.start.open()

    def deploy(self) -> None:
        """Show the panel as it was left (unfolded, or folded to its handle) and start the
        overlay, which waits for the game window."""
        if self.settings.panel_folded:
            self._folded()
        else:
            self.unfold(animate=False)
        QTimer.singleShot(RESTART_MS, lambda: self.restart_overlay(start=True))

    # ---- where the panel lies --------------------------------------------------------------
    def _left(self) -> bool:
        return self.settings.panel_side == "left"

    def _screen(self) -> QScreen:
        """The game's screen (or the one it was last on), else the main one."""
        hud = self.session.hud
        found = hud.window if hud is not None else None
        if found is not None:
            scr = screen_at(found.client)  # (screen pixels to Qt's units, per monitor)
            if scr is not None:
                self._screen_seen = scr
        if self._screen_seen in QGuiApplication.screens():
            return self._screen_seen
        return QGuiApplication.primaryScreen()

    def _panel_rect(self) -> QRect:
        a = self._screen().availableGeometry()
        w = min(PANEL_W, a.width() - 2 * PANEL_MARGIN)
        x = a.left() + PANEL_MARGIN if self._left() else a.right() + 1 - PANEL_MARGIN - w
        return QRect(x, a.top() + PANEL_MARGIN, w, a.height() - 2 * PANEL_MARGIN)

    def _handle_rect(self) -> QRect:
        a = self._screen().availableGeometry()
        x = a.left() if self._left() else a.right() + 1 - HANDLE_W
        return QRect(x, a.top() + (a.height() - HANDLE_H) // 2, HANDLE_W, HANDLE_H)

    def set_panel_hotkey(self, spec: str) -> bool:
        """The panel's hotkey changed: registered anew (False, and the old one kept, when
        another program has it)."""
        self.hotkeys.unregister_all()
        if self.hotkeys.register(spec, self.toggle_panel):
            return True
        self.hotkeys.register(self.settings.panel_hotkey, self.toggle_panel)
        return False

    def dock(self) -> None:
        """To the side of the screen the settings say (the game's screen)."""
        keys = pretty(self.settings.panel_hotkey)
        self.fold_btn.setText("«" if self._left() else "»")
        self.fold_btn.setToolTip(_("Fold away ({keys})").format(keys=keys))
        self.handle.setToolTip(_("Unfold the panel ({keys})").format(keys=keys))
        if self._animating():
            return
        if self.isVisible() and self._panel_rect() != self.geometry():
            self.setGeometry(self._panel_rect())
        if self.handle.isVisible():
            self.handle.dock(self._handle_rect(), self._left())

    def _animating(self) -> bool:
        return self._anim is not None and self._anim.state() == QAbstractAnimation.State.Running

    def toggle_panel(self) -> None:
        if self.isVisible() and not self._folding:
            self.fold()
        else:
            self.unfold()

    def unfold(self, animate: bool = True) -> None:
        self._stop_anim()
        self._folding = False
        self.settings.panel_folded = False
        self.handle.hide()
        self.dock()
        end = self._panel_rect()
        if not animate or self.isVisible():
            self.setGeometry(end)
            self.setWindowOpacity(1.0)
            self.show()
            self.keep_on_top()
            return
        start = end.translated((-1 if self._left() else 1) * SLIDE_PX, 0)
        self.setGeometry(start)
        self.setWindowOpacity(0.0)
        self.show()
        self.keep_on_top()
        self._slide(start.topLeft(), end.topLeft(), 0.0, 1.0, None)

    def fold(self) -> None:
        if not self.isVisible() or self._folding:
            return
        self._stop_anim()
        self._folding = True
        self.settings.panel_folded = True
        r = self.geometry()
        away = r.topLeft() + QPoint((-1 if self._left() else 1) * SLIDE_PX, 0)
        self._slide(r.topLeft(), away, self.windowOpacity(), 0.0, self._folded)

    def _folded(self) -> None:
        self._folding = False
        self.hide()
        self.setWindowOpacity(1.0)
        self.dock()
        if self.settings.panel_handle:
            self.handle.dock(self._handle_rect(), self._left())
            self.handle.show()
            self.handle.raise_()

    def _slide(self, p0: QPoint, p1: QPoint, o0: float, o1: float, then) -> None:  # noqa: ANN001
        group = QParallelAnimationGroup(self)
        for prop, a, b in ((b"pos", p0, p1), (b"windowOpacity", o0, o1)):
            anim = QPropertyAnimation(self, prop, group)
            anim.setDuration(SLIDE_MS)
            anim.setStartValue(a)
            anim.setEndValue(b)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(anim)
        if then is not None:
            group.finished.connect(then)
        self._anim = group
        group.start()

    def _stop_anim(self) -> None:
        if self._anim is not None:
            self._anim.stop()
            self._anim = None

    def keep_on_top(self) -> None:
        """Above the overlay, which goes to the top every second (above the game)."""
        if self.isVisible():
            self.raise_()
        if self.handle.isVisible():
            self.handle.raise_()

    def showEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        super().showEvent(event)
        current().exclude_from_capture(self)  # the overlay reads the game under it

    # ---- shared ----------------------------------------------------------------------------
    def status(self, text: str) -> None:
        self.msg.setText(text)
        self.start.status(text)

    def refresh_all(self) -> None:
        game = self.settings.current_game
        self.title.setText(game.name if game is not None else APP_NAME)
        self.controls.refresh()
        if self.start.isVisible():
            self.start.show_game()

    def game_changed(self) -> None:
        """Another game, or its language or databases changed: everything shows it, and a
        running overlay takes it."""
        self.refresh_all()
        self.restart_overlay()

    def keys_changed(self) -> None:
        """An API key was saved somewhere: every place that shows one shows it."""
        self.controls.refresh()

    def db_busy(self) -> str | None:
        """What is writing the game's databases right now (a download), if anything:
        another such job waits until it is done."""
        if self.start.task is not None:
            return _("A database is being downloaded")
        return None

    def refuse_if_busy(self) -> bool:
        """True (and the player is told why) while another job writes the databases."""
        busy = self.db_busy()
        if busy:
            self.status(_("{what}: try again when it is done.").format(what=busy))
        return busy is not None

    def _tick(self) -> None:
        self._ticks += 1
        self._update_state()
        self.dock()
        game = self.settings.current_game
        if (self._ticks % FOLDER_EVERY_TICKS == 0 and self.session.running and game is not None
                and self.library.db_state(game) != self.session.db_state):
            self.status(_("The database folder changed: the overlay takes it anew."))
            self.restart_overlay(game=game)
        if self.isVisible():
            # the overlay knows its window; without it the windows are listed (slower)
            if self.session.hud is not None or self._ticks % 2 == 0:
                self.controls.refresh_window()
            self.controls.set_running(self.session.running, self.session.busy)
            self.controls.live.show_usage(self.session.live_stats())
            self.controls.more.show_misses()
            hud = self.session.hud
            data = hud.diagnostics() if hud is not None and hud.window is not None else None
            self.controls.update_diagnostics(data, self._problem or "")

    def _update_state(self) -> None:
        s = self.session
        problem = None
        if s.busy:
            key, text = "wait", _("Starting the overlay…")
        elif s.running:
            hud = s.hud
            problem = problem_text(hud.problem) if hud is not None and hud.problem else None
            if problem:
                key, text = "problem", _("Overlay on: it cannot work (see Diagnostics)")
            elif hud is not None and hud.window is None:
                key, text = "wait", _("Overlay on: waiting for the game window")
            elif hud is not None and not hud.overlay.enabled:
                key, text = "wait", _("Overlay on: translations hidden")
            else:
                key, text = "ok", _("Overlay on")
            stats = s.live_stats()
            if stats is not None:
                if stats["halted"]:
                    text += " · " + _("live translation stopped")
                    problem = problem or _("Live translation stopped: {why}").format(
                        why=stats["halted"])
                else:
                    text += " · " + _("live: {n} lines").format(n=stats["translated"])
        else:
            key, text = "off", _("Overlay off")
        if problem != self._problem:  # said once, when it comes (and that it went)
            if problem:
                self.status(f"⚠ {problem}")
                if self.tray.isVisible() and not self.isVisible():
                    self.tray.showMessage(APP_NAME, problem,
                                          QSystemTrayIcon.MessageIcon.Warning, 8000)
            elif self._problem:
                self.status(_("The overlay works again."))
            self._problem = problem
        colour = COLOURS[key]
        self.state_label.setText(f"<span style='color:{colour}'>●</span> {text}")
        self.handle.set_dot(colour)
        self.tray.setToolTip(f"{APP_NAME}: {text}")

    # ---- the overlay ---------------------------------------------------------------------
    def toggle_session(self) -> None:
        if self.session.running or self.session.busy:
            self.session.stop()
            return
        game = self.settings.current_game
        if game is None:
            QMessageBox.information(self, APP_NAME, _("Choose a game first (◀ Games)."))
            return
        if not self.library.has_window(game):
            QMessageBox.information(self, APP_NAME, _(
                "Tell the app which window is the game: press 'Choose window…'."))
            return
        self.session.start(game)

    def toggle_translations(self) -> None:
        self.session.toggle_overlay()
        self.controls.set_running(self.session.running, self.session.busy)
        self._update_state()

    def retry_gpu(self) -> None:
        """Start the text reader again, letting it choose the GPU if it can."""
        self.settings.ocr_device = "auto"
        self.settings.save()
        self.status(_("Starting the text reader again…"))
        self.restart_overlay(start=True)

    def restart_overlay(self, start: bool = False, game: GameEntry | None = None) -> None:
        """The game, its window, language or databases, or a dialogue setting changed: the
        overlay restarts with them if it runs (with *start*: also if it does not). Changes
        made in a row restart it once."""
        if self.quitting or (game is not None and game is not self.settings.current_game):
            return
        if start or self.session.running or self.session.busy:
            self._restart.start()

    def _do_restart(self) -> None:
        if self.quitting:
            return
        game = self.settings.current_game
        self.session.stop(quiet=True)
        if game is None:
            return
        if not self.library.has_window(game):
            self.status(_("Choose the game window, and the overlay starts."))
            return
        self.session.start(game)

    def live_changed(self) -> None:
        """Live translation switched on or off, or to another engine or model."""
        if self.session.running:
            self.session.reload_live()
        self._update_state()

    def _session_state(self, state: str) -> None:
        self.controls.set_running(state == "running", state == "starting")
        on = state in ("running", "starting")
        self.tray_toggle.setText(_("Stop the overlay") if on else _("Start the overlay"))
        self._update_state()

    # ---- tray ------------------------------------------------------------------------------
    def _tray(self, show: bool = True) -> None:
        self.tray = QSystemTrayIcon(self.icon, self)
        self.tray.setToolTip(APP_NAME)
        self.tray_menu = QMenu()
        actions = []
        for text, slot in ((_("Games…"), self.back), (_("Show / fold the panel"),
                                                      self.toggle_panel)):
            a = QAction(text, self)
            a.triggered.connect(slot)
            actions.append(a)
        self.tray_toggle = QAction(_("Start the overlay"), self)
        self.tray_toggle.triggered.connect(self.toggle_session)
        hide = QAction(_("Hide / show translations"), self)
        hide.triggered.connect(self.toggle_translations)
        quit_ = QAction(_("Close the app"), self)
        quit_.triggered.connect(self.quit)
        for a in (*actions, self.tray_toggle, hide):
            self.tray_menu.addAction(a)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(quit_)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(
            lambda reason: self.show_panel()
            if reason == QSystemTrayIcon.ActivationReason.Trigger else None
        )
        if show:
            self.tray.show()

    def show_panel(self) -> None:
        """Another start of the app, or the tray: the HUD's panel unfolded when the HUD is
        up, else the start window."""
        if self.isVisible() or self.handle.isVisible() or self.session.running:
            self.unfold()
            self.activateWindow()
        else:
            self.start.open()

    # ---- closing ---------------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self.quitting:
            event.accept()
            return
        event.ignore()  # (Alt+F4) the app closes the way ✕ closes it
        self.quit()

    def quit(self) -> None:
        if self.quitting:
            return
        try:
            self.close_down()
        finally:
            QApplication.instance().quit()  # whatever failed on the way: the app closes

    def close_down(self) -> None:
        """Everything the app runs stopped, its windows gone, the settings saved. A part that
        fails to stop is logged and passed over: the windows always go."""
        self.quitting = True
        self._clock.stop()
        self._restart.stop()
        self._stop_anim()
        self.handle.hide()
        self.hide()
        self.start.hide()
        self.tray.hide()
        try:
            self.session.stop(quiet=True)
        except Exception:  # noqa: BLE001 - closing goes on
            log.exception("stopping the overlay failed")
        try:
            self.settings.save()
        except OSError:
            log.exception("could not save the settings")
        self.hotkeys.unregister_all()
        self.hotkeys.uninstall()
