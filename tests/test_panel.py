"""The app's panel: where it lies, folding away to its handle and back, what a change does
to the overlay, and the start window's language choice. The windows are laid out and
painted but never put on the screen (WA_DontShowOnScreen), and the overlay is never
started."""

from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from app.panel import MainWindow
    from app.theme import apply_theme

    apply_theme(app)
    win = MainWindow(Settings(), desktop=False)  # no tray icon, no hotkey
    for w in (win, win.handle, win.start):
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    yield win
    win.close_down()
    win.deleteLater()


def wait(ms: int) -> None:
    from PySide6.QtTest import QTest

    QTest.qWait(ms)


def test_the_panel_lies_on_the_side_and_folds_to_a_handle(panel):
    from app.panel.window import PANEL_MARGIN

    panel.deploy()
    a = panel._screen().availableGeometry()
    g = panel.geometry()
    assert panel.isVisible() and not panel.handle.isVisible()
    assert g.right() == a.right() - PANEL_MARGIN and g.top() == a.top() + PANEL_MARGIN
    assert g.height() == a.height() - 2 * PANEL_MARGIN
    panel.toggle_panel()  # » or the hotkey
    wait(400)
    assert not panel.isVisible() and panel.handle.isVisible() and panel.settings.panel_folded
    h = panel.handle.geometry()
    assert h.right() == a.right() and a.top() < h.top() < a.bottom()  # on the edge
    panel.toggle_panel()
    wait(400)
    assert panel.isVisible() and not panel.handle.isVisible()
    assert panel.geometry() == g and panel.windowOpacity() == 1.0
    panel.settings.panel_side = "left"
    panel.dock()
    assert panel.geometry().left() == a.left() + PANEL_MARGIN


def test_the_panels_contents_fit_its_width(panel):
    from PySide6.QtWidgets import QScrollArea

    from app.panel.window import PANEL_W

    panel.deploy()
    panel.controls.more.parentWidget()  # (More folded in or out: the widest row counts)
    area = panel.controls.findChild(QScrollArea)
    inner = area.widget()
    for fold in (False, True):
        for f in panel.controls.findChildren(type(panel.controls.diag_fold)):
            f.open(fold)
        wait(50)
        assert inner.minimumSizeHint().width() <= PANEL_W - 40


def test_without_the_handle_folding_hides_everything(panel):
    panel.settings.panel_handle = False
    panel.deploy()
    panel.fold()
    wait(400)
    assert not panel.isVisible() and not panel.handle.isVisible()
    panel.toggle_panel()  # the hotkey (or the tray icon) brings it back
    wait(400)
    assert panel.isVisible()


def test_a_folded_panel_comes_back_folded(panel):
    panel.settings.panel_folded = True
    panel.deploy()
    assert not panel.isVisible() and panel.handle.isVisible()


def test_a_game_without_a_window_is_not_started_but_explained(panel):
    g = panel.library.add_game("Some other game")  # no window chosen: it becomes current
    panel.refresh_all()
    panel.restart_overlay(start=True)
    wait(500)
    assert not panel.session.busy and not panel.session.running
    assert "Choose the game window" in panel.msg.text()
    assert panel.settings.current_game is g


def test_a_problem_is_said_once_and_opens_the_diagnostics(panel):
    from types import SimpleNamespace

    from hud.app import Problem

    class Hud:
        problem = Problem.EXCLUSIVE_FULLSCREEN
        window = SimpleNamespace(title="Azur Promilia", client=(0, 0, 1920, 1080))
        overlay = SimpleNamespace(enabled=True)

        def stop(self):
            pass

        def diagnostics(self):
            return {"window": "Azur Promilia", "ocr": {"device": "dml"}}

    panel.session.hud = Hud()
    panel._update_state()
    assert "exclusive full screen" in panel.msg.text()
    panel.deploy()
    panel._tick()
    fold = panel.controls.diag_fold
    assert not fold.isHidden() and fold.toggle.isChecked()
    assert "exclusive full screen" in panel.controls.diagnostics.warning.text()
    panel.session.hud = None


def test_diagnostics_explains_cpu_fallback_and_timings(panel):
    data = {
        "window": "Azur Promilia",
        "capture": "dxcam",
        "ocr": {"device": "cpu", "requested": "auto",
                "fallback_reason": "DirectML provider is unavailable."},
        "timings": {"dialogue": {"median": 430.0, "p95": 510.0},
                    "ui": {"median": 900.0, "p95": 1200.0},
                    "tick": {"median": 950.0, "p95": 1300.0}},
        "cover_cache": {"hits": 8, "misses": 3},
        "recall": {"hits": 5},
        "lookup": {"misses": 4},
    }
    c = panel.controls
    c.update_diagnostics(data)
    d = c.diagnostics
    assert not c.diag_fold.isHidden() and c.diag_fold.toggle.isChecked()
    assert "DirectML provider is unavailable." in d.warning.text()
    assert not d.gpu_btn.isHidden()
    assert "430 / 510 ms" in d.timing.text() and "8 hits / 3 misses" in d.cache.text()
    data["ocr"] = {"device": "dml"}
    c.update_diagnostics(data)
    assert d.gpu_btn.isHidden() and not d.warning.text()


def test_try_the_gpu_again_lets_the_reader_choose(panel):
    panel.settings.ocr_device = "cpu"
    panel.retry_gpu()
    assert panel.settings.ocr_device == "auto"


def test_a_change_to_another_game_leaves_the_overlay_alone(panel):
    other = panel.library.add_game("Another game", window_title="Another")
    panel.settings.current = "azur-promilia-jp"
    panel.restart_overlay(game=other)  # not the current game, nothing runs: nothing to do
    assert not panel._restart.isActive()


def test_closing_always_takes_the_panel_away(panel, monkeypatch):
    from PySide6.QtWidgets import QApplication

    panel.deploy()

    def stuck(quiet: bool = False) -> None:
        raise RuntimeError("the reader would not stop")

    monkeypatch.setattr(panel.session, "stop", stuck)
    quits = []
    monkeypatch.setattr(QApplication.instance(), "quit", lambda: quits.append(1))
    panel.quit()  # ✕
    assert quits and panel.quitting
    assert not panel.isVisible() and not panel.handle.isVisible()


def test_a_refresh_keeps_what_the_player_typed_in_the_start_window(panel):
    start = panel.start
    start.fill()
    start.link.setText("https://example.com/db.sqlite")  # typed, not saved yet
    start.show_game()  # e.g. a file dropped into the database folder meanwhile
    assert start.link.text() == "https://example.com/db.sqlite"
    panel.library.add_game("Another game")  # the list changed: filled anew
    start.fill()
    assert start.list.count() == 2


def test_the_language_read_picks_the_databases(panel):
    from db.store import Store

    start, lib = panel.start, panel.library
    g = lib.add_game("Some Game")
    with Store(lib.db_dir(g) / "g.es-419.sqlite") as store:
        store.set_info(game="g", source_lang="ja", target_lang="es-419")
    start.fill()
    langs = [start.target.itemData(i) for i in range(start.target.count())]
    assert langs[0] == "es-419" and "✔" in start.target.itemText(0)  # a database: first
    assert "en" in langs and "de" in langs  # every other language: live translation
    assert "No English database yet" in start.summary.text()
    start.target.setCurrentIndex(0)
    start._target_changed()
    assert g.target_lang == "es-419" and lib.db_paths(g)[0].name == "g.es-419.sqlite"
    assert "Español" in start.summary.text()


def test_play_opens_the_hud_and_back_returns_to_the_games(panel, monkeypatch):
    started = []
    monkeypatch.setattr(panel.session, "start", lambda game: started.append(game))
    panel.launch()
    assert panel.start.isVisible() and not panel.isVisible()
    g = panel.settings.current_game
    g.window_title = "Azur Promilia"
    panel.play(g)
    wait(1000)  # (the overlay starts a moment after the panel is up)
    assert panel.isVisible() and not panel.start.isVisible() and started == [g]
    assert panel.title.text() == g.name
    panel.back()  # ◀
    assert not panel.isVisible() and not panel.handle.isVisible() and panel.start.isVisible()


def test_one_job_at_a_time_writes_a_database(panel):
    assert panel.db_busy() is None
    panel.start.task = object()  # a download under way
    assert panel.refuse_if_busy() and "downloaded" in panel.msg.text()
    panel.start.task = None
    assert not panel.refuse_if_busy()


def test_the_overlays_reader_stops_and_is_waited_for():
    import threading
    import time
    from types import SimpleNamespace

    import numpy as np

    from hud.app import OcrWorker

    reading = threading.Event()

    class Core:
        def tick(self, _frame, _newest, _now):
            reading.set()
            time.sleep(0.1)  # a read under way
            return SimpleNamespace(ui=None, dialogue=None)

    w = OcrWorker(Core(), verbose=False)
    w.start()
    w.offer(np.zeros((4, 4, 3), np.uint8))
    assert reading.wait(2.0)
    w.stop()
    w.join(timeout=2.0)
    assert not w.is_alive()
