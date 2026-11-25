"""The HUD's side of the operating system, on any system: following the game's window,
the capture thread, the OCR worker, what the HUD reports, where the app keeps its data,
one app at a time, the OCR devices."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capture.profile import WindowMatch
from hud.platform import PART, WHOLE, WindowRef

ASPECT = 16 / 9


def pillarboxed() -> np.ndarray:
    frame = np.zeros((1080, 2560, 3), np.uint8)
    frame[:, 320:2240] = np.random.default_rng(3).integers(30, 220, (1080, 1920, 3))
    return frame


class FakePlatform:
    """Windows to find, as a test sets them."""

    name = "fake"

    def __init__(self) -> None:
        self.wins: dict[int, WindowRef] = {}

    def find(self, match: WindowMatch) -> WindowRef | None:
        from hud.platform import best_window

        return best_window([WindowRef(**vars(w)) for w in self.wins.values()], match,
                           own_pid=-1)  # fmt: skip

    def client_rect(self, window_id: int):
        w = self.wins.get(window_id)
        return w.client if w is not None else None


# ---- the game's window ------------------------------------------------------------------
def test_the_tracker_finds_follows_and_loses_the_window():
    from hud.app import GameWindowTracker

    plat = FakePlatform()
    match = WindowMatch(title_patterns=["Azur Promilia"])
    tracker = GameWindowTracker(plat, match, lambda rect: None)
    assert tracker.refresh(0.0) == tracker.NONE
    plat.wins[1] = WindowRef(1, "Azur Promilia Wiki", 5, "game.exe", (0, 0, 800, 600))
    assert tracker.refresh(0.5) == tracker.NONE  # looked a moment ago
    assert tracker.refresh(1.5) == tracker.FOUND and tracker.window.score == PART
    plat.wins[1].client = (10, 20, 800, 600)  # it moved
    assert tracker.refresh(2.0) == tracker.SAME and tracker.window.client == (10, 20, 800, 600)
    # matched by a part of its title: a whole match turning up later takes its place
    plat.wins[2] = WindowRef(2, "Azur Promilia", 6, "game.exe", (0, 0, 1920, 1080))
    assert tracker.refresh(3.0) == tracker.SAME  # (looked again only every few seconds)
    assert tracker.refresh(7.0) == tracker.CHANGED and tracker.window.id == 2
    assert tracker.window.score == WHOLE
    del plat.wins[2]
    assert tracker.refresh(7.1) == tracker.LOST and tracker.window is None


def test_the_tracker_takes_a_picture_between_bars_when_two_looks_agree():
    from hud.app import PICTURE_CHECK_S, GameWindowTracker

    frame = pillarboxed()
    tracker = GameWindowTracker(FakePlatform(), WindowMatch(aspect=ASPECT), lambda r: frame)
    client = (100, 50, 2560, 1080)
    assert not tracker.look_for_bars(client, 100.0) and tracker.picture is None  # once: not yet
    assert not tracker.look_for_bars(client, 100.5)  # too soon to look again
    assert tracker.look_for_bars(client, 100.0 + PICTURE_CHECK_S)
    assert tracker.picture == (320, 0, 1920, 1080)
    assert tracker.picture_rect(client) == (420, 50, 1920, 1080)
    assert tracker.picture_rect((0, 0, 1280, 720)) == (0, 0, 1280, 720)  # a smaller window


# ---- the OCR worker ----------------------------------------------------------------------
class CountingCore:
    def __init__(self) -> None:
        self.ticks = 0
        self.work_left = False
        self.read = threading.Event()

    def tick(self, _frame, _newest, _now):
        self.ticks += 1
        self.read.set()
        return SimpleNamespace(ui=None, dialogue=None)

    def pending(self) -> bool:
        return self.work_left


def test_the_worker_reads_each_frame_once_unless_work_is_left():
    from hud.app import SAME_FRAME_EVERY_S, OcrWorker

    core = CountingCore()
    w = OcrWorker(core, verbose=False)
    w.start()
    try:
        frame = np.zeros((4, 4, 3), np.uint8)
        w.offer(frame, 1)
        assert core.read.wait(2.0)
        time.sleep(3 * SAME_FRAME_EVERY_S)
        w.offer(frame, 1)  # the same frame again: nothing new to read
        time.sleep(3 * SAME_FRAME_EVERY_S)
        assert core.ticks == 1
        core.work_left = True  # the core left work for later: the same frame is read again
        time.sleep(4 * SAME_FRAME_EVERY_S)
        assert core.ticks >= 2
        core.work_left, before = False, core.ticks
        w.offer(frame, 2)  # a new frame
        time.sleep(3 * SAME_FRAME_EVERY_S)
        assert core.ticks == before + 1
    finally:
        w.stop()
        w.join(timeout=2.0)
    assert not w.is_alive()


def test_a_live_translation_wakes_the_worker_without_a_new_frame():
    from hud.app import SAME_FRAME_EVERY_S, OcrWorker

    core = CountingCore()
    w = OcrWorker(core, verbose=False)
    w.start()
    try:
        w.offer(np.zeros((4, 4, 3), np.uint8), 1)
        assert core.read.wait(2.0)
        time.sleep(3 * SAME_FRAME_EVERY_S)
        assert core.ticks == 1
        core.read.clear()
        core.wake()  # (set by the worker: a translation came in, nothing else pending)
        assert core.read.wait(2.0) and core.ticks == 2
    finally:
        w.stop()
        w.join(timeout=2.0)


# ---- what the HUD reports ----------------------------------------------------------------
def make_hud(age: float = 0.0, failures: int = 0, exclusive: bool = False):
    from hud.app import HudApp

    hud = HudApp.__new__(HudApp)
    hud.source = SimpleNamespace(age=lambda: age)
    hud.worker = SimpleNamespace(failures=failures)
    hud.platform = SimpleNamespace(exclusive_fullscreen=lambda window: exclusive)
    hud.tracker = SimpleNamespace(window=None)
    hud._black_since, hud._fullscreen_at, hud._exclusive = None, -1e9, False
    return hud


def test_a_black_picture_for_long_is_a_game_hidden_from_capture():
    from hud.app import BLACK_S, Problem

    hud = make_hud()
    black = np.zeros((90, 160, 3), np.uint8)
    assert hud._problem(black, 10.0) is None  # a fade to black: not yet
    assert hud._problem(black, 10.0 + BLACK_S) is Problem.BLACK_CAPTURE
    scene = black.copy()
    scene[40:50, 70:90] = 200
    assert hud._problem(scene, 20.0) is None  # the picture came back
    assert hud._problem(black, 20.5) is None  # ...and a new dark spell starts from zero


def test_problems_say_what_matters_most():
    from hud.app import Problem, problem_text

    assert make_hud(age=5.0)._problem(None, 0.0) is Problem.NO_FRAMES
    assert make_hud(failures=50)._problem(None, 0.0) is Problem.READER_FAILING
    assert make_hud(exclusive=True)._problem(None, 0.0) is Problem.EXCLUSIVE_FULLSCREEN
    assert all(problem_text(p) for p in Problem)
    assert "Windows" not in problem_text(Problem.NOT_EXCLUDED)  # (every system's words)


def test_the_command_line_makes_a_config():
    import argparse

    from hud.app import add_core_args, add_window_args, config_from_args

    ap = argparse.ArgumentParser()
    add_core_args(ap)
    add_window_args(ap)
    args = ap.parse_args(["--game", "g", "--db", "a.sqlite", "--db", "b.sqlite", "--target",
                          "es-419", "--title", "Game", "--no-cache"])  # fmt: skip
    config = config_from_args(args)
    assert config.dbs == [Path("a.sqlite"), Path("b.sqlite")] and config.target_lang == "es-419"
    assert config.title == "Game" and config.cache_dir is None and config.cover_mode == "grow"


# ---- the capture thread --------------------------------------------------------------------
class FakeGrabber:
    backend = "fake"

    def __init__(self, results: list) -> None:
        self.results = results
        self.targets: list = []
        self.resets = 0

    def grab(self, area, target):
        self.targets.append(target)
        return self.results.pop(0) if self.results else (None, True)

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        pass


def test_the_capture_counts_new_frames_and_remakes_a_failing_grabber(monkeypatch):
    import hud.screen_source as ss

    frame = np.ones((10, 10, 3), np.uint8)
    grabber = FakeGrabber([(frame, True), (None, True)] + [(None, False)] * ss.RECREATE_AFTER)
    src = ss.ScreenSource(start=False)
    src.set_area((0, 0, 10, 10), target=42)
    src._grab_once(grabber, (0, 0, 10, 10), 42)
    assert src.seq == 1 and src.grab((0, 0, 10, 10)) is frame
    src._grab_once(grabber, (0, 0, 10, 10), 42)  # nothing changed: the same frame
    assert src.seq == 1 and src.fail_count == 0
    for _ in range(ss.RECREATE_AFTER):
        src._grab_once(grabber, (0, 0, 10, 10), 42)
    assert grabber.resets == 1 and grabber.targets[0] == 42


# ---- the app's data and one app at a time --------------------------------------------------
def test_the_data_folder_follows_the_system(monkeypatch, tmp_path):
    import app.paths as paths

    monkeypatch.delenv("GTHUD_DATA", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert paths.data_dir() == tmp_path / "xdg" / paths.APP_ID
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert paths.user_data_root() == tmp_path / "home" / ".local" / "share"
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert paths.user_data_root() == tmp_path / "local"
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "portable"))
    assert paths.data_dir() == tmp_path / "portable"


def test_a_key_without_a_credential_store_is_a_private_file(monkeypatch):
    import os
    import stat

    import app.keys as keys

    monkeypatch.setattr(keys, "_keyring", lambda: None)
    where = keys.set_key("sk-test-1234567890", "openai")
    assert keys.get_key("openai") == "sk-test-1234567890"
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(where).st_mode) == 0o600
    keys.set_key("", "openai")
    assert keys.get_key("openai") == ""


def test_one_app_at_a_time(tmp_path):
    import app.__main__ as entry

    assert entry._first_instance(tmp_path)
    try:
        assert not entry._first_instance(tmp_path)  # a second start finds it taken
    finally:
        entry._instance_lock.unlock()
        entry._instance_lock = None
    assert entry._channel_name("GameTranslationHUD").startswith("GameTranslationHUD-")


# ---- reading devices -----------------------------------------------------------------------
def test_ocr_devices_fall_back_to_the_cpu():
    from capture.ocr import CPU, DEVICES, OcrEngine, device_available, preferred_devices

    devices = preferred_devices()
    assert devices[-1] == CPU and set(devices) <= set(DEVICES)
    assert device_available(CPU)
    with pytest.raises(ValueError):
        OcrEngine(device="quantum")


def test_the_tuned_reader_reads_what_rapidocr_reads():
    """The detector's input made in float32 is rapidocr's own; the recognizer with shapes
    that repeat (used on a GPU) reads the same lines, character cells and all."""
    import cv2
    import numpy as np
    from rapidocr.ch_ppocr_det.utils import DetPreProcess

    from capture.ocr import OcrEngine, _FixedShapes, _Float32DetInput

    img = np.full((180, 900, 3), 235, dtype=np.uint8)
    for i, text in enumerate(("READY 123", "SETTINGS AND MORE", "OK")):
        cv2.putText(img, text, (20 + 40 * i, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (20, 20, 20), 2)
    det_in = _Float32DetInput(1280, "max", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ref = DetPreProcess(1280, "max", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    assert det_in(img).dtype == np.float32 and np.allclose(det_in(img), ref, atol=1e-5)
    eng = OcrEngine(lang="japan", device="cpu")
    plain = [(ln.text, ln.chars is not None) for ln in eng.read(img).lines]
    eng.engine.text_rec = _FixedShapes(eng.engine.text_rec)
    tuned = [(ln.text, ln.chars is not None) for ln in eng.read(img).lines]
    assert tuned == plain and any("123" in t for t, _c in plain)


def test_frames_shrink_to_the_detection_height_and_their_bands_alike():
    import numpy as np

    from capture.ocr import OcrEngine, _Float32DetInput
    from capture.profile import Region

    det_in = _Float32DetInput(1280, "max", [0.5] * 3, [0.5] * 3)
    det_in.scale = 800 / 1080
    assert det_in(np.zeros((1080, 2560, 3), np.uint8)).shape == (1, 3, 800, 1888)
    assert det_in(np.zeros((256, 2560, 3), np.uint8)).shape == (1, 3, 192, 1888)  # a band
    eng = OcrEngine(lang="japan", device="cpu", detect_height=800)
    frame = np.full((1440, 2560, 3), 255, np.uint8)
    eng.read(frame, Region(0.0, 0.5, 1.0, 0.1))
    assert eng._prepare.scale == pytest.approx(800 / 1440)  # the frame's scale, not the band's
    eng.read(frame, Region(0.0, 0.5, 1.0, 0.1), fine=True)
    assert eng._prepare.scale == 1.0  # the dialogue: full resolution


def test_the_recognizer_reads_a_crop_it_has_read_before_from_memory():
    import numpy as np
    from rapidocr.ch_ppocr_rec.typings import TextRecInput, TextRecOutput

    from capture.ocr import _Remembered

    calls = []

    class Rec:
        def __call__(self, args):
            calls.append(len(args.img))
            txts = tuple(f"t{int(im[0, 0, 0])}" for im in args.img)
            return TextRecOutput(args.img, txts, tuple(0.9 for _ in txts),
                                 tuple(("w", 0.9, None) for _ in txts), 0.0)  # fmt: skip

    rec = _Remembered(Rec())
    a, b, c = (np.full((8, 20, 3), v, np.uint8) for v in (1, 2, 3))
    out = rec(TextRecInput(img=[a, b, a.copy()], return_word_box=True))
    assert out.txts == ("t1", "t2", "t1") and calls == [2]  # the same pixels: read once
    out = rec(TextRecInput(img=[b, c], return_word_box=True))
    assert out.txts == ("t2", "t3") and calls == [2, 1] and len(out.word_results) == 2
    a2 = a.copy()
    a2[3, 4] = 9  # one pixel differs: read again
    rec(TextRecInput(img=[a2], return_word_box=True))
    assert calls == [2, 1, 1]


def test_korean_rows_keep_the_space_between_words():
    from capture.ocr import join_rows

    assert join_rows(["안녕하세요", "여러분"]) == "안녕하세요 여러분"
    assert join_rows(["こんにちは", "世界"]) == "こんにちは世界"
