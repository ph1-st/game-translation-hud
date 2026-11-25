"""The live HUD: find the game's window, capture it, read it, cover its text.

    python -m hud.app --game azur-promilia --db data/db/azur-promilia.sqlite --player Name

The cover protocol is hud/pipeline.py's; this module runs it live:

* an OCR worker thread runs the core's ticks on the newest frame
* the Qt thread, every 50 ms: follows the game window (:class:`GameWindowTracker`),
  captures, runs the pixel watch, hands the overlay its items and says what keeps the
  HUD from working (:class:`Problem`)
* while there is frosted glass, a 16 ms timer remakes it from the newest frame

The overlay is click-through, always on top and never in the captured frames
(hud.platform). Development extras (recording, screenshots, the test player's control
file) are in tools/hud_dev.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from capture.ocr import CPU, DEVICES, OcrEngine
from capture.profile import GAMES_DIR, WindowMatch, load_overrides, load_profile
from db.glossary import Glossary
from db.lookup import Lookup
from hud.config import CONFIRMATION_MODES, COVER_MODES, HudConfig
from hud.covercache import CoverCache
from hud.hotkey import GlobalHotkeys
from hud.overlay import OverlayWindow
from hud.picture import UNKNOWN, find_picture
from hud.pipeline import HudCore
from hud.platform import WHOLE, Platform, Rect, WindowRef, current, use_x11_for_qt
from hud.screen_source import FAST_FPS, FPS, ScreenSource
from hud.style import DEFAULT_STYLE, STYLES, get_style
from hud.textwidth import build_table
from i18n import N_, _
from translate import languages

if TYPE_CHECKING:
    from capture.profile import Profile

log = logging.getLogger("hud")

BLIND_S = 0.5  # no captured frame for this long: every cover comes off
PROBLEM_S = 3.0  # ...and for this long, the HUD says so
FIND_EVERY_S = 1.0  # no game window: how often to look for one
BETTER_WINDOW_S = 5.0  # a window matched by a part of its title: how often to look again
PICTURE_CHECK_S = 2.0  # black bars around the game's picture are looked for this often
GLASS_MS = 16  # frosted glass is remade from a new frame this often (the newer, the calmer)
SAME_FRAME_EVERY_S = 0.1  # a still screen with work left for later is read again this often
WORKER_JOIN_S = 1.5  # stopping waits this long at most for the OCR read under way
WORKER_IDLE_S = 0.02  # after a tick that read nothing, the worker rests this long
WORKER_RETRY_S = 0.2  # after a failed tick
READER_FAILURES = 20  # failed ticks in a row before the HUD says the reader is broken
EXCLUDE_RETRY_S = 1.0  # capture exclusion refused: how often to ask again
FULLSCREEN_CHECK_S = 2.0  # exclusive full screen is checked this often
BLACK_LEVEL = 8  # a frame whose brightest sampled pixel is this dark is black
BLACK_STEP = 8  # every eighth pixel of every eighth row is sampled
BLACK_S = 3.0  # a black picture for this long: the game hides itself from capture


class Problem(StrEnum):
    """What keeps the HUD from working, for the player."""

    NOT_EXCLUDED = "not_excluded"
    NO_FRAMES = "no_frames"
    BLACK_CAPTURE = "black_capture"
    READER_FAILING = "reader_failing"
    EXCLUSIVE_FULLSCREEN = "exclusive_fullscreen"


PROBLEM_TEXT = {
    Problem.NOT_EXCLUDED: N_(
        "The system refused to keep the overlay out of the screen capture, so it would "
        "read its own translations: it stays off."
    ),
    Problem.NO_FRAMES: N_(
        "The game's window cannot be captured right now (a lock screen, a lost display?)."
    ),
    Problem.BLACK_CAPTURE: N_(
        "The game hides its window from screen capture: its text cannot be read."
    ),
    Problem.READER_FAILING: N_("The text reader keeps failing: see the log in the data folder."),
    Problem.EXCLUSIVE_FULLSCREEN: N_(
        "The game runs in exclusive full screen, where no overlay can show: choose "
        "borderless or windowed in the game's display settings."
    ),
}


def problem_text(problem: Problem) -> str:
    """The problem in words, in the interface's language."""
    return _(PROBLEM_TEXT[problem])


# ---- the core -------------------------------------------------------------------------
def start_ocr(profile: Profile, device: str, reader: str = "standard",
              reader_dir: Path | None = None) -> OcrEngine:
    """The text reader for whole game frames, warmed up: MeikiOCR on CUDA when chosen and
    the game is Japanese (capture/cuda_helper.py), else the standard one. A reader that
    fails to start gives way to the next (a usable HUD beats none); ``fallback_reason``
    says why."""
    why = ""
    if reader == "meiki" and reader_dir is not None:
        if str(profile.ocr.get("lang", "japan")) != "japan":
            why = "MeikiOCR reads Japanese only: the standard reader reads this game."
        else:
            from capture.cuda_helper import MeikiCudaEngine

            try:
                meiki = MeikiCudaEngine(reader_dir / "runtime", reader_dir / "models")
                # (the standard reader reads again what MeikiOCR read that matches nothing)
                meiki.second = _start_standard(profile, device)
                return meiki
            except Exception as exc:  # noqa: BLE001 - the standard reader takes over
                log.exception("MeikiOCR (CUDA) could not start")
                why = (f"MeikiOCR (CUDA) could not start ({exc}): the standard reader reads "
                       "instead. Update the NVIDIA driver, or install MeikiOCR again.")
    engine = _start_standard(profile, device)
    if why:
        engine.fallback_reason = why
    return engine


def _start_standard(profile: Profile, device: str) -> OcrEngine:
    try:
        engine = OcrEngine.for_screen(profile.ocr, device=device)
        engine.warm_up()
        return engine
    except Exception as exc:
        if device == CPU:
            raise
        log.exception("the text reader could not start on %s: it starts on the CPU", device)
        engine = OcrEngine.for_screen(profile.ocr, device=CPU)
        engine.requested_device = device
        engine.fallback_reason = (
            f"The GPU reader failed to start ({exc.__class__.__name__}: {exc}). Update the "
            "graphics driver, then try the GPU again."
        )
        engine.warm_up()
        return engine


def build_core(config: HudConfig, engine: OcrEngine | None = None) -> HudCore:
    """The HUD core for *config*: the profile, the databases, the glossary, the cover
    cache and *engine*, the text reader (by default the profile's, warmed up: that takes
    seconds, so build it off the GUI thread)."""
    profile = config.profile or load_profile(config.game)
    if config.overrides:
        profile = profile.with_overrides(load_overrides(config.overrides))
    lookup = Lookup.from_dbs(
        list(config.dbs), player_name=config.player,
        player_fallback=languages.you(config.target_lang),
    )  # fmt: skip
    glossary = config.glossary
    if not isinstance(glossary, Glossary):
        # the databases' own names first (their lines use them), then the game's file
        path = glossary or GAMES_DIR / config.game / "glossary.yaml"
        glossary = Glossary.combine(
            Glossary.from_dbs(config.dbs), Glossary.load(path, target=config.target_lang)
        )
    cache = CoverCache(Path(config.cache_dir) / "covers.json" if config.cache_dir else None)
    return HudCore(
        profile, lookup, glossary, cache,
        engine or start_ocr(profile, config.device, config.reader, config.reader_dir),
        get_style(config.cover_style), cover_mode=config.cover_mode,
        confirmation_mode=config.confirmation_mode, target_lang=config.target_lang,
    )  # fmt: skip


class OcrWorker(threading.Thread):
    """Runs the core's ticks on the newest frame, one after the other, while there is
    something to read: a new frame, or work the core left for the same frame."""

    def __init__(self, core: HudCore, verbose: bool) -> None:
        super().__init__(daemon=True, name="ocr")
        self.core = core
        self.verbose = verbose
        self._frame: np.ndarray | None = None
        self._token: object = None  # which frame it is (the capture's sequence number...)
        self._lock = threading.Lock()
        self._new = threading.Event()
        # (not "_stop": threading.Thread has a method of that name, which join() calls)
        self._halt = threading.Event()
        self._hurry = False  # a live translation came in: placed at once, frame or not
        self.paused = False
        self.failures = 0  # ticks failed in a row
        core.wake = self.hurry

    def offer(self, frame: np.ndarray, token: object = None) -> None:
        """The newest frame; *token* tells frames apart (the same token: the same frame)."""
        with self._lock:
            self._frame, self._token = frame, token if token is not None else id(frame)
        self._new.set()

    def newest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def hurry(self) -> None:
        """A tick now, even without a new frame (any thread)."""
        self._hurry = True
        self._new.set()

    def stop(self) -> None:
        self._halt.set()
        self._new.set()

    def run(self) -> None:
        last_token: object = None
        last_at = 0.0
        while not self._halt.is_set():
            self._new.wait(SAME_FRAME_EVERY_S)
            self._new.clear()
            hurry, self._hurry = self._hurry, False
            with self._lock:
                frame, token = self._frame, self._token
            if self._halt.is_set() or frame is None or self.paused:
                continue
            now = time.monotonic()
            if token == last_token and not hurry and not (
                self.core.pending() and now - last_at >= SAME_FRAME_EVERY_S
            ):
                continue  # read already, nothing left for later: wait for the next frame
            last_token, last_at = token, now
            self._tick(frame, now)

    def _tick(self, frame: np.ndarray, now: float) -> None:
        try:
            rep = self.core.tick(frame, self.newest, now)
            self.failures = 0
        except Exception:  # noqa: BLE001 - the HUD stays alive; the problem is reported
            self.failures += 1
            if self.failures <= 3 or self.failures % 100 == 0:
                log.exception("ocr tick failed (%d in a row)", self.failures)
            self._halt.wait(WORKER_RETRY_S)
            return
        if self.verbose and (rep.ui or rep.dialogue):
            log.info(
                "tick: ui=%s dialogue=%s total=%.0f ms dialogue=%.0f ms ui=%.0f ms, "
                "%d placed, %d unsettled",
                rep.ui or "-", rep.dialogue, rep.ocr_ms, rep.dialogue_ms, rep.ui_ms,
                rep.placed, rep.unsettled,
            )  # fmt: skip
        if not rep.ui and not rep.dialogue:
            self._halt.wait(WORKER_IDLE_S)


# ---- the game's window ------------------------------------------------------------------
class GameWindowTracker:
    """Finds the game's window and follows it: where its client area is, when it goes,
    a better match turning up, and where the game's picture lies in it (a 16:9 game full
    screen on a 21:9 monitor sits between black bars; the profile's regions are fractions
    of the picture)."""

    FOUND, LOST, CHANGED, SAME, NONE = "found", "lost", "changed", "same", "none"

    def __init__(self, platform: Platform, match: WindowMatch,
                 grab: Callable[[Rect], np.ndarray | None]) -> None:  # fmt: skip
        self.platform = platform
        self.match = match
        self.grab = grab  # the newest frame of a screen rect (the capture's)
        self.window: WindowRef | None = None
        self.found_at = 0.0
        self._last_find = -1e9
        # the picture within the client area (x, y, w, h); None: the whole client area
        self.picture: Rect | None = None
        self._picture_seen: object = None  # a different finding, taken when seen again
        self._picture_check = -1e9

    def refresh(self, now: float) -> str:
        """Follow the window: FOUND (a new one), LOST, CHANGED (a better match: everything
        from scratch), SAME, or NONE (still no window)."""
        if self.window is not None:
            rect = self.platform.client_rect(self.window.id)
            if rect is None:
                log.info("game window lost")
                self._forget()
                return self.LOST
            self.window.client = rect
            if self.window.score < WHOLE and now - self._last_find >= BETTER_WINDOW_S:
                self._last_find = now
                return self._look_for_better(now)
            return self.SAME
        if now - self._last_find < FIND_EVERY_S:
            return self.NONE
        self._last_find = now
        self.window = self.platform.find(self.match)
        if self.window is None:
            return self.NONE
        self.found_at = now
        log.info("found window '%s' at %s", self.window.title, self.window.client)
        return self.FOUND

    def _look_for_better(self, now: float) -> str:
        """Matched by a part of its title: the game may have opened since (or a better
        window of an emulator, larger)."""
        better = self.platform.find(self.match)
        cur = self.window
        assert cur is not None

        def rank(w: WindowRef) -> tuple[int, int]:
            return w.score, w.client[2] * w.client[3]

        if better is None or better.id == cur.id or rank(better) <= rank(cur):
            return self.SAME
        log.info("found a better match for the game window: '%s'", better.title)
        self._forget()
        self.window, self.found_at = better, now
        return self.CHANGED

    def _forget(self) -> None:
        self.window = None
        self.picture = self._picture_seen = None
        self._picture_check = -1e9

    def look_for_bars(self, client: Rect, now: float) -> bool:
        """Every few seconds, for a profile that knows its picture's shape: is the picture
        between black bars? A different finding is taken when the next look agrees.
        Returns whether the picture changed (covers placed in the old one must go)."""
        aspect = self.match.aspect
        if not aspect or now - self._picture_check < PICTURE_CHECK_S:
            return False
        self._picture_check = now
        whole = self.grab(client)
        if whole is None or whole.shape[:2] != (client[3], client[2]):
            return False
        found = find_picture(whole, aspect)
        if found == UNKNOWN or found == self.picture:
            self._picture_seen = None
            return False
        if found != self._picture_seen:
            self._picture_seen = found  # once more to be sure
            return False
        self.picture, self._picture_seen = found, None
        log.info(
            "game picture: %s",
            "the whole window" if found is None else f"{found} between black bars",
        )
        return True

    def picture_rect(self, client: Rect) -> Rect:
        """The game's picture on the screen: the client area, or the part between bars."""
        if self.picture is None:
            return client
        cx, cy, cw, ch = client
        px, py, pw, ph = self.picture
        if px + pw > cw or py + ph > ch:  # the window shrank since: look again
            self.picture, self._picture_check = None, -1e9
            return client
        return cx + px, cy + py, pw, ph


# ---- the HUD ----------------------------------------------------------------------------
class HudApp:
    def __init__(self, config: HudConfig, core: HudCore | None = None) -> None:
        self.config = config
        self.core = core or build_core(config)
        self.profile = self.core.profile
        self.platform = current()
        self.worker = OcrWorker(self.core, config.verbose)
        self.source = ScreenSource(config.capture)
        self.overlay = OverlayWindow()
        build_table()  # text metrics for the layout (GUI thread only)
        match = self.profile.window
        if config.title:  # a title of its own instead of the profile's
            match = dataclasses.replace(match, title_patterns=[config.title], exact_titles=[])
        self.tracker = GameWindowTracker(self.platform, match, self.source.grab)
        self.hotkeys = GlobalHotkeys()
        self.hotkey_failures: list[str] = []  # hotkeys another program has
        self.problem: Problem | None = None
        self.started = time.monotonic()
        self.ticks = 0
        self._rect: Rect | None = None  # where the overlay lies (the game's picture)
        self._version: tuple | None = None
        self._last_raise = -1
        self._glass_seq = -1  # the frame the glass was last made from
        self._exclude_at = -1e9
        self._fullscreen_at = -1e9
        self._exclusive = False
        self._black_since: float | None = None
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)  # 50 ms, not the 15.6 ms grid
        self.timer.setInterval(config.interval_ms)
        self.timer.timeout.connect(self.tick)
        self.glass_timer = QTimer()
        self.glass_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.glass_timer.setInterval(GLASS_MS)
        self.glass_timer.timeout.connect(self._glass_tick)

    @property
    def window(self) -> WindowRef | None:
        return self.tracker.window

    def start(self, app: QApplication) -> None:
        self.worker.start()
        self.timer.start()
        self.glass_timer.start()
        self.hotkeys.install(app)
        on_quit = self.config.on_quit or app.quit  # (the app keeps running; the CLI quits)
        for spec, action in self.hotkey_actions(on_quit):
            if not self.hotkeys.register(spec, action):
                log.warning("could not register hotkey %s", spec)
                self.hotkey_failures.append(spec)
        log.info(
            "HUD started: %d lines, ocr=%s, platform=%s, toggle=%s quit=%s",
            len(self.core.maker.lookup.keys), self.core.engine.device, self.platform.name,
            self.config.toggle_hotkey, self.config.quit_hotkey,
        )  # fmt: skip

    def hotkey_actions(self, on_quit: Callable[[], None]) -> list[tuple[str, Callable]]:
        return [(self.config.toggle_hotkey, self.toggle), (self.config.quit_hotkey, on_quit)]

    def stop(self) -> None:
        """Everything stopped and put away; a step that fails is logged and the others
        still run (the overlay must never stay up, nor its hotkeys taken)."""
        steps = [
            self.timer.stop,
            self.glass_timer.stop,
            self.worker.stop,
            # the read under way finishes first (a restart must not find it still running)
            lambda: self.worker.join(timeout=WORKER_JOIN_S) if self.worker.is_alive() else None,
            self.overlay.close_builder,
            self.overlay.hide,
            self.hotkeys.uninstall,
            self.core.cache.save,
            self.core.save_memory,
            self.source.close,
            getattr(self.core.engine, "close", lambda: None),  # (a reader's own process)
            self.overlay.deleteLater,
        ]
        for step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 - the rest still runs
                log.exception("stopping the HUD: a step failed")

    def toggle(self) -> None:
        self.overlay.set_enabled(not self.overlay.enabled)
        log.info("overlay %s", "on" if self.overlay.enabled else "off")

    def diagnostics(self) -> dict[str, object]:
        from app import VERSION

        out = self.core.diagnostics()
        out.update({
            "version": VERSION,
            "ocr": self.core.engine.diagnostics(),
            "platform": self.platform.name,
            "capture": self.source.backend,
            "overlay": "colour key" if self.overlay.colour_key else "translucent",
            "exclude_error": self.overlay.exclude_error,
            "window": self.window.title if self.window is not None else "",
            "problem": self.problem.value if self.problem else "",
            "problem_text": problem_text(self.problem) if self.problem else "",
            "worker_failures": self.worker.failures,
        })  # fmt: skip
        return out

    # ---- every 50 ms ----------------------------------------------------------------------
    def tick(self) -> None:
        self.ticks += 1
        now = time.monotonic()
        state = self.tracker.refresh(now)
        if self.window is None:
            if state == GameWindowTracker.LOST or self._rect is not None:
                self._lose_window()
            return
        if state == GameWindowTracker.CHANGED:
            self._lose_window()
        client = self.window.client
        self.source.set_area(client, self.window.id)  # the whole window is captured
        self.source.fps = self.capture_fps()
        if self.tracker.look_for_bars(client, now):
            self.core.clear()  # (placed in the other picture)
        rect = self.tracker.picture_rect(client)
        self._place_overlay(rect, now)
        if not self.overlay.capture_excluded:
            # the HUD would read its own translations (and blink): nothing is shown at all
            self.problem = Problem.NOT_EXCLUDED
            if self.core.items():
                self.core.clear()
            return
        self._keep_on_top(now)
        frame = self.frame_for(rect)
        blind = self.source.age() > BLIND_S
        self.problem = self._problem(frame, now)
        if frame is not None and not blind:
            self.worker.offer(frame, (self.source.seq, rect))
            self.core.watch(frame, now)  # the pixel watch: every covered line, no OCR
        elif blind and self.core.items():
            # nothing captured for a while (a lost display, the lock screen, a window off
            # every screen): a blind HUD shows nothing rather than covers it cannot check
            self.core.clear()
            log.info("no frames for %.1f s: covers off", self.source.age())
        version = self.core.version()
        if version != self._version:
            self._version = version
            items = self.core.items()
            self.overlay.set_items(items)
            log.debug("overlay: %d items", len(items))
        if frame is not None:
            self.overlay.set_frame(frame)  # what is painted over follows the screen

    def frame_for(self, rect: Rect) -> np.ndarray | None:
        """The newest frame of *rect* (the game's picture)."""
        return self.source.grab(rect)

    def capture_fps(self) -> float:
        return FAST_FPS if self.overlay.has_glass() else FPS

    def _lose_window(self) -> None:
        """The window went (or another one took its place): a window that comes back is a
        new one, even at the same place: shown, measured and read from scratch."""
        self.overlay.hide()
        self._rect = None
        self.core.clear()
        self.source.set_area(None)
        self._black_since = None

    def _place_overlay(self, rect: Rect, now: float) -> None:
        if rect != self._rect or self.overlay.misplaced(rect):
            self._rect = rect
            self.overlay.show_over(rect)
            self._raised()
            if not self.overlay.capture_excluded:
                log.warning("the overlay is NOT excluded from capture (%s); asking again",
                            self.platform.exclude_detail())
        elif not self.overlay.capture_excluded and now - self._exclude_at >= EXCLUDE_RETRY_S:
            self._exclude_at = now  # (refused once: the system may say yes now)
            self.overlay.prepare()

    def _keep_on_top(self, now: float) -> None:
        second = int(now - self.started)
        if self.overlay.isVisible() and second != self._last_raise:
            self._last_raise = second
            self.overlay.raise_()  # above the game even after it is clicked
            self._raised()

    def _problem(self, frame: np.ndarray | None, now: float) -> Problem | None:
        if self.source.age() > PROBLEM_S:
            return Problem.NO_FRAMES
        if self.worker.failures >= READER_FAILURES:
            return Problem.READER_FAILING
        if frame is not None and self._black_for_long(frame, now):
            return Problem.BLACK_CAPTURE
        if now - self._fullscreen_at >= FULLSCREEN_CHECK_S:
            self._fullscreen_at = now
            self._exclusive = self.platform.exclusive_fullscreen(self.window)
        return Problem.EXCLUSIVE_FULLSCREEN if self._exclusive else None

    def _black_for_long(self, frame: np.ndarray, now: float) -> bool:
        """Has the visible game window been pure black in every frame for BLACK_S? (A game
        protected from capture shows as black; a fade or a loading screen never lasts.)"""
        if int(frame[::BLACK_STEP, ::BLACK_STEP].max(initial=0)) > BLACK_LEVEL:
            self._black_since = None
            return False
        if self._black_since is None:
            self._black_since = now
        return now - self._black_since >= BLACK_S

    def _raised(self) -> None:
        """The overlay went to the top: the app's own panel goes back above it."""
        if self.config.on_raise is not None:
            self.config.on_raise()

    def _glass_tick(self) -> None:
        """Frosted glass is a blur of the scene behind it: made from the newest frame every
        few milliseconds, it moves with the scene (the window's tick is 50 ms)."""
        if self.window is None or self._rect is None or not self.overlay.has_glass():
            return
        if self.source.seq == self._glass_seq:
            return  # no newer frame: the glass is made from this one already
        self._glass_seq = self.source.seq
        frame = self.frame_for(self._rect)
        if frame is not None and frame.shape[:2] == (self._rect[3], self._rect[2]):
            self.overlay.refresh_glass(frame)

    def summary(self) -> str:
        c = self.core
        avg = c.ocr_ms / c.ocr_calls if c.ocr_calls else 0.0
        return f"ticks={self.ticks} ocr_calls={c.ocr_calls} avg_ocr_ms={avg:.0f}"


# ---- the command line ---------------------------------------------------------------------
def add_core_args(ap: argparse.ArgumentParser) -> None:
    """Options shared by the HUD and the offline tools that run its core."""
    ap.add_argument("--game", required=True)
    ap.add_argument("--db", action="append", required=True,
                    help="a translation database (again for more: the first wins)")
    ap.add_argument("--player", default=None)
    ap.add_argument("--target", default="en", help="the databases' language (en, es-419...)")
    ap.add_argument("--device", default="auto", choices=DEVICES, help="OCR device")
    ap.add_argument("--reader", default="standard", choices=("standard", "meiki"),
                    help="meiki: MeikiOCR on CUDA, installed from the app")
    ap.add_argument("--confirmation-mode", default="balanced", choices=CONFIRMATION_MODES)
    ap.add_argument("--cover-mode", default="grow", choices=COVER_MODES)
    ap.add_argument(
        "--cover-style", default=DEFAULT_STYLE, choices=sorted(STYLES),
        help="the glow under the English: glow (faint white) | shade (faint dark)",
    )  # fmt: skip
    ap.add_argument("--overrides", default=None, help="extra profile yaml (e.g. a recording)")
    ap.add_argument("--cache-dir", default=None, help="cover cache folder (default "
                    "data/cache/<game>)")  # fmt: skip
    ap.add_argument("--no-cache", action="store_true", help="do not load or save the cover cache")


def add_window_args(ap: argparse.ArgumentParser) -> None:
    """Options of the live HUD (its window, capture and hotkeys)."""
    ap.add_argument("--title", default=None, help="window title regex (overrides the profile)")
    ap.add_argument("--capture", default="auto", help="auto | dxcam | mss (Windows)")
    ap.add_argument("--interval-ms", type=int, default=50, help="capture + pixel watch period")
    ap.add_argument("--toggle-hotkey", default="ctrl+shift+F8")
    ap.add_argument("--quit-hotkey", default="ctrl+shift+F9")
    ap.add_argument("--verbose", action="store_true")


def config_from_args(args: argparse.Namespace) -> HudConfig:
    """A HudConfig from the options of add_core_args (and add_window_args, if given)."""
    cache = None if args.no_cache else Path(args.cache_dir or Path("data/cache") / args.game)
    config = HudConfig(
        game=args.game, dbs=[Path(p) for p in args.db], player=args.player,
        target_lang=args.target, overrides=Path(args.overrides) if args.overrides else None,
        cache_dir=cache, device=args.device, confirmation_mode=args.confirmation_mode,
        cover_style=args.cover_style, cover_mode=args.cover_mode, reader=args.reader,
    )  # fmt: skip
    if args.reader == "meiki":
        from app.paths import models_dir
        from app.reader import MeikiReader

        config.reader_dir = MeikiReader(models_dir()).folder
    for name in ("title", "capture", "interval_ms", "toggle_hotkey", "quit_hotkey", "verbose"):
        if hasattr(args, name):
            setattr(config, name, getattr(args, name))
    return config


def run(hud_factory: Callable[[HudConfig], HudApp], config: HudConfig) -> int:
    """Run a HUD until it is quit; the process's exit code."""
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    if config.verbose:  # (when the picture rested, when the overlay got its covers)
        log.setLevel(logging.DEBUG)
    cv2.setNumThreads(2)
    use_x11_for_qt()
    app = QApplication.instance() or QApplication(sys.argv)
    hud = hud_factory(config)
    hud.start(app)
    code = app.exec()
    hud.stop()
    log.info("HUD stopped: %s", hud.summary())
    return int(code)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_core_args(ap)
    add_window_args(ap)
    return run(HudApp, config_from_args(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
