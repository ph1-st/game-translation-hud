"""The live HUD with development extras:

    python -m tools.hud_dev --game azur-promilia --db data/db/x.sqlite --player Name \
        [--record out.mp4 --record-seconds 60] [--screenshot shot.png --screenshot-at 10] \
        [--control data/out/hud_control.json] [--duration 120]

* --record: what the player sees (the game with the overlay on it) as an mp4. The overlay
  is invisible to screen recorders on purpose, so the recorder puts the overlay window's
  own pixels on every frame it grabs (tools/recorder.py).
* --screenshot and the shot hotkey: one such picture.
* --control: the test player (tools/play_video.py) pauses the HUD and asks for pictures.
* --duration: quit after that many seconds.
* --live 2b: what no database has is translated by PLaMo on this computer (installed from
  the app), as the app does it, and shown faded like the app shows it.
* --whole-window: the game's picture is its whole window (no looking for black bars): a
  recording never stops because a dark screen looked like bars.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
from PySide6.QtWidgets import QApplication

from hud.app import HudApp, add_core_args, add_window_args, config_from_args, run
from hud.config import HudConfig
from hud.screen_source import FAST_FPS
from tools.recorder import Recorder, blend_layers

log = logging.getLogger("hud")

RECORD_AFTER_S = 1.0  # a recording starts this long after the game window appeared: a
# window just created still shows what was on the screen before it


@dataclass
class DevOptions:
    record: Path | None = None
    record_fps: float = 60.0
    record_seconds: float = 0.0  # 0: until the HUD quits
    screenshot: Path | None = None
    screenshot_at: float = 10.0  # seconds after the start
    shot_hotkey: str | None = "ctrl+shift+F10"
    shots_dir: Path = Path("data/out/shots")
    control: Path | None = None
    duration: float = 0.0  # 0: until quit
    live: str = ""  # PLaMo's size ("2b"): live translation
    whole_window: bool = False  # the picture is the whole window (no black bars)


class Live:
    """Live translation in the dev HUD, as the app's session runs it: a reading no database
    has, once settled (app.session.LiveReadings), goes to PLaMo on this computer; its
    translation comes back to the overlay. The conversation shown is the context."""

    def __init__(self, core, size: str, game: str, target: str, player: str | None) -> None:  # noqa: ANN001
        from app.paths import models_dir
        from app.session import SETTLE_OPEN_LOCAL_S, LiveReadings
        from translate import GameMemory, LiveTranslator, PlamoBackend
        from translate.local import MODELS, LocalModels, shared_server

        self.core, self.player = core, player
        backend = PlamoBackend(shared_server(LocalModels(models_dir()), size), MODELS[size].name)
        self.memory = GameMemory(None, game, target=target)
        self.memory.set_glossary(core.glossary.pairs())
        self.readings = LiveReadings(SETTLE_OPEN_LOCAL_S)
        self.asked: dict[str, tuple] = {}
        self.translator = LiveTranslator(backend, self.memory, on_result=self._translated,
                                         batch_wait=0.02)
        core.set_live(self._unmatched, on_line=lambda s, t, who: self.memory.remember(
            s, t, who, "dialogue"))
        self.translator.start()
        threading.Thread(target=backend.warm_up, name="local-model", daemon=True).start()

    def _unmatched(self, text: str, kind: str, speaker, conf: float, box) -> bool:  # noqa: ANN001
        from app.session import WAIT
        from translate import Item

        verdict = self.readings.judge(text, kind, conf, self.player)
        if not verdict:
            return verdict == WAIT
        item = Item(verdict, kind, speaker)
        self.asked[item.text] = (text, box)
        self.translator.request(item)
        return False

    def _translated(self, item, translation: str) -> None:  # noqa: ANN001
        text, box = self.asked.get(item.text, (item.text, None))
        self.core.add_translation(text, translation, item.kind, item.speaker, box)

    def stop(self) -> None:
        from translate.local import stop_servers

        self.translator.stop(timeout=0)
        stop_servers()


class DevHud(HudApp):
    def __init__(self, config: HudConfig, options: DevOptions) -> None:
        super().__init__(config)
        self.options = options
        self.recorder: Recorder | None = None
        self.paused = False
        self._control_mtime = 0.0
        self._last_snap = 0
        self._shot_done = False
        self.live = Live(self.core, options.live, config.game, config.target_lang,
                         config.player) if options.live else None
        if options.whole_window:
            self.tracker.match = dataclasses.replace(self.tracker.match, aspect=None)

    def hotkey_actions(self, on_quit: Callable[[], None]) -> list[tuple[str, Callable]]:
        keys = super().hotkey_actions(on_quit)
        if self.options.shot_hotkey:
            keys.append((self.options.shot_hotkey, self.snap))
        return keys

    def _recording(self) -> bool:
        return self.recorder is not None and self.recorder.running

    def frame_for(self, rect):  # noqa: ANN001, ANN201 - HudApp.frame_for
        if self._recording():
            return self.recorder.latest()  # (it holds the capture while it records)
        return super().frame_for(rect)

    def capture_fps(self) -> float:
        return FAST_FPS if self._recording() else super().capture_fps()

    def tick(self) -> None:
        elapsed = time.monotonic() - self.started
        if self.options.duration and elapsed > self.options.duration:
            QApplication.instance().quit()
            return
        self._poll_control()
        if self.paused:
            return  # frozen: no new frames, no removals, the overlay stays as it is
        self._stop_recording_if_moved()
        super().tick()
        self._start_recording()
        opts = self.options
        if (opts.screenshot and not self._shot_done and elapsed >= opts.screenshot_at
                and self.core.items()):  # fmt: skip
            self._shot_done = True
            self.snap(opts.screenshot)

    def _glass_tick(self) -> None:
        if not self.paused:
            super()._glass_tick()

    def _start_recording(self) -> None:
        opts = self.options
        if (opts.record is None or self.recorder is not None or self._rect is None
                or time.monotonic() - self.tracker.found_at < RECORD_AFTER_S):  # fmt: skip
            return
        self.recorder = Recorder(self.source, self.overlay, str(opts.record), opts.record_fps,
                                 opts.record_seconds)  # fmt: skip
        self.recorder.start(self._rect)

    def _stop_recording_if_moved(self) -> None:
        """A recording is of a fixed area of the screen: without the game there, it would
        record whatever else is on the screen."""
        if not self._recording():
            return
        window = self.window
        if window is None or self.tracker.picture_rect(window.client) != self.recorder.rect:
            log.info("recording stopped: the game window %s",
                     "went" if window is None else "moved")  # fmt: skip
            self.recorder.stop()

    def snap(self, path: Path | None = None) -> None:
        """A picture of what the player sees: the newest frame with the overlay's own pixels
        on it (nothing is made capturable for it)."""
        if self._rect is None:
            log.info("no game window: nothing to snapshot")
            return
        frame = self.frame_for(self._rect)
        if frame is None:
            return
        _version, layers = self.overlay.snapshot()
        if path is None:
            self.options.shots_dir.mkdir(parents=True, exist_ok=True)
            path = self.options.shots_dir / f"hud_{time.strftime('%H%M%S')}.png"
        cv2.imwrite(str(path), blend_layers(frame, layers))
        log.info("screenshot written to %s", path)

    def _poll_control(self) -> None:
        """The test player writes {paused, snap_id, timecode}: pause freezes the HUD, a new
        snap_id saves a picture beside the control file."""
        path = self.options.control
        if path is None:
            return
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if mtime == self._control_mtime:
            return
        self._control_mtime = mtime
        try:
            ctl = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        paused = bool(ctl.get("paused", False))
        if paused != self.paused:
            self.paused = self.worker.paused = paused
            log.info("HUD %s", "paused" if paused else "resumed")
        snap_id = int(ctl.get("snap_id", 0))
        if snap_id > self._last_snap:
            self._last_snap = snap_id
            tc = str(ctl.get("timecode", "")).replace(":", "") or time.strftime("%H%M%S")
            self.snap(path.parent / f"hud_{tc}.png")

    def stop(self) -> None:
        if self.recorder is not None:
            self.recorder.stop()
        if self.live is not None:
            self.live.stop()
        super().stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_core_args(ap)
    add_window_args(ap)
    ap.add_argument("--record", type=Path, help="record what the player sees to this mp4")
    ap.add_argument("--record-fps", type=float, default=60.0)
    ap.add_argument("--record-seconds", type=float, default=0.0, help="0: until quit")
    ap.add_argument("--screenshot", type=Path, help="write one composite picture")
    ap.add_argument("--screenshot-at", type=float, default=10.0, help="seconds after start")
    ap.add_argument("--shot-hotkey", default="ctrl+shift+F10")
    ap.add_argument("--control", type=Path, help="the test player's control file")
    ap.add_argument("--duration", type=float, default=0.0, help="quit after N seconds")
    ap.add_argument("--live", default="", help="PLaMo's size (2b...): live translation")
    ap.add_argument("--whole-window", action="store_true", help="no looking for black bars")
    args = ap.parse_args(argv)
    options = DevOptions(
        record=args.record, record_fps=args.record_fps, record_seconds=args.record_seconds,
        screenshot=args.screenshot, screenshot_at=args.screenshot_at,
        shot_hotkey=args.shot_hotkey or None, control=args.control, duration=args.duration,
        live=args.live, whole_window=args.whole_window,
    )  # fmt: skip
    return run(lambda config: DevHud(config, options), config_from_args(args))


if __name__ == "__main__":
    sys.exit(main())
