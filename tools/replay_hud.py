"""Replay a stretch of footage through the HUD core, frame by frame, without a window.

    uv run python tools/replay_hud.py --game azur-promilia --db data/db/test_segment_v2.sqlite \
        --player NAME --start 2790 --duration 120 --out data/out/replay/run.mp4

Time is simulated at the HUD's 50 ms tick: every tick the pixel watch runs on the frame of
that moment; the OCR runs with its real cost, and while it runs the watch keeps going on
the frames that pass, so the covers it places are checked against the frame of the moment
it finished, exactly as in the live HUD. Drawing this script's own video is not counted:
the live HUD draws on other threads. Writes the composite (game + covers) as an mp4,
an event log next to it (placements, removals, dialogue), and prints a summary: covers
placed, removed, and "blinks" (a cover removed and the same line placed again at the same
spot within a second: a removal the pixels did not justify).

To see what the player sees, smoothly: ``--tick-ms 16.667 --watch-ms 50 --lag-ms 15`` (60
frames a second, the watch on the window's 50 ms tick, each backdrop made from the frame
the window was handed before it, as the live overlay has them: a frame behind the game).
``--crop-top 0.5`` leaves out the top of the frame.

``--live 2b`` translates what no database has with PLaMo on this computer (installed from
the app), as the app does: a reading counts once it settled (app.session.LiveReadings), the
conversation and the names go with it, and the translation is shown as late as it took to
make. ``--clean`` leaves the clock out of the video (a demo).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

from hud.app import add_core_args, build_core, config_from_args  # noqa: E402
from hud.overlay import build_backdrop  # noqa: E402
from hud.textwidth import build_table  # noqa: E402
from tools.replay import Animator, compose  # noqa: E402

log = logging.getLogger("hud")


class ReplayLive:
    """Live translation in a replay: PLaMo on this computer, as the app runs it."""

    def __init__(self, core, size: str, game: str, target: str, player: str | None) -> None:  # noqa: ANN001
        from app.paths import models_dir
        from app.session import SETTLE_OPEN_LOCAL_S, LiveReadings
        from translate import GameMemory, PlamoBackend
        from translate.local import MODELS, LocalModels, shared_server

        self.core, self.player = core, player
        self.backend = PlamoBackend(shared_server(LocalModels(models_dir()), size),
                                    MODELS[size].name)
        self.backend.warm_up()
        self.memory = GameMemory(None, game, target=target)
        self.memory.set_glossary(core.glossary.pairs())
        self.readings = LiveReadings(SETTLE_OPEN_LOCAL_S)
        self.asked: set[str] = set()
        self.queue: list[tuple] = []
        self.due: list[tuple] = []
        core.set_live(self.unmatched, on_line=lambda s, t, who: self.memory.remember(
            s, t, who, "dialogue"))

    def unmatched(self, text: str, kind: str, speaker, conf: float, box) -> bool:  # noqa: ANN001
        from app.session import WAIT
        from translate import Item

        verdict = self.readings.judge(text, kind, conf, self.player)
        if not verdict:
            return verdict == WAIT
        if verdict not in self.asked:  # (asked once)
            self.asked.add(verdict)
            self.queue.append((Item(verdict, kind, speaker), text, box))
        return False

    def after_tick(self, now: float) -> None:
        from translate import TranslationError

        for item, text, box in self.queue:
            t0 = time.perf_counter()
            try:
                tr = self.backend.translate([item], self.memory.context())[0]
            except TranslationError as e:
                log.info("live: %s (%s)", item.text, e)
                continue
            log.info("live: %s -> %s", item.text, tr)
            self.due.append((now + time.perf_counter() - t0, text, tr, item, box))
        self.queue.clear()
        for d in [d for d in self.due if d[0] <= now]:
            _at, text, tr, item, box = d
            self.core.add_translation(text, tr, item.kind, item.speaker, box)
            self.due.remove(d)

    def close(self) -> None:
        from translate.local import stop_servers

        stop_servers()


class Footage:
    """Frames of the recording at the HUD's tick rate, scaled like the dev-loop window."""

    def __init__(self, path: str, start: float, width: int, tick: float) -> None:
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.start = start
        self.width = width
        self.tick = tick
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * self.fps)))
        self.index = int(round(start * self.fps))
        self.cache: dict[int, object] = {}

    def at(self, k: int):
        """The frame of tick k (decoded in order, kept until used)."""
        if k in self.cache:
            return self.cache[k]
        want = int(round((self.start + k * self.tick) * self.fps))
        frame = None
        while self.index <= want:
            ok, src = self.cap.read()
            self.index += 1
            if not ok:
                return None
            frame = src
        if frame is None:
            return None
        h = int(round(self.width * frame.shape[0] / frame.shape[1]))
        out = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_LINEAR)
        self.cache[k] = out
        for old in [i for i in self.cache if i < k - 2]:
            del self.cache[old]
        return out


TIMES: dict[str, list[float]] = {"ocr tick": [], "watch": []}  # ms (and backdrops by kind)


class Backdrops:
    """Backdrops as the window has them: made from the frame it was handed, kept while the
    item and that frame stay the same."""

    def __init__(self, timed: bool = False) -> None:
        self.kept: dict[str, tuple] = {}
        self.timed = timed

    def get(self, item, k: int, frame):
        ident = item.identity()
        got = self.kept.get(ident)
        if got is not None and got[0] == item and got[1] == k:
            return got[2]
        t0 = time.perf_counter()
        built = build_backdrop(frame, item)
        if self.timed:
            kind = "glass" if item.frost is not None else item.style
            TIMES.setdefault(f"backdrop {kind}", []).append((time.perf_counter() - t0) * 1000)
        self.kept[ident] = (item, k, built)
        return built


def main() -> int:
    ap = argparse.ArgumentParser()
    add_core_args(ap)
    ap.add_argument("--video", default="data/footage/azur-promilia/cbt_stream_1080p.mp4")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--tick-ms", type=float, default=50.0, help="a video frame every this long")
    ap.add_argument(
        "--watch-ms", type=float, default=0.0, help="the pixel watch runs this often (0: per frame)"
    )
    ap.add_argument(
        "--lag-ms", type=float, default=0.0,
        help="backdrops come from the frame the window was handed this long before",
    )  # fmt: skip
    ap.add_argument("--crop-top", type=float, default=0.0, help="leave out this share at the top")
    ap.add_argument("--out", default="data/out/replay/replay.mp4")
    ap.add_argument("--fresh", action="store_true", help="start from an empty cover cache")
    ap.add_argument("--no-video", action="store_true", help="time the HUD, write no video")
    ap.add_argument("--live", default=None, help="PLaMo's size (2b...): live translation")
    ap.add_argument("--clean", action="store_true", help="no clock in the video")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is None:
        args.cache_dir = str(out.parent / "cache")  # never the live HUD's cache
    if args.fresh:
        (Path(args.cache_dir) / "covers.json").unlink(missing_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(out.with_suffix(".log"), mode="w", encoding="utf-8")],
    )
    app = QApplication.instance() or QApplication(sys.argv[:1])
    build_table()
    core = build_core(config_from_args(args))
    live = ReplayLive(core, args.live, args.game, args.target, args.player) if args.live \
        else None
    tick = args.tick_ms / 1000.0
    feed = Footage(args.video, args.start, args.width, tick)
    writer = None
    n_ticks = int(args.duration / tick)
    stamp = {"k": 0}
    drawing = {"s": 0.0}  # wall time spent composing the preview video (not the HUD's)

    history: dict[int, np.ndarray] = {}  # recent frames: the window's backdrops lag behind
    kept = Backdrops(timed=True)
    animator = Animator()  # fades and typing, as the window does them
    watched = {"t": -1e9}

    def sim_time(k: int) -> float:
        return args.start + k * tick

    def source(k: int) -> int:
        """The frame the window's backdrops are made from at frame k: the newest one it
        was handed (one per watch tick) at least --lag-ms before."""
        if not args.lag_ms:
            return k
        every = (args.watch_ms or args.tick_ms) / 1000.0
        t = k * tick - args.lag_ms / 1000.0
        j = int(round(np.floor(t / every) * every / tick))
        return max(min(history), min(k, j))

    def watch_and_write(k: int, frame) -> None:
        nonlocal writer
        stamp["k"] = k
        history[k] = frame
        for old in [i for i in history if i < k - 30]:
            del history[old]
        now = sim_time(k)
        if not args.watch_ms or now - watched["t"] >= args.watch_ms / 1000.0 - 1e-6:
            watched["t"] = now
            w0 = time.perf_counter()
            core.watch(frame, now)
            TIMES["watch"].append((time.perf_counter() - w0) * 1000)
        t0 = time.perf_counter()
        items = core.items()
        src = source(k)
        comp = compose(
            frame, items, now, animator, lambda it: kept.get(it, src, history[src])
        )  # fmt: skip
        y0 = int(args.crop_top * frame.shape[0])
        if y0:
            comp = np.ascontiguousarray(comp[y0:])
        if not args.clean:
            for colour, width in (((0, 0, 0), 4), ((80, 255, 80), 2)):
                cv2.putText(comp, f"{sim_time(k):7.2f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, colour, width)
        if writer is None and not args.no_video:
            h, w = comp.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / tick, (w, h))
        if writer is not None:
            writer.write(comp)
        # the live HUD draws on other threads: drawing this video costs the OCR nothing
        drawing["s"] += time.perf_counter() - t0

    k = 0
    t_wall = time.perf_counter()
    while k < n_ticks:
        frame = feed.at(k)
        if frame is None:
            break
        watch_and_write(k, frame)
        log.info("@%.2f tick", sim_time(k))
        started = time.perf_counter()
        drawn0 = drawing["s"]
        last = {"k": k}

        def newest(k0=k, t0=started, d0=drawn0, last=last):
            # the OCR has run for this long: the watch went on over the frames that
            # passed meanwhile, and the covers are placed against the one of this moment
            spent = time.perf_counter() - t0 - (drawing["s"] - d0)
            done = k0 + int(spent / tick)
            for j in range(last["k"] + 1, min(done, n_ticks - 1) + 1):
                f = feed.at(j)
                if f is None:
                    break
                watch_and_write(j, f)
                last["k"] = j
            return feed.at(last["k"])

        rep = core.tick(frame, newest, sim_time(k))
        if rep.ui or rep.dialogue:
            TIMES["ocr tick"].append(
                (time.perf_counter() - started - (drawing["s"] - drawn0)) * 1000
            )
        newest()  # frames that passed while the placements were made
        k = last["k"] + 1
        if live is not None:  # (the app translates on its own thread: not the OCR's time)
            live.after_tick(sim_time(k))
        if rep.ui or rep.dialogue:
            log.info(
                "@%.2f ocr ui=%s dialogue=%s total=%.0f ms dialogue=%.0f ms ui=%.0f ms, "
                "%d placed, %d unsettled",
                sim_time(k), rep.ui or "-", rep.dialogue, rep.ocr_ms, rep.dialogue_ms,
                rep.ui_ms, rep.placed, rep.unsettled,
            )  # fmt: skip
    if writer is not None:
        writer.release()
    if live is not None:
        live.close()
    core.cache.save()
    del app
    wall = time.perf_counter() - t_wall
    print(f"replayed {k * tick:.1f} s of footage in {wall:.0f} s -> {out}")
    for name, ms in TIMES.items():
        if ms:
            a = np.array(ms)
            print(
                f"{name}: {len(a)} times, median {np.median(a):.1f} ms, "
                f"95 % {np.percentile(a, 95):.1f} ms, max {a.max():.1f} ms"
            )
    summarize(out.with_suffix(".log"))
    return 0


def summarize(path: Path) -> None:
    """Placements, removals and blinks from the event log."""
    import re

    t = 0.0
    placed: dict[tuple[str, str], float] = {}  # (text, box) -> time placed
    gone: dict[tuple[str, str], float] = {}  # (text, box) -> time removed
    n_place = n_gone = 0
    blinks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"@([\d.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = re.match(r"place: (.*) at (\(.*?\)) font", line)
        if m:
            n_place += 1
            key = (m.group(1), m.group(2))
            if key in gone and t - gone[key] <= 1.0:
                blinks.append(f"{t:7.2f} {m.group(1)[:40]} (gone at {gone[key]:.2f})")
            placed[key] = t
            continue
        m = re.match(r"gone: (.*) \((\w+)\)", line)
        if m:
            n_gone += 1
            for key in [k for k in placed if k[0] == m.group(1)]:
                gone[key] = t
    print(f"covers placed: {n_place}, removed: {n_gone}, blinks: {len(blinks)}")
    for b in blinks[:40]:
        print("  blink", b)


if __name__ == "__main__":
    raise SystemExit(main())
