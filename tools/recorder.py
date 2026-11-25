"""Recording what the player sees: the game window with the overlay on it, at the game's
frame rate (tools/hud_dev.py --record).

The overlay is invisible to screen recorders on purpose (it is excluded from capture, so
the HUD never reads its own English). The recorder takes over the HUD's screen capture
while it runs: one thread grabs the game window at the recording's frame rate (60 fps
for the test footage) and hands the newest frame to the HUD's 50 ms tick, which runs as
it always does. Onto every frame it grabs it puts what the overlay shows at that moment:
the very pixels the window draws (its backdrops and pre-rendered English, with their
fades), blended the way the desktop composes a translucent window. The frames go to
ffmpeg (H.264 on the GPU when there is one) at a steady rate in real time: a frame that
took long to grab is repeated, never time-stretched.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from hud.overlay.keyed import prepare_layer

log = logging.getLogger("hud")

FFMPEG_CANDIDATES = ("data/tools/ffmpeg.exe", "data/tools/ffmpeg", "ffmpeg")
QUEUE_FRAMES = 240  # frames waiting for the encoder (4 s at 60 fps)
SWITCH_S = 0.001  # Python's thread switch interval while recording (default 5 ms)


Layers = list[tuple[tuple[int, int, np.ndarray], float]]


def blend_layers(frame: np.ndarray, layers: Layers) -> np.ndarray:
    """*frame* (BGR) with the overlay's layers (premultiplied RGBA at x, y; fade) on it,
    blended as the desktop composes the translucent window."""
    return Composer().frame(frame, 0, layers)


class Composer:
    """Puts the overlay window's layers (OverlayWindow.snapshot) onto frames: what a
    recording shows. A layer is made ready once (its premultiplied colour and what it lets
    through, as bytes) and kept while the overlay shows the same pixels with the same
    fade; every frame then costs two quick blends per layer, on the layer's own area.
    (tools/replay.py paints overlay items offline instead, where no window exists.)"""

    def __init__(self) -> None:
        self._ready: dict[int, tuple] = {}  # id of the layer's pixels -> prepared bytes

    def frame(self, frame: np.ndarray, version: int, layers: Layers) -> np.ndarray:
        out = frame.copy()
        fh, fw = out.shape[:2]
        ready: dict[int, tuple] = {}
        for (x, y, rgba), fade in layers:
            got = self._ready.get(id(rgba))
            if got is None or got[0] is not rgba or got[1] != fade:
                got = (rgba, fade, *prepare_layer(rgba, fade))
            ready[id(rgba)] = got
            _, _, colour, through = got
            h, w = rgba.shape[:2]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            sy, sx = slice(y0 - y, y1 - y), slice(x0 - x, x1 - x)
            reg = cv2.multiply(out[y0:y1, x0:x1], through[sy, sx], scale=1.0 / 255.0)
            out[y0:y1, x0:x1] = cv2.add(reg, colour[sy, sx])
        self._ready = ready  # layers no longer shown are let go
        return out


def _raise_priority() -> None:
    """Frames are due on a clock: the recorder's thread goes first when it wakes up (on
    Windows; elsewhere the thread runs at normal priority)."""
    if sys.platform != "win32":
        return
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.SetThreadPriority(k32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST


def _ffmpeg() -> str | None:
    for c in FFMPEG_CANDIDATES:
        if Path(c).exists():
            return str(Path(c).resolve())  # Windows does not start "data/tools/..."
        found = shutil.which(c)
        if found:
            return found
    return None


class Recorder:
    def __init__(self, source, overlay, path: str, fps: float = 60.0, seconds: float = 0.0):
        self.source = source
        self.overlay = overlay
        self.path = Path(path)
        self.fps = fps
        self.seconds = seconds
        self.running = False
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self._proc: subprocess.Popen | None = None
        self.composer = Composer()
        self.frames = 0
        self.repeated = 0
        self.dropped = 0
        self.spent = {"grab": 0.0, "compose": 0.0, "loops": 0}  # seconds, for the log

    def latest(self) -> np.ndarray | None:
        """The newest frame grabbed (for the HUD's tick while the recorder captures)."""
        with self._lock:
            return self._latest

    def start(self, rect: tuple[int, int, int, int]) -> None:
        self.rect = rect
        self.running = True
        self._thread = threading.Thread(target=self._capture, name="recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)

    # ---- the capture thread ----------------------------------------------------------
    def _capture(self) -> None:
        period = 1.0 / self.fps
        t0 = time.perf_counter()
        written = 0
        # the other threads hand over the interpreter more often while frames are due
        switch = sys.getswitchinterval()
        sys.setswitchinterval(SWITCH_S)
        _raise_priority()
        deadline = time.perf_counter()
        try:
            while not self._stop.is_set():
                deadline += period
                tg = time.perf_counter()
                frame = self.source.grab(self.rect)
                self.spent["grab"] += time.perf_counter() - tg
                if frame is not None:
                    with self._lock:
                        self._latest = frame
                cur = self.latest()
                now = time.perf_counter()
                if cur is not None:
                    if self._proc is None:
                        self._open(cur.shape[1], cur.shape[0])
                        # the file's clock starts once the encoder is up (it takes ~0.5 s):
                        # no run of one repeated frame at the start, times match the log
                        now = time.perf_counter()
                        t0, written = now, 0
                    tc = time.perf_counter()
                    version, layers = self.overlay.snapshot()
                    comp = self.composer.frame(cur, version, layers)
                    self.spent["compose"] += time.perf_counter() - tc
                    self.spent["loops"] += 1
                    # a steady rate in real time: a slot passed without a frame repeats it
                    due = int((now - t0) * self.fps) + 1 - written
                    for i in range(max(0, min(due, 10))):
                        self._push(comp)
                        written += 1
                        if i:
                            self.repeated += 1
                if self.seconds and now - t0 >= self.seconds and self._proc is not None:
                    log.info("recording: %.0f s recorded", self.seconds)
                    break
                rest = deadline - time.perf_counter()
                if rest > 0:
                    time.sleep(rest)  # on a fixed clock: one frame every period
                else:
                    deadline = time.perf_counter()  # behind: on from now, no burst
        except Exception:  # noqa: BLE001 - the HUD must go on without its recording
            log.exception("recording failed")
        finally:
            sys.setswitchinterval(switch)
            self.running = False  # the HUD captures on its own again
            self._finish()

    def _push(self, comp: np.ndarray) -> None:
        try:
            self._queue.put_nowait(comp)
            self.frames += 1
        except queue.Full:
            self.dropped += 1

    # ---- the encoder -----------------------------------------------------------------
    def _open(self, w: int, h: int) -> None:
        exe = _ffmpeg()
        if exe is None:
            raise RuntimeError("ffmpeg not found (data/tools/ffmpeg.exe or on PATH)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps:g}",
            "-i", "-",
            *self._codec(exe),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._writer = threading.Thread(target=self._write, name="encoder", daemon=True)
        self._writer.start()
        log.info("recording to %s at %g fps (%dx%d)", self.path, self.fps, w, h)

    @staticmethod
    def _codec(exe: str) -> list[str]:
        """H.264 on the GPU when ffmpeg can (NVENC), else on the CPU, fast."""
        probe = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30,
        )  # fmt: skip
        if probe.returncode == 0:
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]

    def _write(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        while True:
            comp = self._queue.get()
            if comp is None:
                break
            try:
                self._proc.stdin.write(memoryview(np.ascontiguousarray(comp)))
            except OSError:
                log.exception("the encoder stopped")
                break

    def _finish(self) -> None:
        if self._proc is None:
            return
        self._queue.put(None)
        if self._writer is not None:
            self._writer.join(timeout=30)
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        n = max(1, self.spent["loops"])
        log.info(
            "recording closed: %s (%d frames, %d repeated, %d dropped; grab %.1f ms, "
            "compose %.1f ms a frame)",
            self.path, self.frames, self.repeated, self.dropped,
            1000 * self.spent["grab"] / n, 1000 * self.spent["compose"] / n,
        )  # fmt: skip
        self._proc = None
