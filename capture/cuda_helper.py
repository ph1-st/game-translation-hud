"""MeikiOCR on an NVIDIA graphics card (CUDA), in a process of its own.

The app ships ONNX Runtime's DirectML build; its CUDA build cannot be loaded beside it in
one process (both are the module ``onnxruntime``). So the CUDA build, and NVIDIA's
libraries, are downloaded once with the player's consent (app/reader.py) and a helper
process runs the reader with them: the same Python (the app's own executable, or this
interpreter), with ``onnxruntime`` taken from the downloaded folder.

The HUD's side, :class:`MeikiCudaEngine`, has the interface of capture.ocr.OcrEngine. A
read copies the picture into shared memory, sends one line of JSON to the helper's stdin
and gets the readings back as one line of JSON on its stdout (a few milliseconds besides
the read itself).

    python -m capture.cuda_helper RUNTIME_DIR MODELS_DIR     (the helper; app: --ocr-helper)
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from capture.ocr import OcrChar, OcrLine, OcrResult
from capture.profile import Region

log = logging.getLogger(__name__)

HELPER_FLAG = "--ocr-helper"  # the app's executable runs the helper with this
START_TIMEOUT_S = 120.0  # loading CUDA and the models, and the first reads (seconds)
READ_TIMEOUT_S = 10.0
CUDA = "cuda"


# ---- the helper's side --------------------------------------------------------------------
class _RuntimeFinder(importlib.abc.MetaPathFinder):
    """``onnxruntime`` and its submodules from the downloaded folder, before any other
    (the app's own build, frozen or installed, never answers for it)."""

    def __init__(self, root: Path) -> None:
        self.root = str(root)

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001, ANN201
        if fullname != "onnxruntime" and not fullname.startswith("onnxruntime."):
            return None
        where = [self.root] if fullname == "onnxruntime" else path
        return importlib.machinery.PathFinder.find_spec(fullname, where, target)


def use_runtime(runtime: Path) -> None:
    """Load ``onnxruntime`` and NVIDIA's libraries from *runtime* (app/reader.py's folder)."""
    for bin_dir in sorted(runtime.glob("nvidia/*/bin")):
        os.add_dll_directory(str(bin_dir))  # (cuDNN loads parts of itself later, by name)
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    sys.meta_path.insert(0, _RuntimeFinder(runtime))
    for name in [m for m in sys.modules if m == "onnxruntime" or m.startswith("onnxruntime.")]:
        del sys.modules[name]


def serve(runtime: Path, models: Path) -> int:
    """The helper: read what is asked, one request at a time, until stdin closes."""
    from multiprocessing import shared_memory

    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # (nothing but replies goes to stdout)

    def reply(msg: dict) -> None:
        out.write(json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n")
        out.flush()

    try:
        use_runtime(runtime)
        from capture.meiki import MeikiEngine

        engine = MeikiEngine(models, CUDA)
        engine.warm_up()
        if engine.device != CUDA:
            raise RuntimeError(engine.fallback_reason or "CUDA did not start")
    except Exception as e:  # noqa: BLE001 - told to the app, which reads on without it
        reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    reply({"ok": True, "device": engine.device})
    mem = None
    for raw in sys.stdin.buffer:
        req = json.loads(raw)
        if req.get("op") == "quit":
            break
        try:
            if mem is None or mem.name != req["shm"]:
                if mem is not None:
                    mem.close()
                mem = shared_memory.SharedMemory(name=req["shm"])
            h, w = req["h"], req["w"]
            img = np.ndarray((h, w, 3), dtype=np.uint8, buffer=mem.buf)
            res = engine.read(img)
            reply({"ok": True, "ms": res.elapsed_ms, "lines": [
                [ln.text, ln.conf, *_px(ln.box, w, h),
                 [[c.ch, c.x0 * w, c.x1 * w] for c in ln.chars or ()]]
                for ln in res.lines
            ]})  # fmt: skip
        except Exception as e:  # noqa: BLE001 - one failed read is reported, the next goes on
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
    if mem is not None:
        mem.close()
    return 0


def _px(box: Region, w: int, h: int) -> list[float]:
    return [box.x * w, box.y * h, box.w * w, box.h * h]


# ---- the HUD's side ---------------------------------------------------------------------
class MeikiCudaEngine:
    """MeikiOCR on CUDA through the helper process; the interface of OcrEngine."""

    lang = "japan"

    def __init__(self, runtime: Path, models: Path) -> None:
        from multiprocessing import shared_memory

        self.requested_device = CUDA
        self.fallback_reason = ""
        self.second = None  # the standard reader, for second readings (hud/app.py)
        self.device = CUDA
        self._lock = threading.Lock()
        self._shm_cls = shared_memory.SharedMemory
        self._mem = None
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, HELPER_FLAG, str(runtime), str(models)]
        else:
            cmd = [sys.executable, "-m", "capture.cuda_helper", str(runtime), str(models)]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        root = Path(__file__).resolve().parent.parent
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=root, creationflags=flags,
        )  # fmt: skip
        hello = self._receive(START_TIMEOUT_S)
        if not hello.get("ok"):
            self.close()
            raise RuntimeError(hello.get("error") or "the CUDA reader did not start")

    def warm_up(self) -> None:
        """(The helper warmed up before it said it was ready.)"""

    def diagnostics(self) -> dict[str, object]:
        return {"engine": "meiki", "requested": CUDA, "device": self.device,
                "fallback_reason": self.fallback_reason, "providers": [], "warm": True}

    def read(
        self,
        frame: np.ndarray,
        region: Region | None = None,
        min_conf: float = 0.0,
        fine: bool = False,
    ) -> OcrResult:
        fh, fw = frame.shape[:2]
        if region is not None:
            x0, y0, x1, y1 = region.to_pixels(fw, fh)
            img = frame[y0:y1, x0:x1]
        else:
            x0, y0, img = 0, 0, frame
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return OcrResult([], 0.0)
        t0 = time.perf_counter()
        with self._lock:
            mem = self._memory(h * w * 3)
            np.ndarray((h, w, 3), dtype=np.uint8, buffer=mem.buf)[:] = img[:, :, :3]
            self._send({"op": "read", "shm": mem.name, "h": h, "w": w})
            got = self._receive(READ_TIMEOUT_S)
        if not got.get("ok"):
            raise RuntimeError(got.get("error") or "the CUDA reader failed")
        lines = []
        for text, conf, bx, by, bw, bh, chars in got["lines"]:
            if conf < min_conf:
                continue
            lines.append(OcrLine(
                text=text, box=Region((x0 + bx) / fw, (y0 + by) / fh, bw / fw, bh / fh),
                conf=float(conf),
                chars=[OcrChar(c, (x0 + a) / fw, (x0 + b) / fw) for c, a, b in chars] or None,
            ))  # fmt: skip
        return OcrResult(lines, (time.perf_counter() - t0) * 1000.0)

    def close(self) -> None:
        proc, self._proc = getattr(self, "_proc", None), None
        if proc is not None and proc.poll() is None:
            try:
                self._send_to(proc, {"op": "quit"})
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001 - it goes either way
                proc.kill()
        if self._mem is not None:
            self._mem.close()
            self._mem.unlink()
            self._mem = None

    def __del__(self) -> None:  # (a HUD that stopped without closing it)
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- plumbing ---------------------------------------------------------------------------
    def _memory(self, size: int):  # noqa: ANN202
        if self._mem is None or self._mem.size < size:
            if self._mem is not None:
                self._mem.close()
                self._mem.unlink()
            self._mem = self._shm_cls(create=True, size=max(size, 1920 * 1088 * 3))
        return self._mem

    def _send(self, msg: dict) -> None:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("the CUDA reader stopped")
        self._send_to(self._proc, msg)

    @staticmethod
    def _send_to(proc: subprocess.Popen, msg: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(msg).encode("utf-8") + b"\n")
        proc.stdin.flush()

    def _receive(self, timeout: float) -> dict[str, Any]:
        """One reply; the helper is stopped if it says nothing in time (or stopped)."""
        proc = self._proc
        if proc is None:
            raise RuntimeError("the CUDA reader stopped")
        assert proc.stdout is not None
        box: list[bytes] = []
        reader = threading.Thread(target=lambda: box.append(proc.stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout)
        if not box or not box[0]:
            proc.kill()
            self._proc = None
            raise RuntimeError("the CUDA reader stopped answering")
        return json.loads(box[0])


if __name__ == "__main__":
    raise SystemExit(serve(Path(sys.argv[1]), Path(sys.argv[2])))
