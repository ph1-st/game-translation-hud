"""OCR engine wrapper (RapidOCR, PP-OCRv4) with relative-coordinate output.

The same wrapper is used by the capture tool and the HUD, so that recognition errors are
systematic: a line misread the same way at capture time and at run time still matches.

Devices: DirectML (any DirectX 12 GPU, Windows), CUDA (NVIDIA, when onnxruntime-gpu is
installed), OpenVINO (Intel, when the openvino package is installed) and the CPU, which
every system has. "auto" takes the first of :func:`preferred_devices`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from capture.profile import Region

log = logging.getLogger(__name__)

AUTO, CPU, DML, CUDA, OPENVINO = "auto", "cpu", "dml", "cuda", "openvino"
DEVICES = (AUTO, CPU, DML, CUDA, OPENVINO)
# the ONNX Runtime provider a GPU device needs (OpenVINO runs without ONNX Runtime)
PROVIDERS = {DML: "DmlExecutionProvider", CUDA: "CUDAExecutionProvider"}
# The HUD's frames are shrunk to about this many rows for detection (a band cut from a
# frame as much as the whole frame: text keeps one size), whatever the screen's resolution:
# the detector still finds nearly every line the database can match (98 in 100 on 1080p
# footage) at about half the cost of a whole 1440p frame, and recognition still reads the
# original pixels. Tiny text (under ~14 px at 1080p) may go unseen.
SCREEN_DETECT_HEIGHT = 800
# On a GPU the recognizer's input shapes repeat: a batch's width is rounded up to this
# many pixels and the batch is always full (DirectML builds its program again whenever the
# shape changes: with a full batch a dialogue band reads in ~95 ms instead of ~150)
REC_WIDTH_STEP = 160
REC_MEMORY = 512  # crops the recognizer read, kept with their reading (:class:`_Remembered`)
UNAVAILABLE = {
    DML: "DirectML is not available in this installation: update the graphics driver "
         "(DirectX 12), or reinstall the app.",
    CUDA: "CUDA is not available: install the GPU package (onnxruntime-gpu) and NVIDIA's "
          "CUDA 12 runtime.",
    OPENVINO: "OpenVINO is not installed (pip install openvino).",
}  # fmt: skip


def available_providers() -> list[str]:
    """ONNX Runtime's execution providers in this installation."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - diagnostics never stop OCR startup
        return []


def _openvino_installed() -> bool:
    return importlib.util.find_spec("openvino") is not None


def preferred_devices() -> list[str]:
    """The devices this installation can read with, best first (what "auto" tries)."""
    providers = available_providers()
    if sys.platform == "win32":
        out = [DML] if PROVIDERS[DML] in providers else []
    else:
        out = [CUDA] if PROVIDERS[CUDA] in providers else []
    if _openvino_installed():
        out.append(OPENVINO)
    return [*out, CPU]


def device_available(device: str) -> bool:
    if device == CPU:
        return True
    if device == OPENVINO:
        return _openvino_installed()
    return PROVIDERS.get(device, "") in available_providers()


def cpu_threads() -> int:
    """Threads for reading on the CPU: about half the physical cores (the game runs on the
    rest)."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    except ImportError:  # pragma: no cover - psutil is a dependency
        cores = os.cpu_count() or 2
    return max(1, cores // 2)


def _quiet_runtime() -> None:
    """No usage data sent by ONNX Runtime (its Windows builds have it on), and no warning
    from rapidocr for every read that finds no text."""
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    try:
        import onnxruntime as ort

        ort.disable_telemetry_events()
    except Exception:  # noqa: BLE001 - older runtimes have no switch
        pass


@dataclass
class OcrChar:
    """One recognized character and the columns it occupies (relative to the full frame).

    The positions come from the recognizer's CTC alignment: each character's centre is the
    output step where it was emitted, its cell the average character pitch around it. They
    tell which part of a detector box is which character (a leading icon read as 日, the
    digits of a counter), not the exact glyph edges."""

    ch: str
    x0: float
    x1: float


@dataclass
class OcrLine:
    text: str
    box: Region  # relative to the full frame
    conf: float
    chars: list[OcrChar] | None = None  # per character, left to right; None if unknown

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "box": self.box.as_list(),
            "conf": round(self.conf, 3),
        }
        if self.chars is not None:
            d["chars"] = [[c.ch, round(c.x0, 5), round(c.x1, 5)] for c in self.chars]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OcrLine":
        x, y, w, h = d["box"]
        chars = d.get("chars")
        return cls(
            text=str(d["text"]),
            box=Region(float(x), float(y), float(w), float(h)),
            conf=float(d.get("conf", 1.0)),
            chars=None
            if chars is None
            else [OcrChar(str(c), float(a), float(b)) for c, a, b in chars],
        )


@dataclass
class OcrResult:
    lines: list[OcrLine]
    elapsed_ms: float

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


# onnxruntime-directml crashes when two sessions run inference concurrently from different
# threads (access violation inside Run), so DirectML inference is serialized process-wide.
_DML_LOCK = threading.Lock()


class OcrEngine:
    """Thin wrapper around RapidOCR.

    device: "auto" or one of :data:`DEVICES`; a device this installation lacks falls back
    to the CPU (``fallback_reason`` says why). threads: for the CPU's share of the work
    (None: half the physical cores, :func:`cpu_threads`).
    use_cls: the text-direction classifier is useless for game UI (always horizontal) and
    costs up to 200 ms on busy frames, so it is off by default.

    det_limit_type must stay "max": rapidocr's default "min" upscales the *shorter* side to
    det_limit_side_len, which turns a thin dialogue crop into a huge image (500 ms+ on CPU).
    Detection runs on the resized image; recognition crops come from the original pixels.
    detect_height: instead of the limit, every frame is shrunk (never enlarged) to about
    this many rows, and a region of it as much (:data:`SCREEN_DETECT_HEIGHT`).
    """

    def __init__(
        self,
        lang: str = "japan",
        device: str = AUTO,
        rec_model_type: str = "mobile",
        rec_version: str = "PP-OCRv4",
        det_model_type: str = "mobile",
        det_version: str = "PP-OCRv4",
        text_score: float = 0.5,
        det_limit_side_len: int = 1280,
        det_limit_type: str = "max",
        threads: int | None = None,
        use_cls: bool = False,
        detect_height: int | None = None,
    ) -> None:
        from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

        _quiet_runtime()
        if device not in DEVICES:
            raise ValueError(f"unknown OCR device {device!r} (one of {', '.join(DEVICES)})")
        self.requested_device = device
        self.fallback_reason = ""
        self.available_providers = available_providers()
        if device == AUTO:
            device = preferred_devices()[0]
        elif not device_available(device):
            self.fallback_reason = UNAVAILABLE.get(device, "")
            device = CPU
        threads = cpu_threads() if threads is None else threads
        params: dict[str, Any] = {
            "Global.use_cls": use_cls,
            "Rec.lang_type": LangRec(lang),
            "Rec.ocr_version": OCRVersion(rec_version),
            "Rec.model_type": ModelType(rec_model_type),
            "Det.ocr_version": OCRVersion(det_version),
            "Det.model_type": ModelType(det_model_type),
            "Det.limit_side_len": det_limit_side_len,
            "Det.limit_type": det_limit_type,
            "Global.text_score": text_score,
            "Global.log_level": "error",  # (no warning for every read that finds no text)
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        }
        if device == DML:
            params["EngineConfig.onnxruntime.use_dml"] = True
        elif device == CUDA:
            params["EngineConfig.onnxruntime.use_cuda"] = True
            # (the default, EXHAUSTIVE, benchmarks again for every new crop width)
            params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "HEURISTIC"
        elif device == OPENVINO:
            for part in ("Det", "Rec", "Cls"):
                params[f"{part}.engine_type"] = EngineType.OPENVINO
            params["EngineConfig.openvino.inference_num_threads"] = threads
        self.device = device
        self.use_cls = use_cls
        self.detect_height = detect_height
        self.engine = RapidOCR(params=params)
        self._prepare = self._tune(fixed_shapes=device in (DML, CUDA))
        self._warm = False

    def _tune(self, fixed_shapes: bool) -> "_Float32DetInput":
        """The detector's input made in float32 (rapidocr makes it in float64: ~50 ms of a
        1080p frame); on a GPU, recognizer input shapes that repeat
        (:class:`_FixedShapes`); crops read before not read again (:class:`_Remembered`).
        Both models are loaded now for it."""
        det = self.engine._load_det_model()  # noqa: SLF001 - rapidocr loads them lazily
        rec = self.engine._load_rec_model()  # noqa: SLF001
        prepare = _Float32DetInput(det.limit_side_len, det.limit_type, det.mean, det.std)
        det.get_preprocess = lambda: prepare
        self.engine.text_rec = _Remembered(_FixedShapes(rec) if fixed_shapes else rec)
        return prepare

    @classmethod
    def from_profile(cls, ocr_cfg: dict[str, Any], device: str | None = None) -> "OcrEngine":
        """Build an engine from the ``ocr:`` section of a game profile."""
        cfg = dict(ocr_cfg)
        if device is not None:
            cfg["device"] = device
        return cls(**cfg)

    @classmethod
    def for_screen(cls, ocr_cfg: dict[str, Any], device: str | None = None) -> "OcrEngine":
        """An engine for whole game frames (:data:`SCREEN_DETECT_HEIGHT`)."""
        return cls.from_profile({**ocr_cfg, "detect_height": SCREEN_DETECT_HEIGHT}, device)

    def warm_up(self) -> None:
        """Load the detector and the recognizer now (with text to read: a blank picture
        never reaches the recognizer), and tell which device they really run on."""
        if self._warm:
            return
        import cv2

        img = np.full((64, 320, 3), 255, dtype=np.uint8)
        cv2.putText(img, "READY 123", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
        self._run(img, {"use_det": True, "use_cls": self.use_cls, "use_rec": True})
        self._warm = True
        provider = PROVIDERS.get(self.device)
        if provider is None:
            return
        missing = [name for name, attached in self._session_providers().items()
                   if provider not in attached]  # fmt: skip
        if missing:
            log.warning("%s did not start: the text reader runs on the CPU (slower)", provider)
            self.fallback_reason = (
                f"{provider} was installed, but ONNX Runtime did not attach it to the OCR "
                f"model ({', '.join(missing)}). Update the graphics driver and try the GPU "
                "again."
            )
            self.device = CPU

    def _session_providers(self) -> dict[str, list[str]]:
        """The providers each model's ONNX Runtime session really runs on (empty when
        rapidocr hides its sessions)."""
        out: dict[str, list[str]] = {}
        for name in ("text_det", "text_rec"):
            try:
                out[name] = list(getattr(self.engine, name).session.session.get_providers())
            except AttributeError:
                continue
        return out

    def _run(self, img: np.ndarray, kw: dict[str, Any]) -> Any:
        if self.device == DML:
            with _DML_LOCK:  # (another engine may be reading: never two at once)
                return self.engine(img, **kw)
        return self.engine(img, **kw)

    def diagnostics(self) -> dict[str, object]:
        """Small, serializable description for the in-app diagnostics panel."""
        return {
            "requested": self.requested_device,
            "device": self.device,
            "fallback_reason": self.fallback_reason,
            "providers": list(self.available_providers),
            "warm": self._warm,
        }

    def read(
        self,
        frame: np.ndarray,
        region: Region | None = None,
        min_conf: float = 0.0,
        fine: bool = False,
    ) -> OcrResult:
        """OCR the whole frame or a region of it. Boxes come back relative to the full frame.
        *fine*: detected at full resolution even by an engine that shrinks frames
        (:attr:`detect_height`): the dialogue and the names, whose rows' exact places set
        the layout of their translation."""
        fh, fw = frame.shape[:2]
        if region is not None:
            x0, y0, x1, y1 = region.to_pixels(fw, fh)
            img = frame[y0:y1, x0:x1]
        else:
            x0, y0 = 0, 0
            img = frame
        if img.size == 0:
            return OcrResult([], 0.0)

        # rapidocr remembers per-call use_* flags on the engine object: always be explicit
        kw = {
            "use_det": True,
            "use_cls": self.use_cls,
            "use_rec": True,
            "return_word_box": True,
            "return_single_char_box": True,
        }
        if self.detect_height:  # (the frame's scale: a band shrinks as much as its frame)
            self._prepare.scale = 1.0 if fine else min(1.0, self.detect_height / fh)
        t0 = time.perf_counter()
        out = self._run(img, kw)
        elapsed = (time.perf_counter() - t0) * 1000.0

        lines: list[OcrLine] = []
        if out is not None and out.boxes is not None and out.txts:
            cells = _char_cells(out.txts, getattr(out, "word_results", None))
            for box, txt, score, chars in zip(out.boxes, out.txts, out.scores, cells, strict=False):
                if score < min_conf or not txt.strip():
                    continue
                pts = np.asarray(box, dtype=np.float32)
                bx0, by0 = pts.min(axis=0)
                bx1, by1 = pts.max(axis=0)
                rel = Region(
                    x=float(x0 + bx0) / fw,
                    y=float(y0 + by0) / fh,
                    w=float(bx1 - bx0) / fw,
                    h=float(by1 - by0) / fh,
                )
                rel_chars = None
                if chars is not None:
                    rel_chars = [OcrChar(c, (x0 + a) / fw, (x0 + b) / fw) for c, a, b in chars]
                lines.append(OcrLine(text=txt, box=rel, conf=float(score), chars=rel_chars))
        lines.sort(key=lambda line: (round(line.box.y, 2), line.box.x))
        return OcrResult(lines, elapsed)


def recognize_boxes(
    engine: OcrEngine, frame: np.ndarray, boxes: list[Region]
) -> list[tuple[str, float] | None]:
    """*engine*'s recognizer alone on lines of *frame* (no detection: the boxes are given,
    grown by a little), in one batch: (text, confidence) per box, or None. A second reading
    of lines another reader read (capture/meiki.py: its readings that match nothing)."""
    from rapidocr.ch_ppocr_rec.typings import TextRecInput

    fh, fw = frame.shape[:2]
    crops, where = [], []
    for i, box in enumerate(boxes):
        pad = box.h * fh * 0.15
        x0, y0 = max(0, int(box.x * fw - pad)), max(0, int(box.y * fh - pad))
        x1 = min(fw, int((box.x + box.w) * fw + pad))
        y1 = min(fh, int((box.y + box.h) * fh + pad))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            crops.append(np.ascontiguousarray(frame[y0:y1, x0:x1]))
            where.append(i)
    out: list[tuple[str, float] | None] = [None] * len(boxes)
    if not crops:
        return out
    rec = engine.engine.text_rec
    args = TextRecInput(img=crops, return_word_box=True)
    if engine.device == DML:
        with _DML_LOCK:
            got = rec(args)
    else:
        got = rec(args)
    for i, txt, score in zip(where, got.txts, got.scores, strict=True):
        out[i] = (str(txt), float(score))
    return out


class _Float32DetInput:
    """rapidocr's detector input (resized to a multiple of 32, normalized, channels
    first), the same numbers computed in float32 throughout. *scale*, when set, replaces
    rapidocr's size limit: the image is resized by it (each side to a multiple of 32)."""

    def __init__(self, limit_side_len: int, limit_type: str, mean: Any, std: Any) -> None:
        from rapidocr.ch_ppocr_det.utils import DetPreProcess

        self._resize = DetPreProcess(limit_side_len, limit_type, mean, std).resize
        self.scale: float | None = None
        mean = np.asarray(mean if mean is not None else [0.5] * 3, dtype=np.float64)
        std = np.asarray(std if std is not None else [0.5] * 3, dtype=np.float64)
        self._mult = (1 / 255.0 / std).astype(np.float32)
        self._shift = (-mean / std).astype(np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray | None:
        import cv2

        if self.scale is None:
            resized = self._resize(img)
            if resized is None:
                return None
        else:
            h, w = img.shape[:2]
            size = (max(32, round(w * self.scale / 32) * 32),
                    max(32, round(h * self.scale / 32) * 32))  # fmt: skip
            # (bilinear, cv2's default: averaging pixels, INTER_AREA, loses short labels: 中)
            resized = img if size == (w, h) else cv2.resize(img, size)
        x = resized.astype(np.float32)
        x *= self._mult
        x += self._shift
        return np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]


class _FixedShapes:
    """rapidocr's recognizer with input shapes that repeat: each batch's width rounded up
    to :data:`REC_WIDTH_STEP` and the batch filled up to the recognizer's batch size (the
    filling is read and thrown away). Everything else is the recognizer's own."""

    def __init__(self, rec: Any) -> None:
        self.rec = rec

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rec, name)

    def __call__(self, args: Any) -> Any:
        from rapidocr.ch_ppocr_rec.typings import TextRecOutput

        rec = self.rec
        t0 = time.perf_counter()
        imgs = [args.img] if isinstance(args.img, np.ndarray) else list(args.img)
        order = np.argsort([im.shape[1] / float(im.shape[0]) for im in imgs])
        out: list[tuple] = [(("", 0.0), None)] * len(imgs)
        _c, height, width = rec.rec_image_shape[:3]
        for start in range(0, len(imgs), rec.rec_batch_num):
            idx = order[start:start + rec.rec_batch_num]
            ratios = [imgs[i].shape[1] / float(imgs[i].shape[0]) for i in idx]
            padded = math.ceil(height * max(width / height, *ratios) / REC_WIDTH_STEP)
            ratio = padded * REC_WIDTH_STEP / height
            batch = [rec.resize_norm_img(imgs[i], ratio)[np.newaxis] for i in idx]
            batch += [np.zeros_like(batch[0])] * (rec.rec_batch_num - len(batch))
            preds = rec.session(np.concatenate(batch).astype(np.float32))[: len(idx)]
            lines, cells = rec.postprocess_op(preds, args.return_word_box,
                                              wh_ratio_list=ratios, max_wh_ratio=ratio)
            for k, i in enumerate(idx):
                out[i] = (lines[k], cells[k] if args.return_word_box else None)
        txts = tuple(line[0] for line, _w in out)
        scores = tuple(line[1] for line, _w in out)
        words = tuple(w for _line, w in out)
        return TextRecOutput(imgs, txts, scores, words, time.perf_counter() - t0)


class _Remembered:
    """The recognizer, reading only crops it has not read before pixel for pixel: a label
    that stayed put while the scene around it moved, a list scrolled by whole pixels. The
    same pixels read the same, so their reading (text, confidence, character cells) comes
    from memory; the :data:`REC_MEMORY` most recent are kept."""

    def __init__(self, rec: Any) -> None:
        self.rec = rec
        self._memory: OrderedDict[bytes, tuple[str, float, Any]] = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rec, name)

    def __call__(self, args: Any) -> Any:
        from rapidocr.ch_ppocr_rec.typings import TextRecInput, TextRecOutput

        if not args.return_word_box:  # (the warm-up: nothing worth keeping)
            return self.rec(args)
        t0 = time.perf_counter()
        imgs = [args.img] if isinstance(args.img, np.ndarray) else list(args.img)
        keys = [_crop_key(im) for im in imgs]
        new = list(dict.fromkeys(k for k in keys if k not in self._memory))
        if new:
            first = {k: imgs[keys.index(k)] for k in new}
            got = self.rec(TextRecInput(img=list(first.values()), return_word_box=True))
            for k, txt, score, words in zip(new, got.txts, got.scores, got.word_results,
                                            strict=True):  # fmt: skip
                self._memory[k] = (txt, float(score), words)
        read = []
        for k in keys:
            self._memory.move_to_end(k)
            read.append(self._memory[k])
        while len(self._memory) > REC_MEMORY:
            self._memory.popitem(last=False)
        return TextRecOutput(
            imgs, tuple(r[0] for r in read), tuple(r[1] for r in read),
            tuple(r[2] for r in read), time.perf_counter() - t0,
        )  # fmt: skip


def _crop_key(img: np.ndarray) -> bytes:
    """A crop's identity: its shape and every pixel."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(img.shape, np.int32).tobytes())
    h.update(np.ascontiguousarray(img).data)
    return h.digest()


def _char_cells(txts, word_results) -> list[list[tuple[str, float, float]] | None]:
    """Per line, the recognizer's character cells as (char, x0, x1) in image pixels, or
    None where they are not known.

    rapidocr drops lines without character boxes from its word results and then indexes
    them by line, so the lists can slip out of step: a line only gets cells whose
    characters spell exactly its text."""
    out: list[list[tuple[str, float, float]] | None] = [None] * len(txts)
    if not word_results:
        return out
    groups = []
    for wl in word_results:
        try:
            cells = []
            for ch, _score, quad in wl:
                if quad is None:
                    continue
                q = np.asarray(quad, dtype=np.float32)
                cells.append((str(ch), float(q[:, 0].min()), float(q[:, 0].max())))
        except (TypeError, ValueError):
            continue
        cells.sort(key=lambda c: c[1])
        groups.append(cells)
    j = 0
    for i, txt in enumerate(txts):
        want = "".join(str(txt).split())
        for k in range(j, len(groups)):
            if "".join(c[0] for c in groups[k]) == want:
                out[i] = groups[k]
                j = k + 1
                break
    return out


def merge_lines(lines: list[OcrLine], same_row_tol: float = 0.5) -> list[OcrLine]:
    """Join boxes that sit on the same text row (left to right), keep rows top to bottom.

    same_row_tol is the allowed vertical centre offset as a fraction of the line height.
    """
    if not lines:
        return []
    rows: list[list[OcrLine]] = []
    for ln in sorted(lines, key=lambda line: line.box.y):
        cy = ln.box.y + ln.box.h / 2
        for row in rows:
            r0 = row[0]
            rcy = r0.box.y + r0.box.h / 2
            if abs(cy - rcy) <= same_row_tol * max(r0.box.h, ln.box.h):
                row.append(ln)
                break
        else:
            rows.append([ln])
    merged: list[OcrLine] = []
    for row in rows:
        row.sort(key=lambda line: line.box.x)
        x0 = min(line.box.x for line in row)
        y0 = min(line.box.y for line in row)
        x1 = max(line.box.x2 for line in row)
        y1 = max(line.box.y2 for line in row)
        text = "".join(line.text for line in row)
        conf = min(line.conf for line in row)
        chars = None
        if all(line.chars is not None for line in row):
            chars = [c for line in row for c in line.chars or []]
        merged.append(OcrLine(text, Region(x0, y0, x1 - x0, y1 - y0), conf, chars))
    return merged


def _hangul(ch: str) -> bool:
    return "\uac00" <= ch <= "\ud7a3"


def join_rows(texts: list[str]) -> str:
    """Rows of one block as one text: Japanese and Chinese have no spaces between words,
    so rows join directly; a row break between two Korean words is a space."""
    out = ""
    for t in texts:
        t = t.strip()
        if out and t and _hangul(out[-1]) and _hangul(t[0]):
            out += " "
        out += t
    return out


def block_text(lines: list[OcrLine]) -> tuple[str, float, Region | None]:
    """Collapse a multi-line block (e.g. a dialogue box) into one string.
    Returns (text, min_conf, bounding box)."""
    rows = merge_lines(lines)
    if not rows:
        return "", 0.0, None
    text = join_rows([r.text for r in rows])
    conf = min(r.conf for r in rows)
    x0 = min(r.box.x for r in rows)
    y0 = min(r.box.y for r in rows)
    x1 = max(r.box.x2 for r in rows)
    y1 = max(r.box.y2 for r in rows)
    return text, conf, Region(x0, y0, x1 - x0, y1 - y0)
