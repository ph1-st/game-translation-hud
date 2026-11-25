"""MeikiOCR: a small text detector and recognizer made for Japanese game text, much faster
than a general OCR (models by rtr46 on Hugging Face, LGPL-3.0; the pipeline follows
meikiocr 0.3.5, Apache-2.0).

The detector always reads a 960x544 picture: a frame (or a region of it) is shrunk to fit,
never enlarged, and padded. Each line it finds is shrunk to 32 rows and read by the
recognizer, which gives every character its own box. Input shapes never change (the
recognizer's batch is always full), so a GPU builds its programs once.

It reads Japanese only, horizontal lines; the HUD only needs a line close enough to match
a database line, so speed beats the last bit of accuracy.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from capture.ocr import (
    _DML_LOCK,
    AUTO,
    CPU,
    DEVICES,
    DML,
    PROVIDERS,
    UNAVAILABLE,
    OcrChar,
    OcrLine,
    OcrResult,
    _quiet_runtime,
    available_providers,
    cpu_threads,
    device_available,
    preferred_devices,
)
from capture.profile import Region

log = logging.getLogger(__name__)

DET_MODEL = "meiki.text.detect.v0.1.960x544.onnx"
REC_MODEL = "meiki.text.rec.v0.960x32.onnx"
MODEL_URLS = {
    DET_MODEL: f"https://huggingface.co/rtr46/meiki.text.detect.v0/resolve/main/{DET_MODEL}",
    REC_MODEL: f"https://huggingface.co/rtr46/meiki.txt.recognition.v0/resolve/main/{REC_MODEL}",
}
DET_W, DET_H = 960, 544
REC_W, REC_H = 960, 32
REC_BATCH = 8  # lines per recognizer run (always full: one shape)
DET_THRESHOLD = 0.5  # boxes the detector is at least this sure of
CHAR_THRESHOLD = 0.1  # characters the recognizer is at least this sure of
CHAR_OVERLAP = 0.3  # two characters sharing more of their width than this: one is dropped


def models_present(folder: Path) -> bool:
    return all((folder / name).is_file() for name in (DET_MODEL, REC_MODEL))


class MeikiEngine:
    """The same interface as capture.ocr.OcrEngine (read, warm_up, device, diagnostics)."""

    lang = "japan"

    def __init__(self, folder: Path, device: str = AUTO, threads: int | None = None) -> None:
        import onnxruntime as ort

        _quiet_runtime()
        if device not in DEVICES:
            raise ValueError(f"unknown OCR device {device!r} (one of {', '.join(DEVICES)})")
        if not models_present(folder):
            raise FileNotFoundError(f"the MeikiOCR models are not in {folder}")
        self.requested_device = device
        self.fallback_reason = ""
        self.available_providers = available_providers()
        if device == AUTO:
            device = preferred_devices()[0]
        if device not in (CPU, *PROVIDERS):  # (OpenVINO: not for these models)
            device = CPU
        if device != CPU and not device_available(device):
            self.fallback_reason = UNAVAILABLE.get(device, "")
            device = CPU
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = cpu_threads() if threads is None else threads
        # (no busy-waiting threads: the game keeps the CPU)
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        providers = [PROVIDERS[device], "CPUExecutionProvider"] if device != CPU else [
            "CPUExecutionProvider"
        ]
        if device == DML:  # (DirectML wants these)
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.det = ort.InferenceSession(str(folder / DET_MODEL), opts, providers=providers)
        self.rec = ort.InferenceSession(str(folder / REC_MODEL), opts, providers=providers)
        self.device = device
        self._det_in = [i.name for i in self.det.get_inputs()]
        self._warm = False

    # ---- the interface -------------------------------------------------------------------
    def warm_up(self) -> None:
        """Load both models now (a line to read reaches the recognizer), and tell which
        device they really run on."""
        if self._warm:
            return
        import cv2

        img = np.full((96, 480, 3), 245, dtype=np.uint8)
        cv2.putText(img, "READY 123", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3)
        self.read(img)
        self._warm = True
        provider = PROVIDERS.get(self.device)
        if provider is not None and not all(
            provider in s.get_providers() for s in (self.det, self.rec)
        ):
            log.warning("%s did not start: MeikiOCR runs on the CPU (slower)", provider)
            self.fallback_reason = f"{provider} did not attach to the OCR models."
            self.device = CPU

    def diagnostics(self) -> dict[str, object]:
        return {
            "engine": "meiki",
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
        """OCR the whole frame or a region of it; boxes relative to the full frame. (*fine*
        is OcrEngine's: here every read is shrunk to the detector's picture, a region less
        than the whole frame.)"""
        fh, fw = frame.shape[:2]
        if region is not None:
            x0, y0, x1, y1 = region.to_pixels(fw, fh)
            img = frame[y0:y1, x0:x1]
        else:
            x0, y0, img = 0, 0, frame
        if img.size == 0:
            return OcrResult([], 0.0)
        t0 = time.perf_counter()
        boxes = self._detect(img)
        rows = self._recognize(img, boxes) if boxes else []
        elapsed = (time.perf_counter() - t0) * 1000.0
        lines: list[OcrLine] = []
        for (bx0, by0, bx1, by1), chars in zip(boxes, rows, strict=True):
            if not chars:
                continue
            text = "".join(c for c, _a, _b, _s in chars)
            conf = float(np.mean([s for _c, _a, _b, s in chars]))
            if conf < min_conf or not text.strip():
                continue
            lines.append(OcrLine(
                text=text,
                box=Region((x0 + bx0) / fw, (y0 + by0) / fh, (bx1 - bx0) / fw, (by1 - by0) / fh),
                conf=conf,
                chars=[OcrChar(c, (x0 + a) / fw, (x0 + b) / fw) for c, a, b, _s in chars],
            ))  # fmt: skip
        lines.sort(key=lambda ln: (round(ln.box.y, 2), ln.box.x))
        return OcrResult(lines, elapsed)

    # ---- the pipeline ----------------------------------------------------------------------
    def _run(self, session: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.device == DML:
            with _DML_LOCK:  # (another engine may be reading: never two at once)
                return session.run(None, feed)
        return session.run(None, feed)

    def _detect(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Line boxes (x0, y0, x1, y1) in *img*'s pixels, top to bottom."""
        import cv2

        h, w = img.shape[:2]
        scale = min(DET_W / w, DET_H / h, 1.0)
        rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
        small = img if (rw, rh) == (w, h) else cv2.resize(img, (rw, rh))
        tensor = np.zeros((1, 3, DET_H, DET_W), dtype=np.float32)
        tensor[0, :, :rh, :rw] = small[:, :, :3].transpose(2, 0, 1)
        tensor *= 1.0 / 255.0
        sizes = np.array([[DET_W / scale, DET_H / scale]], dtype=np.int64)
        _labels, boxes, scores = self._run(self.det, {self._det_in[0]: tensor,
                                                      self._det_in[1]: sizes})  # fmt: skip
        keep = boxes[0][scores[0] > DET_THRESHOLD]
        if not len(keep):
            return []
        keep = np.clip(keep, 0, [w, h, w, h]).astype(np.int32)
        out = [tuple(int(v) for v in b) for b in keep if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
        out.sort(key=lambda b: (b[1], b[0]))
        return out

    def _recognize(
        self, img: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> list[list[tuple[str, float, float, float]]]:
        """Per box, its characters as (char, x0, x1, confidence) in *img*'s pixels, left to
        right. Every line is read as horizontal (game UI)."""
        import cv2

        tensors, widths = [], []
        for x0, y0, x1, y1 in boxes:
            crop = img[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            nh, nw = REC_H, int(round(cw * REC_H / ch))
            if nw > REC_W:
                nh, nw = max(1, int(round(REC_H * REC_W / nw))), REC_W
            line = np.zeros((REC_H, REC_W, 3), dtype=np.float32)
            line[:nh, :nw] = cv2.resize(crop, (max(1, nw), nh))[:, :, :3]
            tensors.append(line.transpose(2, 0, 1))
            widths.append(max(1, nw))
        sizes = np.array([[REC_W, REC_H]], dtype=np.int64)
        out: list[list[tuple[str, float, float, float]]] = []
        for start in range(0, len(tensors), REC_BATCH):
            part = tensors[start : start + REC_BATCH]
            batch = np.zeros((REC_BATCH, 3, REC_H, REC_W), dtype=np.float32)
            batch[: len(part)] = np.stack(part)
            batch *= 1.0 / 255.0
            labels, cboxes, scores = self._run(self.rec, {"images": batch,
                                                          "orig_target_sizes": sizes})  # fmt: skip
            for k in range(len(part)):
                i = start + k
                x0, _y0, x1, _y1 = boxes[i]
                out.append(_chars(labels[k], cboxes[k], scores[k], widths[i], x1 - x0, x0))
        return out


def _chars(labels, boxes, scores, effective_w: int, crop_w: int, left: int):  # noqa: ANN001, ANN202
    """One line's characters, overlapping guesses resolved (the surer one stays)."""
    cands = []
    for lbl, box, score in zip(labels, boxes, scores, strict=True):
        if score < CHAR_THRESHOLD or box[0] >= effective_w:
            continue
        a = left + float(min(box[0], effective_w)) / effective_w * crop_w
        b = left + float(min(box[2], effective_w)) / effective_w * crop_w
        cands.append((float(score), chr(int(lbl)), a, b))
    cands.sort(reverse=True)
    kept: list[tuple[float, str, float, float]] = []
    for score, ch, a, b in cands:
        if any(
            min(b, kb) - max(a, ka) > CHAR_OVERLAP * min(b - a + 1e-6, kb - ka + 1e-6)
            for _s, _c, ka, kb in kept
        ):
            continue
        kept.append((score, ch, a, b))
    kept.sort(key=lambda k: k[2])
    return [(ch, a, b, score) for score, ch, a, b in kept]
