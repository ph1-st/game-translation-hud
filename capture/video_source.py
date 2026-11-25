"""Frames of a video file for the offline capture tool: ``Frame(ts, index, image)`` with a
BGR image, at a chosen sample rate. (The live window's frames come from
hud.screen_source.)"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    ts: float  # seconds since start of the source
    index: int  # source frame index (-1: none, e.g. the end of a capture)
    image: np.ndarray | None  # BGR


class VideoSource:
    """Iterate over a video file at ``sample_fps`` frames per second.

    Frames in between are skipped with ``grab()`` so decoding stays cheap.
    """

    def __init__(
        self,
        path: str | Path,
        sample_fps: float = 4.0,
        start: float = 0.0,
        end: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.frame_count else None
        self.step = max(1, int(round(self.fps / sample_fps)))
        self.sample_fps = self.fps / self.step
        self.start = start
        self.end = end
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    def __iter__(self) -> Iterator[Frame]:
        cap = self.cap
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        while True:
            ok = cap.grab()
            if not ok:
                return
            ts = idx / self.fps
            if self.end is not None and ts > self.end:
                return
            if idx % self.step == 0:
                ok, img = cap.retrieve()
                if not ok:
                    return
                yield Frame(ts=ts, index=idx, image=img)
            idx += 1

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def format_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
