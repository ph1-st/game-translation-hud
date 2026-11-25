"""Play a recording in a plain window at real-time pace, as a stand-in for the game, with
keyboard navigation so scenes can be revisited while the HUD runs on top.

    uv run python tools/play_video.py data/footage/azur-promilia/cbt_stream_1080p.mp4 \
        --title "Azur Promilia (test video)" --start 2790 --width 1600

Keys (the player window must have focus):
    space        pause / resume
    left/right   seek -5 s / +5 s          up/down   seek +30 s / -30 s
    [ / ]        seek -2 min / +2 min      , / .     step one frame back / forward (paused)
    s            screenshot of the window INCLUDING the overlay -> ~/Downloads/hud_<timecode>.png
                 (the HUD takes it; needs the HUD running with the same --control file)
    esc / q      quit
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2

KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 2424832, 2555904, 2490368, 2621440


def fmt(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--title", default="Azur Promilia (test video)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = never)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--x", type=int, default=40)
    ap.add_argument("--y", type=int, default=40)
    ap.add_argument("--topmost", action="store_true", help="keep the window above others")
    ap.add_argument("--no-timecode", action="store_true", help="do not draw the timecode")
    ap.add_argument("--control", default="data/out/hud_control.json", help="HUD control file")
    args = ap.parse_args()
    snap_id = 0

    def control(paused: bool, pos: float) -> None:
        ctl = {"paused": paused, "snap_id": snap_id, "timecode": fmt(pos)}
        tmp = args.control + ".tmp"
        Path(args.control).parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ctl, f)
        os.replace(tmp, args.control)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    w = args.width
    h = int(w * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(args.title, w, h)
    cv2.moveWindow(args.title, args.x, args.y)
    if args.topmost:
        cv2.setWindowProperty(args.title, cv2.WND_PROP_TOPMOST, 1)
    control(False, args.start)

    pos = args.start
    paused = False
    frame = None
    started = time.perf_counter()

    def seek(to: float) -> None:
        nonlocal pos, clock0, pos0
        pos = max(0.0, min(to, max(0.0, total - 0.5)))
        cap.set(cv2.CAP_PROP_POS_MSEC, pos * 1000.0)
        clock0, pos0 = time.perf_counter(), pos

    clock0, pos0 = time.perf_counter(), pos
    seek(pos)
    while True:
        if not paused:
            target = pos0 + (time.perf_counter() - clock0)
            # decode until the frame that matches the wall clock
            while pos < target:
                ok = cap.grab()
                if not ok:
                    paused = True
                    break
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, img = cap.retrieve()
            if ok:
                frame = img
        if frame is not None:
            shown = frame if args.no_timecode else frame.copy()
            if not args.no_timecode:
                label = fmt(pos) + ("  ||" if paused else "")
                cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)
            cv2.imshow(args.title, shown)
        key = cv2.waitKeyEx(8 if not paused else 30)
        if key in (27, ord("q")):
            break
        if key == 32:
            paused = not paused
            if not paused:
                clock0, pos0 = time.perf_counter(), pos
            control(paused, pos)
        elif key == KEY_LEFT:
            seek(pos - 5)
        elif key == KEY_RIGHT:
            seek(pos + 5)
        elif key == KEY_UP:
            seek(pos + 30)
        elif key == KEY_DOWN:
            seek(pos - 30)
        elif key == ord("["):
            seek(pos - 120)
        elif key == ord("]"):
            seek(pos + 120)
        elif key in (ord(","), ord(".")) and paused:
            step = -1 if key == ord(",") else 1
            seek(pos + step / fps if step > 0 else pos - 2 / fps)
            ok = cap.grab()
            if ok:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                ok, img = cap.retrieve()
                if ok:
                    frame = img
        elif key == ord("s"):
            snap_id += 1
            control(paused, pos)  # the HUD saves ~/Downloads/hud_<timecode>.png with the overlay
            print("screenshot requested at", fmt(pos))
        if args.duration and time.perf_counter() - started > args.duration:
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
