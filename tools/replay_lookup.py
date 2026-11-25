"""Replay a recording against the database the way the HUD will: OCR the dialogue region at
the HUD's sample rate and look every reading up. Reports hit rate per attempt kind and, per
captured line, how long after the line first appeared the lookup first resolved it.

    uv run python tools/replay_lookup.py --game azur-promilia \
        --db data/db/azur-promilia.sqlite --video data/footage/azur-promilia/clip_0030.mp4 \
        --overrides games/azur-promilia/sources/RECORDING.yaml --fps 4 --player NAME
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.change import ChangeTracker  # noqa: E402
from capture.ocr import OcrEngine, block_text  # noqa: E402
from capture.profile import load_overrides, load_profile  # noqa: E402
from capture.video_source import VideoSource, format_ts  # noqa: E402
from db.lookup import Lookup  # noqa: E402
from db.text import cjk_count  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--overrides")
    ap.add_argument("--player")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    profile = load_profile(args.game)
    if args.overrides:
        profile = profile.with_overrides(load_overrides(args.overrides))
    region = profile.dialogue
    assert region is not None
    engine = OcrEngine.from_profile(profile.ocr, device=args.device)
    engine.warm_up()
    lookup = Lookup.from_db(args.db, player_name=args.player)
    det = ChangeTracker()

    kinds: collections.Counter[str] = collections.Counter()
    lookup_ms: list[float] = []
    ocr_ms: list[float] = []
    # per matched line: first time the lookup resolved it and first time text was seen
    first_seen: dict[str, float] = {}
    first_hit: dict[str, tuple[float, str]] = {}
    current_key: str | None = None
    pending_since: float | None = None  # first unresolved reading since the region changed
    with VideoSource(args.video, sample_fps=args.fps, start=args.start, end=args.end) as src:
        for frame in src:
            diff = det.update(frame.image)
            if diff is not None and det.changed_fraction(diff, region) < 0.002:
                continue
            res = engine.read(frame.image, region=region)
            ocr_ms.append(res.elapsed_ms)
            rows = [
                ln
                for ln in res.lines
                if ln.conf >= 0.6 and cjk_count(ln.text) >= 2 and not profile.is_ignored(ln.box)
            ]
            text, conf, _ = block_text(rows)
            if not text:
                current_key = None
                pending_since = None
                continue
            t0 = time.perf_counter()
            m = lookup.match(text)
            lookup_ms.append((time.perf_counter() - t0) * 1000)
            kinds[m.kind if m else "miss"] += 1
            if m is None:
                if pending_since is None:
                    pending_since = frame.ts
                continue
            key = m.line.key
            if key != current_key:
                current_key = key
                first_seen.setdefault(key, pending_since if pending_since is not None else frame.ts)
                first_hit.setdefault(key, (frame.ts, m.kind))
            pending_since = None

    total = sum(kinds.values())
    print(
        f"readings: {total}  "
        + "  ".join(f"{k}={v} ({v / max(total, 1):.0%})" for k, v in kinds.items())
    )
    if ocr_ms:
        ocr_ms.sort()
        lookup_ms.sort()
        print(
            f"ocr ms median={ocr_ms[len(ocr_ms) // 2]:.0f} "
            f"p90={ocr_ms[int(len(ocr_ms) * 0.9)]:.0f}   "
            f"lookup ms median={lookup_ms[len(lookup_ms) // 2]:.2f} max={lookup_ms[-1]:.2f}"
        )
    print(f"distinct lines resolved: {len(first_hit)} of {len(lookup.keys)} in db")
    delays = sorted(first_hit[k][0] - first_seen[k] for k in first_hit)
    if delays:
        print(
            f"first-resolve delay after a line appears: median={delays[len(delays) // 2]:.2f}s "
            f"p90={delays[int(len(delays) * 0.9)]:.2f}s"
        )
    for k, (ts, kind) in sorted(first_hit.items(), key=lambda kv: kv[1][0]):
        print(f"  {format_ts(ts)} {kind:6s} {lookup.by_key[k].source_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
