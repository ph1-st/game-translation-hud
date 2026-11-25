"""Build the test fixtures: footage frames plus the OCR results of the HUD's first tick.

    uv run python tools/make_fixtures.py --game azur-promilia \
        --db data/db/test_segment_v2.sqlite --player NAME [NAME ...]

Reads tests/frames/<game>/fixtures.yaml; writes <name>.png and <name>.json (the recorded
OCR calls) next to it. Run it again after an OCR change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.lookup import Lookup  # noqa: E402
from hud.app import add_core_args, build_core, config_from_args  # noqa: E402
from hud.covercache import CoverCache  # noqa: E402
from hud.pipeline import HudCore  # noqa: E402
from tools.replay import FIXTURES, RecordingEngine  # noqa: E402

INTERP = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA, "nearest": cv2.INTER_NEAREST}


def main() -> int:
    ap = argparse.ArgumentParser()
    add_core_args(ap)
    ap.add_argument("names", nargs="*", help="fixtures to build (default: all)")
    args = ap.parse_args()
    args.no_cache = True
    folder = FIXTURES / args.game
    with open(folder / "fixtures.yaml", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    cap = cv2.VideoCapture(manifest["video"])
    if not cap.isOpened():
        print("cannot open", manifest["video"])
        return 1
    core = build_core(config_from_args(args))
    real = core.engine
    for name, fx in manifest["fixtures"].items():
        if args.names and name not in args.names:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fx["frame"]))
        ok, src = cap.read()
        if not ok:
            print("cannot read frame", fx["frame"])
            continue
        w = int(manifest["width"])
        h = int(round(w * src.shape[0] / src.shape[1]))
        frame = cv2.resize(src, (w, h), interpolation=INTERP[manifest.get("interp", "linear")])
        cv2.imwrite(str(folder / f"{name}.png"), frame)
        rec = RecordingEngine(real)
        fresh = HudCore(
            core.profile, core.maker.lookup, core.glossary, CoverCache(None), rec,
            core.ui_board.style,
        )  # fmt: skip
        fresh.tick(frame, lambda f=frame: f, 0.0)
        # and what a tick reads when no database knows the text (live translation's tests)
        bare = HudCore(core.profile, Lookup([]), core.glossary, CoverCache(None), rec,
                       core.ui_board.style)  # fmt: skip
        bare.tick(frame, lambda f=frame: f, 0.0)
        meta = {
            "source": {"video": manifest["video"], "frame": int(fx["frame"]), "width": w},
            "note": fx.get("note", ""),
            "ocr": rec.calls,
        }
        with open(folder / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        n = sum(len(v) for v in rec.calls.values())
        print(f"{name}: {len(rec.calls)} OCR calls, {n} readings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
