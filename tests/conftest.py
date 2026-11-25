"""Shared fixtures: a private data folder for every test, the game profile, the segment's
database lines, Qt font metrics, and a helper that runs a footage fixture through the
production core with its recorded OCR.

Markers (pyproject.toml): ``windows_only`` tests are skipped elsewhere, ``x11`` tests need
an X server ($DISPLAY: run them under Xvfb), ``footage`` tests (and every test that uses the
``lookup``, ``run`` or ``player`` fixture) need the stream frames and lines, which the public source
tree leaves out."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from capture.profile import GAMES_DIR, load_profile
from db.glossary import Glossary
from db.lookup import Lookup
from db.store import Line
from hud.covercache import CoverCache
from hud.pipeline import HudCore
from hud.style import get_style
from tools.replay import RecordedEngine, load_fixture

GAME = "azur-promilia"
ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "tests" / "frames" / GAME
LINES = ROOT / "tests" / "fixtures" / GAME / "segment_4630_lines.json"
FONTS = ROOT / "assets" / "fonts"
HAS_FOOTAGE = FRAMES.is_dir() and LINES.is_file()
FOOTAGE_FIXTURES = {"lookup", "run", "player"}


def pytest_collection_modifyitems(config, items) -> None:
    on_x11 = sys.platform.startswith("linux") and bool(os.environ.get("DISPLAY"))
    for item in items:
        if "windows_only" in item.keywords and sys.platform != "win32":
            item.add_marker(pytest.mark.skip(reason="needs Windows"))
        if "x11" in item.keywords and not on_x11:
            item.add_marker(pytest.mark.skip(reason="needs an X server (Xvfb)"))
        needs_footage = "footage" in item.keywords or FOOTAGE_FIXTURES & set(item.fixturenames)
        if needs_footage and not HAS_FOOTAGE:
            item.add_marker(pytest.mark.skip(reason="needs the stream footage (not public)"))


@pytest.fixture(autouse=True)
def private_data(tmp_path, monkeypatch):
    """Every test gets a data folder of its own: settings, keys and games never touch the
    user's real ones (a test may point GTHUD_DATA elsewhere itself)."""
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "gthud-data"))


@pytest.fixture(scope="session", autouse=True)
def qt_metrics():
    """A QApplication (the panel tests need widgets) and the HUD font's metrics. The fonts
    the app bundles are registered, so metrics are the same on every system (the offscreen
    platform has no system fonts on Windows without QT_QPA_FONTDIR)."""
    try:
        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - PySide6 is a dependency
        pytest.skip("PySide6 missing")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    for font in sorted(FONTS.glob("*.[ot]tf")):
        QFontDatabase.addApplicationFont(str(font))
    from hud.textwidth import build_table

    build_table()
    yield app


@pytest.fixture(scope="session")
def profile():
    return load_profile(GAME)


@pytest.fixture(scope="session")
def player() -> str:
    """The player's name in the footage's game (the frames' manifest says it)."""
    with open(FRAMES / "fixtures.yaml", encoding="utf-8") as f:
        return str(yaml.safe_load(f)["player"])


@pytest.fixture(scope="session")
def lookup(player):
    with open(LINES, encoding="utf-8") as f:
        data = json.load(f)
    return Lookup([Line(**d) for d in data["lines"]], player_name=player)


@pytest.fixture(scope="session")
def glossary():
    return Glossary.load(GAMES_DIR / GAME / "glossary.yaml", target="en")


@pytest.fixture
def run(profile, lookup, glossary):
    """run(name, cache=None) -> (core, frame): one HUD tick on a footage fixture."""

    def _run(name: str, cache: CoverCache | None = None, settle: bool = False):
        frame, meta = load_fixture(GAME, name)
        core = HudCore(
            profile,
            lookup,
            glossary,
            cache if cache is not None else CoverCache(None),
            RecordedEngine(meta["ocr"]),
            get_style("glow"),
        )
        core.tick(frame, lambda: frame, 0.0)
        if settle:  # the screen stays as it is for a while: its covers are settled
            core.tick(frame, lambda: frame, 2.0)
        return core, frame

    return _run
