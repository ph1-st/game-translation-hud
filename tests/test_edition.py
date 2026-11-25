"""The two apps the code makes: the general one, and the one made for Azur Promilia (the
edition is read when the app package is imported, so each case runs in its own Python)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = """
import json
from app import APP_ID, APP_NAME, ONE_GAME
from app.config import Settings
from app.library import Library
s = Settings()
lib = Library(s)
lib.ensure_defaults()
print(json.dumps({"name": APP_NAME, "id": APP_ID, "one": ONE_GAME,
                  "profiles": [p for p, _, _ in lib.profiles()],
                  "games": [g.profile for g in s.games], "current": s.current_game.profile}))
"""


def run(edition: str, tmp: Path) -> dict:
    env = {**os.environ, "GTHUD_EDITION": edition, "GTHUD_DATA": str(tmp / edition)}
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, check=True,
                         capture_output=True, text=True, timeout=120)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_azur_promilia_app_has_that_game_only(tmp_path: Path) -> None:
    ap = run("azur-promilia", tmp_path)
    assert ap["name"] == "Azur Promilia Translator" and ap["id"] == "AzurPromiliaTranslator"
    assert ap["profiles"] == ["azur-promilia"]
    assert ap["games"] == ["azur-promilia"] and ap["current"] == "azur-promilia"


def test_the_general_app_offers_every_preset_and_any_game(tmp_path: Path) -> None:
    gen = run("general", tmp_path)
    assert gen["name"] == "Game Translation HUD" and gen["one"] is None
    assert "azur-promilia" in gen["profiles"] and gen["profiles"][-1] == "_generic"
    assert run("nonsense", tmp_path)["id"] == "GameTranslationHUD"  # unknown: the general app
