"""The desktop application: the start window, the HUD's panel and the overlay they run.

* :mod:`app.paths`: where the app's files and the user's data live
* :mod:`app.config`: settings (the game library, what is translated, live translation)
* :mod:`app.keys`: the API keys, kept in the system's credential store
* :mod:`app.library`: games, each with its databases, caches and translation memory
* :mod:`app.dbfolder`: a game's database folder, checked, with each database's language
* :mod:`app.session`: the overlay for one game (the HUD core of :mod:`hud`)
* :mod:`app.misslog`: the lines no database had, written down while playing
* :mod:`app.reader`: which text reader runs, and installing MeikiOCR (NVIDIA)
* :mod:`app.start_window`: the window the app opens with (games, languages, databases)
* :mod:`app.panel`: the HUD's panel (and the app around it) and the tray icon
* :mod:`app.dialogs`, :mod:`app.widgets`, :mod:`app.theme`: the parts the windows share

The same code is two apps (editions): the general Game Translation HUD, for any game, and
the Azur Promilia Translator, made for that one game (its preset, no game list, a shorter
panel). The packaged app says which it is in ``edition.txt`` beside its files; the
environment variable GTHUD_EDITION chooses when running from the source.
"""

import os
import sys
from pathlib import Path

# edition -> (name, id: file names and the data folder, the one game it is made for)
EDITIONS = {
    "general": ("Game Translation HUD", "GameTranslationHUD", None),
    "azur-promilia": ("Azur Promilia Translator", "AzurPromiliaTranslator", "azur-promilia"),
}


def _edition() -> str:
    chosen = os.environ.get("GTHUD_EDITION", "").strip()
    if not chosen and getattr(sys, "frozen", False):
        here = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        try:
            chosen = (here / "edition.txt").read_text(encoding="utf-8").strip()
        except OSError:
            chosen = ""
    return chosen if chosen in EDITIONS else "general"


EDITION = _edition()
APP_NAME, APP_ID, ONE_GAME = EDITIONS[EDITION]
VERSION = "1.2.0"
# the guide, for both apps (the panel shows no long texts: its Help opens this)
HELP_URL = "https://github.com/ph1-st/game-translation-hud#readme"
