"""Pictures of the app for the guide: the start window and the HUD panel, rendered
offscreen (nothing appears on the screen) with sample games and databases in a data folder
at the drive's root (``GameTranslationHUD``: the folder the pictures show, no one's user
folder), removed afterwards.

    python tools/make_screenshots.py [--lang es] [--out docs/images]

Writes start_window.png and panel.png (``_es`` before the extension for another interface
language).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PANEL_SIZE = (410, 900)


def sample_data(data: Path) -> None:
    """Two games with databases in English and Spanish, as a player would have them."""
    from app.config import GameEntry, Settings
    from db.store import Line, Store

    os.environ["GTHUD_DATA"] = str(data)
    s = Settings()
    s.live.engine, s.live.enabled = "plamo", True
    from app.library import Library

    lib = Library(s)
    lib.ensure_defaults()  # (the presets' games)
    g = s.current_game
    s.games.append(GameEntry("my-visual-novel", "My Visual Novel",
                             window_title="My Visual Novel"))
    g.window_title = "AzurPromiliaJP"
    lines = [("はい", "Yes", "Sí"), ("いいえ", "No", "No"), ("設定", "Settings", "Ajustes")]
    for lang, col, version in (("en", 1, "2026.11.25-1"), ("es-419", 2, "2026.11.25-1")):
        path = lib.db_dir(g) / f"azur-promilia.{lang}.sqlite"
        with Store(path) as store:
            store.set_info(game="azur-promilia", source_lang="ja", target_lang=lang,
                           version=version, display_name="Azur Promilia")
            for i, row in enumerate(lines):
                store.upsert_line(Line(row[0], row[0], None, "ui", False, row[col], "machine",
                                       "batch", None, i, None))
    s.save()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en", help="the interface language")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    args = ap.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    import i18n

    i18n.set_language(args.lang)
    data = Path(ROOT.anchor) / "GameTranslationHUD"
    if data.exists():
        raise SystemExit(f"{data} exists: move it away first (the pictures are made there)")
    try:
        sample_data(data)
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from app.config import Settings
        from app.panel import MainWindow
        from app.theme import apply_theme

        apply_theme(app)
        win = MainWindow(Settings.load(), desktop=False)
        suffix = "" if args.lang == "en" else f"_{args.lang}"
        args.out.mkdir(parents=True, exist_ok=True)
        shots = []
        for w in (win, win.start):
            w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        win.start.open()
        shots.append((win.start, "start_window"))
        win.resize(*PANEL_SIZE)
        win.refresh_all()
        win.show()
        shots.append((win, "panel"))
        for _i in range(20):
            app.processEvents()
        for widget, name in shots:
            path = args.out / f"{name}{suffix}.png"
            widget.grab().save(str(path))
            print(path)
        win.close_down()
    finally:
        shutil.rmtree(data, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
