"""Settings, kept as JSON in the data folder: the game library, what is translated, live
translation and the hotkeys. Unknown keys are ignored and missing or mistyped ones take
their defaults, so an older settings file always loads."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from app.paths import data_dir
from capture.ocr import DEVICES as OCR_DEVICES
from hud.config import CONFIRMATION_MODES

log = logging.getLogger("app")

GLASS_STYLES = ("grow", "window")  # dialogue glass: grows with the text | one fixed window
TEXT_SIZES = (70, 80, 90, 100, 115, 130, 150)  # the translation's size, % of the game's
# PLaMo on this computer, a key of your own, or a model behind a server you name
LIVE_ENGINES = ("plamo", "claude", "openai", "deepl", "custom")
GAME_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")  # a folder name: no separators, no ".."
_SAVE_LOCK = threading.Lock()


@dataclass
class GameEntry:
    """One game in the library: its databases, caches and translation memory live in its
    folder (games/<id>/ in the data folder)."""

    id: str  # folder name, never changes
    name: str  # what the player calls it
    profile: str = "_generic"  # a built-in profile ("azur-promilia") or "_generic"
    window_title: str = ""  # the game window: its title (exact text)
    process_name: str = ""  # ...or the program's file name (e.g. "game.exe")
    player_name: str = ""  # the player's character name (lines with it can be matched)
    db_link: str = ""  # a database file or an index of databases (index.json) to download
    db_sync: str = ""  # what the last downloads were (JSON, per language: etag, sha256...)
    install_dir: str = ""  # the folder the game is installed in (to read its text)
    source_lang: str = ""  # the language the game shows: ja | zh-hans | zh-hant | ko
    target_lang: str = ""  # the language the player reads ("": English)
    # the game's dialogue area: its preset's, or one the player chose
    region_on: bool = True  # off: every text covered where it is ("natural")
    region: str = ""  # the area the player chose: "x,y,w,h" of the game's picture (0-1)
    name_region: str = ""  # optional: where the speaker's name is
    glass_style: str = "grow"  # one of GLASS_STYLES
    keep_glass: bool = False  # the glass stays on screen once it is up
    type_in: bool = True  # the translation types in like the game's text


@dataclass
class LiveSettings:
    enabled: bool = False  # off by default: text no database has is left as it is
    engine: str = "plamo"  # one of LIVE_ENGINES
    plamo_model: str = "2b"  # one of translate.local.MODELS
    claude_model: str = "claude-haiku-5-5"
    openai_model: str = "gpt-5.4-mini"
    custom_url: str = ""  # an OpenAI-compatible server ("http://localhost:11434/v1")
    custom_model: str = ""  # ...and the model it runs


@dataclass
class Settings:
    games: list[GameEntry] = field(default_factory=list)
    current: str = ""  # id of the game chosen in the start window
    ui_language: str = "auto"  # the app's own language ("auto": the system's)
    translate_dialogue: bool = True  # the story: dialogue, subtitles, narration, choices
    translate_ui: bool = True  # menus, labels, key hints and everything else
    log_misses: bool = True  # write down the lines no database has (games/<id>/misses/)
    text_size: int = 100  # the translation's size, one of TEXT_SIZES (100: the game's own)
    white_text: bool = False  # white with a black outline, not the Japanese's colours
    stream_source: bool = False  # the overlay served to OBS as a page (hud/overlay/stream.py)
    stream_port: int = 47811  # ...on this port of 127.0.0.1 (or the next free one)
    live: LiveSettings = field(default_factory=LiveSettings)
    toggle_hotkey: str = "ctrl+shift+F8"  # hides/shows the translations
    stop_hotkey: str = "ctrl+shift+F9"  # stops the overlay
    panel_hotkey: str = "ctrl+shift+F7"  # folds the app's panel away and back
    ocr_device: str = "auto"  # one of capture.ocr.DEVICES
    ocr_reader: str = "standard"  # standard | meiki (app/reader.py)
    confirmation_mode: str = "balanced"  # one of CONFIRMATION_MODES
    auto_sync: bool = True  # on start, say when the game's link has a newer database
    panel_side: str = "right"  # the side of the screen the panel lies on: right | left
    panel_folded: bool = False  # the panel was folded away when the app last closed
    panel_handle: bool = True  # a small handle on the screen's edge while folded

    # ---- persistence -------------------------------------------------------------------
    @classmethod
    def path(cls) -> Path:
        return data_dir() / "settings.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        """The settings in the file (what it holds wrongly takes its default); a file that
        cannot be read at all is kept aside (settings.<time>.bad.json) and the app starts
        fresh."""
        p = path or cls.path()
        try:
            raw = json.loads(p.read_text(encoding="utf-8-sig"))  # (Notepad may add a BOM)
            if not isinstance(raw, dict):
                raise ValueError("not a JSON object")
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, UnicodeDecodeError):
            log.exception("settings unreadable, starting fresh: %s", p)
            try:
                p.replace(p.with_name(f"{p.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bad.json"))
            except OSError:
                pass
            return cls()
        return cls.from_dict(raw)

    def save(self, path: Path | None = None) -> None:
        """Write the settings (atomically: a crash leaves the old file or the new one). Any
        thread may save; saves never interleave."""
        p = path or self.path()
        text = json.dumps(asdict(self), ensure_ascii=False, indent=2)
        with _SAVE_LOCK:
            fd, tmp = tempfile.mkstemp(prefix="settings.", suffix=".tmp", dir=p.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                _replace(Path(tmp), p)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Settings":
        s = _build(cls, d)
        s.games = []
        for g in d.get("games") if isinstance(d.get("games"), list) else []:
            if not isinstance(g, dict) or not isinstance(g.get("id"), str):
                continue
            name = g.get("name")
            entry = _build(GameEntry, {**g, "name": name if isinstance(name, str) else g["id"]})
            if entry.glass_style not in GLASS_STYLES:
                entry.glass_style = "grow"
            # the id is a folder name: nothing that could point elsewhere, no duplicates
            if GAME_ID_RE.fullmatch(entry.id) and s.game(entry.id) is None:
                s.games.append(entry)
        s.live = _build(LiveSettings, d.get("live") if isinstance(d.get("live"), dict) else {})
        if s.live.engine not in LIVE_ENGINES:
            s.live.engine = "plamo"
        if s.text_size not in TEXT_SIZES:
            s.text_size = 100
        if s.panel_side not in ("right", "left"):
            s.panel_side = "right"
        if s.ocr_device not in OCR_DEVICES:
            s.ocr_device = "auto"
        if s.ocr_reader not in ("standard", "meiki"):
            s.ocr_reader = "standard"
        if s.confirmation_mode not in CONFIRMATION_MODES:
            s.confirmation_mode = "balanced"
        if not (s.translate_dialogue or s.translate_ui):  # nothing at all is no choice
            s.translate_dialogue = s.translate_ui = True
        return s

    # ---- the library -------------------------------------------------------------------
    def game(self, game_id: str) -> GameEntry | None:
        return next((g for g in self.games if g.id == game_id), None)

    @property
    def current_game(self) -> GameEntry | None:
        return self.game(self.current) or (self.games[0] if self.games else None)


def _replace(tmp: Path, dest: Path, tries: int = 5) -> None:
    """os.replace, retried a moment when another program (a virus scanner, a backup tool)
    holds the old file open."""
    for i in range(tries):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def _build(cls, d: dict[str, Any]):  # noqa: ANN001, ANN202 - a dataclass type and instance
    """A dataclass from a dict, taking only the keys it knows with a value of the type its
    default has (a hand-edited "yes" for a switch keeps the default)."""
    kw = {}
    for f in fields(cls):
        required = f.default is MISSING and f.default_factory is MISSING
        if f.name not in d:
            if required:
                kw[f.name] = ""
            continue
        v = d[f.name]
        if required:  # (a game's id and name)
            kw[f.name] = str(v)
            continue
        default = f.default if f.default is not MISSING else f.default_factory()
        if isinstance(default, list):
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                kw[f.name] = list(v)
        elif isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
            kw[f.name] = float(v)
        elif type(v) is type(default):
            kw[f.name] = v
    return cls(**kw)
