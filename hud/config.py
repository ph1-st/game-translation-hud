"""What the HUD runs with. The app fills a :class:`HudConfig` from its settings and the
game's folder; ``python -m hud.app`` fills one from its command line."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from hud.style import DEFAULT_STYLE

if TYPE_CHECKING:
    from capture.profile import Profile
    from db.glossary import Glossary

# (the names hud.pipeline and hud.dialogue use; spelled out here, where reading the
# settings needs no OCR or Qt)
COVER_MODES = ("natural", "fixed", "grow")  # how dialogue is covered
CONFIRMATION_MODES = ("instant", "balanced")  # how sure a cover must be first


@dataclass
class HudConfig:
    game: str  # the game's profile id (games/<game>/profile.yaml)
    dbs: list[Path] = field(default_factory=list)  # translation databases, the best first
    player: str | None = None  # the player's name in the game (fills {player})
    target_lang: str = "en"  # the language the databases translate into
    profile: Profile | None = None  # the app's own profile (picked window, set area);
    # None: the game's profile file
    overrides: Path | None = None  # extra profile settings (a recording's)
    glossary: Glossary | Path | None = None  # names and terms; None: games/<game>/glossary.yaml
    cache_dir: Path | None = None  # cover cache and menu memory; None: nothing kept
    device: str = "auto"  # OCR device (capture.ocr.DEVICES)
    reader: str = "standard"  # standard | meiki (MeikiOCR on CUDA: app/reader.py)
    reader_dir: Path | None = None  # MeikiOCR's folder (runtime/, models/)
    confirmation_mode: str = "balanced"
    cover_style: str = DEFAULT_STYLE
    cover_mode: str = "grow"
    text_scale: float = 1.0  # the translation's size (1: the game's own; 0.8: smaller)
    plain_text: bool = False  # white with a black outline, not the Japanese's colours
    title: str | None = None  # a window title regex instead of the profile's
    capture: str = "auto"  # capture backend (Windows: auto | dxcam | mss)
    interval_ms: int = 50  # capture and pixel watch period
    toggle_hotkey: str = "ctrl+shift+F8"  # shows / hides the translations
    quit_hotkey: str = "ctrl+shift+F9"  # stops the overlay
    verbose: bool = False  # log every OCR tick
    on_quit: Callable[[], None] | None = None  # the quit hotkey's action (None: quit Qt)
    on_raise: Callable[[], None] | None = None  # after the overlay went to the top (the
    # app raises its panel above it again)
