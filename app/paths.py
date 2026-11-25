"""Where things are: the app's own files (inside the bundle when packaged) and the user's
data (settings, games, databases, caches, logs).

The user's data folder: %LOCALAPPDATA%\\<app> on Windows, $XDG_DATA_HOME/<app> (by default
~/.local/share/<app>) elsewhere; the environment variable GTHUD_DATA overrides both (tests,
portable installs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from app import APP_ID


def frozen() -> bool:
    """Running as the packaged app (PyInstaller)?"""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """The app's own read-only files: the source tree, or the unpacked bundle."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def builtin_games_dir() -> Path:
    """Game profiles that ship with the app (games/<id>/profile.yaml)."""
    return bundle_dir() / "games"


def assets_dir() -> Path:
    return bundle_dir() / "app" / "assets"


def user_data_root() -> Path:
    """The system's folder for per-user application data."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base) if base else Path.home() / ".local" / "share"


def data_dir() -> Path:
    """The user's data: settings, the game library, logs (made when missing)."""
    override = os.environ.get("GTHUD_DATA")
    root = Path(override) if override else user_data_root() / APP_ID
    root.mkdir(parents=True, exist_ok=True)
    return root


def games_data_dir() -> Path:
    d = data_dir() / "games"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    """The model that translates on this computer, and the program that runs it."""
    return data_dir() / "models"
