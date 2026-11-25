"""The player's own API keys (Claude, ChatGPT), for live translation and for making a
database: kept in the system's credential store (Windows Credential Manager; the Secret
Service, as GNOME Keyring or KWallet, on Linux), never in the settings file, one entry per
service. Without a credential store a key falls back to a file in the data folder that
only the user's account can read.

The functions default to the Claude key, which was the only one before: ``get_key()`` and
``set_key(value)`` read and write it under the names it always had."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from app import APP_ID
from app.paths import data_dir

log = logging.getLogger("app")

SERVICE = APP_ID
SERVICES = ("claude", "openai")
ACCOUNT = "claude-api-key"  # the Claude key's entry (account_for("claude"))


def account_for(service: str = "claude") -> str:
    """The key's entry in the credential store: "claude-api-key", "openai-api-key"."""
    return f"{_checked(service)}-api-key"


def _checked(service: str) -> str:
    if service not in SERVICES:
        raise ValueError(f"unknown translation service {service!r}")
    return service


def _keyring():
    try:
        import keyring

        keyring.get_keyring()
        return keyring
    except Exception:  # noqa: BLE001 - no backend: the file fallback
        return None


def _file(service: str = "claude") -> Path:
    return data_dir() / f"{_checked(service)}_api_key.txt"


def get_key(service: str = "claude") -> str:
    """The stored key of *service* ("claude" or "openai"); empty if none."""
    account, path = account_for(service), _file(service)
    kr = _keyring()
    if kr is not None:
        try:
            v = kr.get_password(SERVICE, account)
            if v:
                return v
        except Exception:  # noqa: BLE001
            log.exception("credential store read failed")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def set_key(value: str, service: str = "claude") -> str:
    """Store *value* as the key of *service* (empty: remove it). Returns where it went, in
    words."""
    account, path = account_for(service), _file(service)
    value = value.strip()
    kr = _keyring()
    if kr is not None:
        try:
            if value:
                kr.set_password(SERVICE, account, value)
            else:
                try:
                    kr.delete_password(SERVICE, account)
                except Exception:  # noqa: BLE001 - nothing stored
                    pass
            path.unlink(missing_ok=True)
            return "the system's credential store"
        except Exception:  # noqa: BLE001
            log.exception("credential store write failed; using a file")
    if value:
        _write_private(path, value)
    else:
        path.unlink(missing_ok=True)
    return str(path)


def _write_private(path: Path, value: str) -> None:
    """Write *value* to a file only the user can read (on Windows the data folder under the
    user's profile is private already)."""
    if sys.platform == "win32":
        path.write_text(value, encoding="utf-8")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(value)
    os.chmod(path, 0o600)  # (a file made before with other permissions)


def mask(key: str) -> str:
    """The key as it is shown: its start and end only."""
    if len(key) <= 12:
        return "•" * len(key)
    return key[:7] + "…" + key[-4:]
