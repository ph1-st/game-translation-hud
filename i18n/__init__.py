"""The app's own words in the player's language (its interface, not the game's text).

Translations are gettext ``.po`` files in ``i18n/locale/<code>.po``, read as they are (no
compile step). English is the source language: a string without a translation shows in
English.

    from i18n import _, ngettext
    _("Play")
    ngettext("{n} line", "{n} lines", n).format(n=n)

The language is set once at start (:func:`set_language`). Strings defined at import time
(constants) are marked with :func:`N_` and translated where they are shown.
"""

from __future__ import annotations

import ast
import locale
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)

LOCALE_DIR = Path(__file__).resolve().parent / "locale"
SOURCE_LANGUAGE = "en"

_catalog: dict[str, str | list[str]] = {}
_plural = lambda n: 0 if n == 1 else 1  # noqa: E731 - replaced per language
_language = SOURCE_LANGUAGE


def _(text: str) -> str:
    """*text* in the interface language (*text* itself when untranslated)."""
    hit = _catalog.get(text)
    return hit if isinstance(hit, str) and hit else text


def ngettext(singular: str, plural: str, n: int) -> str:
    """The form for *n* of a text with plurals."""
    forms = _catalog.get(singular)
    if isinstance(forms, list) and forms:
        i = _plural(n)
        if 0 <= i < len(forms) and forms[i]:
            return forms[i]
    return singular if n == 1 else plural


def N_(text: str) -> str:  # noqa: N802 - the gettext convention
    """Marks *text* for translation without translating it (translate it where shown)."""
    return text


def language() -> str:
    return _language


def available() -> list[str]:
    """The interface languages: English and every catalog shipped."""
    return [SOURCE_LANGUAGE] + sorted(p.stem for p in LOCALE_DIR.glob("*.po"))


def set_language(code: str | None) -> str:
    """Use the catalog for *code* (``es-419`` falls back to ``es``; None or "auto": the
    system's language). Returns the language now in use."""
    global _catalog, _plural, _language
    code = (code or "auto").lower()
    if code == "auto":
        code = system_language()
    _catalog, _language = {}, SOURCE_LANGUAGE
    _plural = lambda n: 0 if n == 1 else 1  # noqa: E731
    for candidate in (code, code.split("-")[0]):
        path = LOCALE_DIR / f"{candidate}.po"
        if candidate != SOURCE_LANGUAGE and path.is_file():
            try:
                _catalog, header = read_po(path)
            except (OSError, ValueError) as e:
                log.warning("interface translation %s not read: %s", path.name, e)
                break
            _plural = plural_rule(header)
            _language = candidate
            break
    return _language


def system_language() -> str:
    """The language of the computer's user interface, as a code (en, es...)."""
    try:
        from PySide6.QtCore import QLocale

        name = QLocale.system().name()  # es_MX
    except ImportError:
        name = locale.getlocale()[0] or ""
    return (name or SOURCE_LANGUAGE).replace("_", "-").lower()


# ---- reading .po files ------------------------------------------------------------------
_KEYWORD = re.compile(r'^(msgctxt|msgid_plural|msgid|msgstr(?:\[(\d+)\])?)\s+(".*")\s*$')


def read_po(path: Path) -> tuple[dict[str, str | list[str]], dict[str, str]]:
    """The translations of a .po file (msgid -> msgstr, or a list of plural forms) and its
    header fields. Fuzzy entries are skipped, as gettext skips them."""
    entries: list[dict] = []
    cur: dict = {}
    field: tuple[str, int | None] | None = None
    fuzzy = False

    def finish() -> None:
        nonlocal cur, fuzzy
        if "msgid" in cur and not fuzzy:
            entries.append(cur)
        cur, fuzzy = {}, False

    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            finish()
            field = None
            continue
        if line.startswith("#"):
            if line.startswith("#,") and "fuzzy" in line:
                fuzzy = True
            continue
        m = _KEYWORD.match(line)
        if m:
            kw, index = m.group(1), m.group(2)
            if kw == "msgid" and "msgid" in cur:
                finish()
            name = "msgstr" if kw.startswith("msgstr") else kw
            field = (name, int(index) if index is not None else None)
            _append(cur, field, _unquote(m.group(3), path, n))
        elif line.startswith('"') and field is not None:
            _append(cur, field, _unquote(line, path, n))
        else:
            raise ValueError(f"{path.name} line {n}: not .po syntax")
    finish()
    catalog: dict[str, str | list[str]] = {}
    header: dict[str, str] = {}
    for e in entries:
        msgid = e.get("msgid", "")
        strs = e.get("msgstr", {})
        if msgid == "":
            for hl in strs.get(None, "").splitlines():
                k, _sep, v = hl.partition(":")
                header[k.strip()] = v.strip()
            continue
        if "msgid_plural" in e:
            catalog[msgid] = [strs[i] for i in sorted(k for k in strs if k is not None)]
        elif strs.get(None):
            catalog[msgid] = strs[None]
    return catalog, header


def _append(cur: dict, field: tuple[str, int | None], text: str) -> None:
    name, index = field
    if name == "msgstr":
        forms = cur.setdefault("msgstr", {})
        forms[index] = forms.get(index, "") + text
    else:
        cur[name] = cur.get(name, "") + text


def _unquote(token: str, path: Path, n: int) -> str:
    try:
        value = ast.literal_eval(token)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"{path.name} line {n}: bad string") from e
    if not isinstance(value, str):
        raise ValueError(f"{path.name} line {n}: bad string")
    return value


_PLURALS = {  # Plural-Forms expressions this app's catalogs use, as functions
    "n != 1": lambda n: int(n != 1),
    "n>1": lambda n: int(n > 1),
    "n > 1": lambda n: int(n > 1),
    "0": lambda n: 0,
}


def plural_rule(header: dict[str, str]):  # noqa: ANN201 - Callable[[int], int]
    """The plural rule a .po header names (Plural-Forms: nplurals=2; plural=(n != 1);).
    Only the common simple rules are known; any other falls back to English's."""
    spec = header.get("Plural-Forms", "")
    m = re.search(r"plural\s*=\s*\(?([^;]*?)\)?\s*;", spec)
    expr = m.group(1).strip() if m else "n != 1"
    rule = _PLURALS.get(expr)
    if rule is None:
        log.warning("unknown plural rule %r: English's used", expr)
        rule = _PLURALS["n != 1"]
    return rule


if "pytest" in sys.modules:  # tests see the source strings
    set_language(SOURCE_LANGUAGE)
