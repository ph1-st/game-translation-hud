"""The app's own words in other languages: every catalog reads, every translation keeps
its placeholders, and a string without one shows in English."""

from __future__ import annotations

import re
import string

import pytest

import i18n

FIELDS = re.compile(r"\{[^{}]*\}")


def placeholders(text: str) -> set[str]:
    return {name for _lit, name, _spec, _conv in string.Formatter().parse(text) if name}


@pytest.fixture(autouse=True)
def english_after():
    yield
    i18n.set_language("en")


@pytest.mark.parametrize("lang", [c for c in i18n.available() if c != "en"])
def test_every_translation_keeps_its_placeholders(lang):
    catalog, header = i18n.read_po(i18n.LOCALE_DIR / f"{lang}.po")
    assert "Plural-Forms" in header
    for msgid, translated in catalog.items():
        forms = translated if isinstance(translated, list) else [translated]
        for form in forms:
            assert placeholders(form) == placeholders(msgid), (msgid, form)


def test_a_language_is_used_with_its_variants_and_english_is_the_fallback():
    assert i18n.set_language("es-419") == "es"
    assert i18n._("Play") == "Jugar"
    assert i18n._("A string no catalog has") == "A string no catalog has"
    assert i18n.ngettext("{n} live translation", "{n} live translations", 1) == \
        "{n} traducción en vivo"
    assert i18n.set_language("xx") == "en" and i18n._("Play") == "Play"


def test_the_catalog_is_complete():
    """Every string the app marks is translated (tools/i18n_extract.py --check)."""
    import subprocess
    import sys

    from i18n import LOCALE_DIR

    tool = LOCALE_DIR.parent.parent / "tools" / "i18n_extract.py"
    run = subprocess.run([sys.executable, "-X", "utf8", str(tool), "--check"],
                         capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stdout + run.stderr


def test_po_syntax_errors_are_reported(tmp_path):
    bad = tmp_path / "xx.po"
    bad.write_text('msgid "a"\nmsgstr "b"\nnonsense\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        i18n.read_po(bad)
