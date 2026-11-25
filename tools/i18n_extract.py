"""Collect the app's translatable strings and bring the catalogs up to date.

    python tools/i18n_extract.py            # i18n/locale/messages.pot and every <lang>.po
    python tools/i18n_extract.py --check    # exit 1 when a catalog lacks a translation

Strings are found in the source with the ast module: calls of ``_("...")``, ``N_("...")``
and ``ngettext("...", "...", n)`` with literal arguments. A catalog keeps its translations;
strings that are new get an empty entry (shown in English until translated), strings no
longer in the source are dropped.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from i18n import LOCALE_DIR, read_po  # noqa: E402

PACKAGES = ("app", "hud", "capture", "db", "extract", "lore", "sync", "translate", "i18n")
# names the language tables give in English, shown translated where they appear
EXTRA = ("Japanese", "Chinese (Simplified)", "Chinese (Traditional)", "Korean")


def collect() -> dict[str, tuple[str | None, list[str]]]:
    """msgid -> (plural or None, where it is used)."""
    found: dict[str, tuple[str | None, list[str]]] = {}

    def add(msgid: str, plural: str | None, where: str) -> None:
        old = found.get(msgid)
        found[msgid] = (plural or (old[0] if old else None), (old[1] if old else []) + [where])

    for pkg in PACKAGES:
        for path in sorted((ROOT / pkg).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), rel)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                args = [a.value if isinstance(a, ast.Constant) and isinstance(a.value, str)
                        else None for a in node.args]
                if node.func.id in ("_", "N_") and args and args[0]:
                    add(args[0], None, f"{rel}:{node.lineno}")
                elif node.func.id == "ngettext" and len(args) >= 2 and args[0] and args[1]:
                    add(args[0], args[1], f"{rel}:{node.lineno}")
    for msgid in EXTRA:
        add(msgid, None, "translate/languages.py")
    return found


def quote(text: str) -> str:
    body = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{body}"'


def write_catalog(path: Path, found: dict, old: dict, header: str) -> int:
    """Write *path* with the translations in *old*; returns how many are missing."""
    out = [header.rstrip("\n"), ""]
    missing = 0
    for msgid in sorted(found, key=str.casefold):
        plural, where = found[msgid]
        out.append("#: " + " ".join(sorted(set(where))[:4]))
        out.append(f"msgid {quote(msgid)}")
        have = old.get(msgid)
        if plural is None:
            text = have if isinstance(have, str) else ""
            missing += not text
            out.append(f"msgstr {quote(text)}")
        else:
            out.append(f"msgid_plural {quote(plural)}")
            forms = have if isinstance(have, list) else ["", ""]
            missing += not all(forms)
            for i, form in enumerate(forms or ["", ""]):
                out.append(f"msgstr[{i}] {quote(form)}")
        out.append("")
    path.write_bytes(("\n".join(out)).encode("utf-8"))
    return missing


def header_of(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    head, _sep, _rest = text.partition("\n\n")
    return head + "\n"


POT_HEADER = '''# Translatable strings of the Game Translation HUD (made by tools/i18n_extract.py).
msgid ""
msgstr ""
"Content-Type: text/plain; charset=UTF-8\\n"
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report missing translations")
    args = ap.parse_args()
    found = collect()
    if not args.check:
        write_catalog(LOCALE_DIR / "messages.pot", found, {}, POT_HEADER)
    bad = 0
    for po in sorted(LOCALE_DIR.glob("*.po")):
        old, _head = read_po(po)
        if args.check:
            missing = [m for m in found if not old.get(m)]
            bad += len(missing)
            print(f"{po.name}: {len(missing)} of {len(found)} strings untranslated")
            continue
        missing = write_catalog(po, found, old, header_of(po))
        print(f"{po.name}: {len(found)} strings, {missing} untranslated")
    return 1 if args.check and bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
