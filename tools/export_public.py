"""Make the public source tree: every tracked file except material that is not ours to
publish or only matters to this working copy, checked for personal data.

    python tools/export_public.py OUT_DIR          # copy, then check
    python tools/export_public.py OUT_DIR --check  # only check an existing tree

Left out: screenshots of a game stream and the game lines with translations made from it
(tests that use them skip without them), the stream-specific capture overrides, and
internal planning and working notes. The check fails when a file holds an e-mail address of this
repository's commits (read from its history, never written down), a user's home folder
path, or a secret-looking token. OUT_DIR is created fresh; it is not a git repository: start the
public history there with ``git init`` (one commit, no private history).
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = (
    "tests/frames/azur-promilia/*",  # screenshots of a stream of the game's beta
    "tests/fixtures/azur-promilia/*",  # the game's lines with translations, from that stream
    "games/azur-promilia/sources/*",  # capture overrides naming a streamer
    "docs/LAUNCH_PLAN.md",  # internal planning
    "CLAUDE.md",  # notes for the maintainer's coding assistant
)
SECRET = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}")
HOME = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+|/home/[a-z][a-z0-9_-]+/|/Users/[^/\s]+/")
HOME_ALLOWED = ("C:\\Users\\...", "C:\\Users\\<you>", "C:\\Users\\me")  # examples
SELF = "tools/export_public.py"


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def excluded(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in EXCLUDE)


def private_emails() -> set[str]:
    """Every e-mail address this repository's history or git config names, except GitHub's
    no-reply addresses (those are meant to be public)."""
    found: set[str] = set()
    for cmd in (["git", "log", "--all", "--format=%ae%n%ce"], ["git", "config", "user.email"]):
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        found |= {e.strip().lower() for e in out.stdout.splitlines() if "@" in e}
    return {e for e in found if not e.endswith("@users.noreply.github.com")}


def check(tree: Path) -> list[str]:
    """Problems found in *tree* (file: what)."""
    problems = []
    private = sorted(private_emails())
    for path in sorted(p for p in tree.rglob("*") if p.is_file()):
        rel = path.relative_to(tree).as_posix()
        if rel.startswith(".git/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # (pictures, fonts: no text to hold an address)
        low = text.lower()
        if any(word in low for word in private):
            problems.append(f"{rel}: holds a committer's e-mail address")
        if rel == SELF:
            continue  # (this file spells the patterns it looks for)
        for m in HOME.finditer(text):
            if not m.group(0).startswith(HOME_ALLOWED):
                problems.append(f"{rel}: a home folder path ({m.group(0)[:40]})")
                break
        if SECRET.search(text):
            problems.append(f"{rel}: something that looks like a secret key")
    return problems


def export(out: Path) -> int:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: choose a new folder")
    n = 0
    for rel in tracked():
        if excluded(rel):
            continue
        src, dest = ROOT / rel, out / rel
        if not src.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", action="store_true", help="only check an existing tree")
    args = ap.parse_args()
    if not args.check:
        print(f"{export(args.out)} files copied to {args.out}")
    problems = check(args.out)
    for p in problems:
        print("✘", p)
    print("✔ nothing personal found" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
