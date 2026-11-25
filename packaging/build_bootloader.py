"""Compile PyInstaller's bootloader (the small launcher at the front of the .exe) from
source, for the PyInstaller version installed here.

    uv run python packaging/build_bootloader.py

The bootloader that ships with PyInstaller's wheels is the same file in every app made
with it, malware included, so antivirus programs flag it by its looks. One compiled here
is ours alone. Needs Visual Studio's C++ build tools (the free "Build Tools for Visual
Studio" with the "Desktop development with C++" workload).

The source is PyInstaller's sdist, the archive uv.lock pins (its SHA-256 is checked). The
result goes to build/bootloader/<version>/; packaging/build.py puts it in place before
each build (and says which bootloader it used).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "bootloader"
FILES = ("run.exe", "runw.exe", "run_d.exe", "runw_d.exe")


def locked_sdist(version: str) -> tuple[str, str]:
    """(url, sha256) of PyInstaller's sdist as uv.lock pins it."""
    text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    block = re.search(
        r'\[\[package\]\]\nname = "pyinstaller"\nversion = "' + re.escape(version) + r'"(.*?)\n\n',
        text,
        re.S,
    )
    m = (
        re.search(r'sdist = \{ url = "([^"]+)", hash = "sha256:([0-9a-f]{64})"', block.group(1))
        if block
        else None
    )
    if m is None:
        raise SystemExit(f"uv.lock pins no sdist of pyinstaller {version}")
    return m.group(1), m.group(2)


def main() -> int:
    import PyInstaller

    version = PyInstaller.__version__
    url, sha = locked_sdist(version)
    work = OUT / "src" / version
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    archive = work / Path(url).name
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(archive, "wb") as f:
        shutil.copyfileobj(resp, f)
    got = hashlib.sha256(archive.read_bytes()).hexdigest()
    if got != sha:
        raise SystemExit(f"sha256 mismatch: {got} (uv.lock pins {sha})")
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")
    src = next(work.glob("pyinstaller-*/bootloader"))
    print("compiling the bootloader (Visual Studio's C++ build tools)...")
    subprocess.run(
        [sys.executable, "waf", "distclean", "all", "--target-arch=64bit"], cwd=src, check=True
    )
    built = src.parent / "PyInstaller" / "bootloader" / "Windows-64bit-intel"
    dest = OUT / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(built / name, dest / name)
        digest = hashlib.sha256((dest / name).read_bytes()).hexdigest()
        print(f"  {dest / name}  sha256 {digest[:16]}...")
    shutil.rmtree(work, ignore_errors=True)
    print("done: packaging/build.py uses it from now on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
