"""Build the app: dist/<app>/ with the executable in it, and an archive of it to share.

    uv run python packaging/build.py [--edition E] [--no-archive] [options]

Windows: dist/GameTranslationHUD/GameTranslationHUD.exe and
dist/GameTranslationHUD-<version>-windows.zip. Linux: dist/GameTranslationHUD/
GameTranslationHUD, dist/GameTranslationHUD-<version>-linux-<arch>.tar.gz and, when
appimagetool is found (on PATH or --appimagetool), an AppImage beside it.

``--edition azur-promilia`` builds the Azur Promilia Translator instead (the same code,
made for that game: dist/AzurPromiliaTranslator/; only its preset ships). The bundle's
edition.txt tells the app which it is.

Before PyInstaller runs, the OCR models the built-in profiles use are downloaded (and
checked against rapidocr's SHA-256 list): exactly these ship, so the app never fetches a
model. Beside the executable go the licence (LICENSE.txt), the licences of everything
bundled (THIRD_PARTY_NOTICES.txt) and a short README.txt. Then the built app runs its
self-test (app/__main__.py).

The Python it runs with is the one bundled: it must carry a supported OpenSSL (downloads
and the Claude API go through it); uv's own Python builds do (``uv python install``).

Windows only: the bootloader (the launcher at the front of the .exe) is one compiled here
when packaging/build_bootloader.py has made it (antivirus programs flag PyInstaller's stock
one, which every PyInstaller app shares, malware included), else PyInstaller's own. Code
signing (the Windows SDK's signtool.exe, timestamped), one of:

    --sign-thumbprint SHA1   a certificate in the Windows certificate store (also one on a
                             USB token or from a cloud signing service that shows there)
    --sign-pfx FILE          a .pfx file; its password in the environment, GTHUD_SIGN_PASSWORD
    --sign-dlib DLL --sign-metadata JSON   Azure Artifact Signing (Microsoft's service)

The .exe is signed (``--sign-all``: every program file in the bundle that carries no
signature of its own too), then the signed .exe runs its self-test.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _edition_arg() -> str:
    """--edition, read before the app's name is imported (the edition sets it)."""
    for i, a in enumerate(sys.argv):
        if a == "--edition" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--edition="):
            return a.split("=", 1)[1]
    return "general"


os.environ["GTHUD_EDITION"] = _edition_arg()
from app import APP_ID, APP_NAME, EDITION, EDITIONS, ONE_GAME, VERSION  # noqa: E402

WINDOWS = sys.platform == "win32"
OUT = ROOT / "dist" / APP_ID
EXECUTABLE = OUT / (f"{APP_ID}.exe" if WINDOWS else APP_ID)
PLATFORM_TAG = "windows" if WINDOWS else f"linux-{platform.machine()}"
FONTS = ROOT / "assets" / "fonts"
_GAME = "Azur Promilia" if ONE_GAME == "azur-promilia" else "Chinese, Japanese and Korean games"
WORK = ROOT / "build" / "pyinstaller"
MIN_OPENSSL = (3, 5)  # the long-term support line (to 2030); older lines are out of support
LICENCE_FILE_RE = re.compile(r"(licen[cs]e|copying|notice|authors)", re.IGNORECASE)
BOOTLOADERS = ("run.exe", "runw.exe", "run_d.exe", "runw_d.exe")
TIMESTAMP_URL = "http://timestamp.digicert.com"  # RFC 3161; the signature outlives the cert
SIGN_BATCH = 40  # files a signtool call signs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", choices=sorted(EDITIONS), default="general",
                    help="which app: the general one, or one made for a single game")
    ap.add_argument("--no-archive", action="store_true", help="no zip / tar.gz / AppImage")
    ap.add_argument(
        "--allow-old-openssl", action="store_true", help="build with this Python's OpenSSL anyway"
    )
    ap.add_argument("--stock-bootloader", action="store_true",
                    help="Windows: PyInstaller's own launcher")  # fmt: skip
    ap.add_argument("--appimagetool", help="Linux: appimagetool (default: on PATH)")
    sign = ap.add_argument_group("code signing (Windows)")
    sign.add_argument("--sign-thumbprint", help="SHA-1 thumbprint of a certificate in the store")
    sign.add_argument("--sign-pfx", help="a .pfx file (password: GTHUD_SIGN_PASSWORD)")
    sign.add_argument("--sign-dlib", help="Azure Artifact Signing: Azure.CodeSigning.Dlib.dll")
    sign.add_argument("--sign-metadata", help="Azure Artifact Signing: its metadata.json")
    sign.add_argument("--sign-all", action="store_true", help="also every unsigned program file")
    sign.add_argument("--timestamp", default=TIMESTAMP_URL, help="RFC 3161 timestamp server")
    sign.add_argument("--sign-url", default="", help="a web page shown with the signature")
    args = ap.parse_args()
    signing = bool(args.sign_thumbprint or args.sign_pfx or args.sign_dlib)
    if signing and not WINDOWS:
        print("code signing is for the Windows build")
        return 1
    if signing and find_signtool() is None:
        print(
            "signtool.exe was not found: install the Windows SDK's signing tools (Visual "
            "Studio Installer, or the Windows SDK installer: 'Signing Tools for Desktop Apps')."
        )
        return 1
    if args.sign_dlib and not args.sign_metadata:
        print("--sign-dlib needs --sign-metadata (the metadata.json of the signing account)")
        return 1
    if ssl.OPENSSL_VERSION_INFO[:2] < MIN_OPENSSL and not args.allow_old_openssl:
        print(
            f"This Python carries {ssl.OPENSSL_VERSION}, which is out of support: the app "
            "would ship it. Build with a uv-managed Python (uv python install 3.11; see "
            ".python-version), or pass --allow-old-openssl."
        )
        return 1
    if WINDOWS:
        print("bootloader:", use_bootloader(stock=args.stock_bootloader))
    models = ocr_models()
    print("OCR models:", ", ".join(p.name for p in models))
    if not build_bundle(models):
        return 1
    if signing and not sign_files(EXECUTABLE, args):
        return 1
    if not self_test(EXECUTABLE):
        return 1
    shutil.copy2(ROOT / "LICENSE", OUT / "LICENSE.txt")
    (OUT / "THIRD_PARTY_NOTICES.txt").write_text(third_party_notices(), encoding="utf-8")
    (OUT / "README.txt").write_text(_readme(), encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and not p.is_symlink())
    print(f"built {EXECUTABLE} ({size / 1e6:.0f} MB in the folder)")
    if args.no_archive:
        return 0
    made = [zip_bundle()] if WINDOWS else [tar_bundle(), *appimage(args.appimagetool)]
    shutil.copy2(ROOT / "docs" / "USER_GUIDE.md", ROOT / "dist" / "USER_GUIDE.md")
    for path in made:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"made {path} ({path.stat().st_size / 1e6:.0f} MB), sha256 {digest}")
    return 0


def build_bundle(models: list[Path]) -> bool:
    """Run PyInstaller on the spec; True when the bundle holds the executable and the
    OCR models."""
    WORK.mkdir(parents=True, exist_ok=True)
    version_file = WORK / "version_info.txt"
    version_file.write_text(_version_info(VERSION), encoding="utf-8")
    edition_file = WORK / "edition.txt"
    edition_file.write_text(EDITION, encoding="utf-8")
    env = {
        **os.environ,
        "GTHUD_OCR_MODELS": os.pathsep.join(str(p) for p in models),
        "GTHUD_VERSION_FILE": str(version_file),
        "GTHUD_EDITION_FILE": str(edition_file),
        "GTHUD_APP_ID": APP_ID,
    }
    spec = ROOT / "packaging" / "GameTranslationHUD.spec"
    cmd = [
        sys.executable, "-m", "PyInstaller", str(spec), "--noconfirm", "--clean",
        "--distpath", str(ROOT / "dist"), "--workpath", str(WORK),
    ]  # fmt: skip
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    if not EXECUTABLE.exists():
        print("no executable built")
        return False
    bundled = OUT / "_internal" / "rapidocr" / "models"
    missing = [p.name for p in models if not (bundled / p.name).is_file()]
    if missing:
        print("the bundle lacks OCR models:", ", ".join(missing))
        return False
    return True


def zip_bundle() -> Path:
    zpath = ROOT / "dist" / f"{APP_ID}-{VERSION}-{PLATFORM_TAG}.zip"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, Path(APP_ID) / p.relative_to(OUT))
    return zpath


def tar_bundle() -> Path:
    """The folder as a .tar.gz (it keeps the executable bit a zip would lose)."""
    tpath = ROOT / "dist" / f"{APP_ID}-{VERSION}-{PLATFORM_TAG}.tar.gz"
    tpath.unlink(missing_ok=True)
    with tarfile.open(tpath, "w:gz") as tar:
        tar.add(OUT, arcname=APP_ID)
    return tpath


DESKTOP_ENTRY = """[Desktop Entry]
Type=Application
Name={name}
Comment=Translations over the text of Chinese, Japanese and Korean games
Exec={app_id}
Icon={app_id}
Categories=Game;Utility;
Terminal=false
"""
APP_RUN = """#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
exec "$here/usr/lib/{app_id}/{app_id}" "$@"
"""


def appimage(tool: str | None) -> list[Path]:
    """The bundle as one AppImage (appimagetool's own runtime is run without FUSE, so it
    works in containers and CI); [] when there is no appimagetool."""
    exe = tool or shutil.which("appimagetool")
    if not exe:
        print("no appimagetool (on PATH or --appimagetool): no AppImage made")
        return []
    appdir = WORK / "AppDir"
    shutil.rmtree(appdir, ignore_errors=True)
    shutil.copytree(OUT, appdir / "usr" / "lib" / APP_ID, symlinks=True)
    (appdir / f"{APP_ID}.desktop").write_text(
        DESKTOP_ENTRY.format(name=APP_NAME, app_id=APP_ID), encoding="utf-8")
    shutil.copy2(ROOT / "app" / "assets" / "icon.png", appdir / f"{APP_ID}.png")
    run = appdir / "AppRun"
    run.write_text(APP_RUN.format(app_id=APP_ID), encoding="utf-8")
    run.chmod(0o755)
    out = ROOT / "dist" / f"{APP_ID}-{VERSION}-{platform.machine()}.AppImage"
    out.unlink(missing_ok=True)
    env = {**os.environ, "ARCH": platform.machine(), "APPIMAGE_EXTRACT_AND_RUN": "1"}
    subprocess.run([exe, str(appdir), str(out)], check=True, env=env)
    return [out]

def use_bootloader(stock: bool = False) -> str:
    """Put the bootloader in place that the Windows build uses: the one compiled here
    (packaging/build_bootloader.py) when there is one, else PyInstaller's own (kept aside
    once, so it can come back)."""
    import PyInstaller

    target = Path(PyInstaller.__file__).parent / "bootloader" / "Windows-64bit-intel"
    ours = ROOT / "build" / "bootloader" / PyInstaller.__version__
    kept = ours / "stock"
    have_ours = all((ours / n).is_file() for n in BOOTLOADERS)
    if have_ours and not kept.is_dir():
        kept.mkdir(parents=True)
        for n in BOOTLOADERS:
            shutil.copy2(target / n, kept / n)
    if have_ours and not stock:
        for n in BOOTLOADERS:
            shutil.copy2(ours / n, target / n)
        return "compiled here (packaging/build_bootloader.py)"
    if kept.is_dir():
        for n in BOOTLOADERS:
            shutil.copy2(kept / n, target / n)
    return "PyInstaller's own (antivirus programs flag it more: packaging/build_bootloader.py)"


def find_signtool() -> Path | None:
    """The newest signtool.exe of the Windows SDK (x64)."""
    base = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    found = list((base / "Windows Kits" / "10" / "bin").glob("10.*/x64/signtool.exe"))

    def version(p: Path) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", p.parent.parent.name))

    return max(found, key=version) if found else None


def has_signature(path: Path) -> bool:
    """Does the program file carry an Authenticode signature (its PE security directory)?"""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        if head[:2] != b"MZ":
            return False
        pe = int.from_bytes(head[0x3C:0x40], "little")
        magic = int.from_bytes(head[pe + 24 : pe + 26], "little")
        dirs = pe + 24 + (112 if magic == 0x20B else 96)  # PE32+ or PE32
        entry = dirs + 4 * 8  # IMAGE_DIRECTORY_ENTRY_SECURITY
        return int.from_bytes(head[entry + 4 : entry + 8], "little") > 0
    except (OSError, ValueError):
        return False


def sign_files(exe: Path, args) -> bool:
    """Sign the .exe (and with --sign-all every program file in the bundle that has no
    signature of its own), timestamped, then check it."""
    tool = find_signtool()
    assert tool is not None
    files = [exe]
    if args.sign_all:
        files += [
            p
            for p in sorted(OUT.rglob("*"))
            if p.suffix.lower() in (".exe", ".dll", ".pyd") and p != exe and not has_signature(p)
        ]
    cmd = [str(tool), "sign", "/fd", "sha256", "/td", "sha256", "/tr", args.timestamp,
           "/d", APP_NAME]  # fmt: skip
    if args.sign_url:
        cmd += ["/du", args.sign_url]
    if args.sign_thumbprint:
        cmd += ["/sha1", args.sign_thumbprint.replace(" ", "")]
    elif args.sign_pfx:
        cmd += ["/f", args.sign_pfx]
        if os.environ.get("GTHUD_SIGN_PASSWORD"):
            cmd += ["/p", os.environ["GTHUD_SIGN_PASSWORD"]]
    else:
        cmd += ["/dlib", args.sign_dlib, "/dmdf", args.sign_metadata]
    print(f"signing {len(files)} file(s) with {tool}")
    for i in range(0, len(files), SIGN_BATCH):
        run = subprocess.run(cmd + [str(f) for f in files[i : i + SIGN_BATCH]])
        if run.returncode != 0:
            print("signing failed (signtool's message is above)")
            return False
    check = subprocess.run([str(tool), "verify", "/pa", "/q", str(exe)])
    if check.returncode != 0:
        print(
            "signed, but Windows does not trust the certificate (a test certificate?): the "
            "app is shown as from an unknown publisher on other PCs"
        )
    else:
        print("signed: the signature verifies")
    return True


def self_test(exe: Path) -> bool:
    """The built app checks itself (app/__main__.py self_test): secure connections, the
    database, reading a test picture with the bundled models, Qt and the rest. Without a
    display (a Linux build machine) Qt runs offscreen; GTHUD_SELFTEST_CPU=1 in the
    environment accepts the CPU on a Windows machine without a DirectX 12 GPU."""
    import json

    report = WORK / "self_test.json"
    report.unlink(missing_ok=True)
    data_dir = WORK / "self_test_data"  # (never the builder's own settings and games)
    env = {**os.environ, "GTHUD_DATA": str(data_dir), "GTHUD_EXPECT_OPENSSL": ssl.OPENSSL_VERSION}
    if any(FONTS.glob("*.[ot]tf")):
        env["GTHUD_EXPECT_FONTS"] = "1"
    if not WINDOWS and not os.environ.get("DISPLAY"):
        env["QT_QPA_PLATFORM"] = "offscreen"
    try:
        run = subprocess.run([str(exe), "--self-test", str(report)], env=env, timeout=180)
        code = run.returncode
    except subprocess.TimeoutExpired:
        print("self-test: the app did not finish within 3 minutes")
        return False
    try:
        results = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(f"self-test: no report (exit code {code})")
        return False
    for name, r in results.items():
        print(f"self-test {'ok  ' if r['ok'] else 'FAIL'} {name}: {r['info']}")
    return code == 0 and all(r["ok"] for r in results.values())

def ocr_models() -> list[Path]:
    """The OCR model files the built-in profiles read with, downloaded now if missing (and
    checked against rapidocr's SHA-256 list either way)."""
    import yaml
    from rapidocr.utils.download_file import DownloadFile

    from capture.ocr import OcrEngine

    used: list[Path] = []
    run = DownloadFile.run  # (bound to the class)

    def recording(_cls, params):
        used.append(Path(params.save_path))
        return run(params)

    DownloadFile.run = classmethod(recording)
    try:
        from translate.languages import OCR

        for prof in sorted((ROOT / "games").glob("*/profile.yaml")):
            cfg = (yaml.safe_load(prof.read_text(encoding="utf-8")) or {}).get("ocr") or {}
            # every language a game may show (a player sets it per game): its recognizer too
            for lang in dict.fromkeys([cfg.get("lang", "japan"), *OCR.values()]):
                OcrEngine.from_profile({**cfg, "lang": lang}, device="cpu").warm_up()
    finally:
        DownloadFile.run = run
    out = sorted(set(used))
    if not out:
        raise SystemExit("no OCR model was loaded: has rapidocr changed how it finds them?")
    return out


def third_party_notices() -> str:
    """The licences of the Python packages the app bundles (its runtime dependencies, all
    the way down), and of Python and the OCR models."""
    from importlib.metadata import PackageNotFoundError, distribution

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    parts = [
        f"{APP_NAME} {VERSION} bundles the software below. Each part keeps its own licence; "
        "their texts follow.\n\nQt (through PySide6) is used under the GNU Lesser General "
        "Public License v3: its libraries are separate files in the _internal folder and may "
        "be replaced; its source is at https://download.qt.io/official_releases/qt/ and "
        "PySide6's at https://code.qt.io/cgit/pyside/pyside-setup.git/.\n\nThe OCR models "
        "(PP-OCRv4 detection and Japanese recognition, converted to ONNX by RapidOCR) come "
        "from PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR), Apache License 2.0.\n"
    ]
    py_licence = Path(sys.base_prefix) / "LICENSE.txt"
    if py_licence.is_file():
        parts.append(_section(f"Python {sys.version.split()[0]}", "PSF-2.0", [py_licence]))
    seen: dict[str, object] = {}
    todo = ["game-translation-hud"]
    while todo:
        name = canonicalize_name(todo.pop())
        if name in seen:
            continue
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        seen[name] = dist
        for req in dist.requires or []:
            r = Requirement(req)
            if r.marker is None or r.marker.evaluate({"extra": ""}):
                todo.append(r.name)
    for name in sorted(seen):
        if name == "game-translation-hud":
            continue
        dist = seen[name]
        meta = dist.metadata
        licence = meta.get("License-Expression") or ""
        if not licence:
            classifiers = [
                c.split("::")[-1].strip()
                for c in meta.get_all("Classifier") or []
                if c.startswith("License ::")
            ]
            licence = ", ".join(classifiers) or (meta.get("License") or "").split("\n")[0]
        files = [
            Path(dist.locate_file(f))
            for f in dist.files or []
            if LICENCE_FILE_RE.search(Path(str(f)).name) and not str(f).endswith((".py", ".pyc"))
        ]
        text = None if files else meta.get("License")
        title = f"{meta['Name']} {meta['Version']}"
        parts.append(_section(title, licence, files, text, meta.get("Home-page")))
    return "\n".join(parts)


def _section(title: str, licence: str, files: list[Path], text: str | None = None,
             home: str | None = None) -> str:  # fmt: skip
    out = ["=" * 78, title, f"Licence: {licence or 'see below'}"]
    if home:
        out.append(f"Home: {home}")
    out.append("=" * 78)
    for f in files:
        try:
            body = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if body:
            out += [f"--- {f.name} ---", body, ""]
    if text and text.strip():
        out += [text.strip(), ""]
    return "\n".join(out) + "\n"


def _version_info(version: str) -> str:
    nums = [int(n) for n in re.findall(r"\d+", version)[:4]]
    t = tuple(nums + [0] * (4 - len(nums)))
    return f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', '{APP_NAME} contributors'),
      StringStruct('FileDescription', '{APP_NAME}'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', '{APP_ID}'),
      StringStruct('LegalCopyright', 'MIT License, see LICENSE.txt'),
      StringStruct('OriginalFilename', '{APP_ID}.exe'),
      StringStruct('ProductName', '{APP_NAME}'),
      StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def _readme() -> str:
    if WINDOWS:
        start = (
            f"Start: {APP_ID}.exe (keep this folder together; it runs from anywhere).\n"
            "Windows may warn that the app is from an unknown publisher (it is not signed): "
            '"More info", then "Run anyway".'
        )
        data = f"%LOCALAPPDATA%\\{APP_ID}"
    else:
        start = (
            f"Start: ./{APP_ID} in this folder (or the AppImage). It runs on X11 and, on a "
            "Wayland desktop, through XWayland; the overlay needs a compositing desktop "
            "(GNOME, KDE Plasma, Xfce with compositing on)."
        )
        data = f"~/.local/share/{APP_ID}"
    return f"""{APP_NAME} {VERSION}

Shows translations on top of the text of {_GAME}, as an overlay. It reads the screen and
draws its own window: it doesn't modify the game, touch its memory or hook the process.
Use it at your own discretion.

{start}

Quick start:
1. Choose a game.
2. Add its .sqlite translation database with "Use a file..." (or paste a shared link).
3. Press Play.
4. Run the game in a window or borderless window (exclusive full screen hides every
   overlay).

Without a database, choose a live-translation provider in the HUD and enter its API key.
The Help link opens the full guide.

Settings and databases are kept in {data}. Removing that folder resets the app and also
removes its local databases.

Licence: MIT (LICENSE.txt). The software it bundles: THIRD_PARTY_NOTICES.txt.
"""

if __name__ == "__main__":
    sys.exit(main())
