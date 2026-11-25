# PyInstaller spec: the app as a folder with its executable in it (Windows and Linux).
#
#   uv run python packaging/build.py
#
# A folder (not one single file) starts in a second instead of unpacking ~300 MB each time,
# and keeps Qt's libraries replaceable (LGPL); the build script zips it for sharing (and
# makes an AppImage on Linux).
#
# The OCR models come from the build script (GTHUD_OCR_MODELS): exactly the ones the
# built-in profiles use, downloaded and checked there, so the app never fetches one.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 - set by PyInstaller
WINDOWS = sys.platform == "win32"

models = [m for m in os.environ.get("GTHUD_OCR_MODELS", "").split(os.pathsep) if m]
if not models:
    raise SystemExit("no OCR models given: build with packaging/build.py")

# Game files kept beside a profile (an install or archive to extract text from) never ship.
GAME_FILES = {"compressed", "gamefiles"}
APP_ID = os.environ.get("GTHUD_APP_ID", "GameTranslationHUD")
EDITION_FILE = os.environ.get("GTHUD_EDITION_FILE", "")
ONE_GAME = ""  # an app made for one game ships that game's preset only
if EDITION_FILE:
    with open(EDITION_FILE, encoding="utf-8") as f:
        _edition = f.read().strip()
    ONE_GAME = "" if _edition == "general" else _edition


def tree(rel: str) -> list[tuple[str, str]]:
    """Every file under ROOT/rel, at the same place in the bundle."""
    out = []
    for folder, _subdirs, files in os.walk(os.path.join(ROOT, rel)):
        out += [(os.path.join(folder, f), os.path.relpath(folder, ROOT)) for f in files]
    return out


datas = []
for _dir, _subdirs, _files in os.walk(os.path.join(ROOT, "games")):  # profiles, glossaries...
    _subdirs[:] = [d for d in _subdirs if d not in GAME_FILES]
    if _dir == os.path.join(ROOT, "games") and ONE_GAME:
        _subdirs[:] = [d for d in _subdirs if d in (ONE_GAME, "_generic")]
    datas += [(os.path.join(_dir, f), os.path.relpath(_dir, ROOT)) for f in _files]
if EDITION_FILE:
    datas.append((EDITION_FILE, "."))
datas += [
    (os.path.join(ROOT, "app", "assets"), os.path.join("app", "assets")),
    (os.path.join(ROOT, "docs", "USER_GUIDE.md"), "docs"),
]
datas += [d for d in tree(os.path.join("i18n", "locale")) if d[0].endswith(".po")]
datas += tree(os.path.join("assets", "fonts"))  # the HUD's fonts and their licences
# rapidocr's configs, without the models its wheel carries (other versions than ours)
datas += collect_data_files("rapidocr", excludes=["**/models/**"])
datas += [(m, os.path.join("rapidocr", "models")) for m in models]
datas += collect_data_files("anthropic")

binaries = collect_dynamic_libs("onnxruntime")  # ONNX Runtime (+ DirectML on Windows)

hiddenimports = (
    collect_submodules("rapidocr")
    + collect_submodules("keyring.backends")
    + collect_submodules("translate")
    + collect_submodules("sync")
    + ["i18n", "hud.platform.screen", "mss", "psutil"]
)
if WINDOWS:
    hiddenimports += ["hud.platform.windows", "win32ctypes.core", "dxcam"]
else:  # (the platform is picked at run time: PyInstaller cannot see the import)
    hiddenimports += ["hud.platform.x11"] + collect_submodules("secretstorage") \
        + collect_submodules("jeepney")

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "run_app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter", "matplotlib", "pytest", "IPython", "imageio_ffmpeg", "torch",
        "setuptools", "pkg_resources", "tools",
        "PySide6.QtWebEngineCore", "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml",
    ],
    noarchive=False,
)

# What the app never uses: video decoding (it reads the screen), software OpenGL (the
# overlay paints with the CPU), Qt's TLS plugins (its network module only serves the app's
# own wake-up channel; downloads go through Python's ssl, whose OpenSSL stays) and
# translations of Qt's dialogs, and the game folders' notes for the test footage.
DROP = (
    "cv2/opencv_videoio_ffmpeg",
    "pyside6/opengl32sw.dll",
    "pyside6/translations/",
    "pyside6/plugins/tls/",
    "pyside6/plugins/networkinformation/",
    "/sources/",
)


def _keep(entry) -> bool:
    name = entry[0].replace("\\", "/").lower()
    return not any(d in name or name.startswith(d) for d in DROP)


a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

if WINDOWS:
    # Python's ssl module with Python's own OpenSSL: PyInstaller finds DLLs of these names
    # on PATH too (Git for Windows has its own), and the first one found would ship. (uv's
    # Linux Pythons link OpenSSL into the ssl module itself.)
    OPENSSL = ("libssl-3-x64.dll", "libcrypto-3-x64.dll", "libssl-3.dll", "libcrypto-3.dll")
    own = [
        (name, os.path.join(sys.base_prefix, "DLLs", name), "BINARY")
        for name in OPENSSL
        if os.path.isfile(os.path.join(sys.base_prefix, "DLLs", name))
    ]
    if not own:
        raise SystemExit(f"no OpenSSL DLLs in {os.path.join(sys.base_prefix, 'DLLs')}")
    a.binaries = [b for b in a.binaries if os.path.basename(b[0]).lower() not in OPENSSL] + own

pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_ID,
    icon=os.path.join(ROOT, "app", "assets", "icon.ico") if WINDOWS else None,
    console=False,
    disable_windowed_traceback=False,
    upx=False,
    version=(os.environ.get("GTHUD_VERSION_FILE") or None) if WINDOWS else None,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_ID,
)
