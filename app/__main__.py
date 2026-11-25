"""Start the app: python -m app (or the packaged GameTranslationHUD executable)."""

from __future__ import annotations

import faulthandler
import getpass
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

log = logging.getLogger("app")
_crash_file = None  # kept open: faulthandler writes into it when the process crashes
_instance_lock = None  # held for the life of the process (one app per user)
WAKE_TIMEOUT_MS = 1000  # a second start waits this long for the first one to answer
LOG_BYTES, LOG_BACKUPS = 2_000_000, 3


def main() -> int:
    if "--ocr-helper" in sys.argv:  # MeikiOCR on CUDA, for the HUD (capture/cuda_helper.py)
        from capture.cuda_helper import serve

        i = sys.argv.index("--ocr-helper")
        return serve(Path(sys.argv[i + 1]), Path(sys.argv[i + 2]))
    if "--self-test" in sys.argv:  # the build checks every package it makes (no window)
        i = sys.argv.index("--self-test")
        end(self_test(sys.argv[i + 1] if i + 1 < len(sys.argv) else "self-test.json"))

    from app import APP_ID, APP_NAME
    from app.paths import assets_dir, data_dir, logs_dir
    from hud.platform import use_x11_for_qt

    handlers: list[logging.Handler] = [
        RotatingFileHandler(logs_dir() / "app.log", maxBytes=LOG_BYTES,
                            backupCount=LOG_BACKUPS, encoding="utf-8")
    ]  # fmt: skip
    if sys.stdout is not None:  # (no console in the packaged app)
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=handlers
    )
    _log_errors(logs_dir())
    use_x11_for_qt()  # (before any Qt application exists)

    try:
        import cv2
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
    except Exception as e:  # noqa: BLE001 - a damaged install: said, not a silent exit
        log.exception("could not load the app's libraries")
        _fatal(APP_NAME, f"The app could not start: {e}\n\nReinstall it (unzip it again).")
        return 1

    if not _first_instance(data_dir()):  # one app at a time: the first one's panel unfolds
        _wake(_channel_name(APP_ID))
        return 0

    cv2.setNumThreads(2)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(assets_dir() / "icon.png")))
    app.setQuitOnLastWindowClosed(False)  # folded away, it is still running
    server = _listen(_channel_name(APP_ID))

    try:
        import i18n
        from app.config import Settings
        from app.panel import MainWindow
        from app.theme import apply_theme

        settings = Settings.load()
        i18n.set_language(settings.ui_language)  # (before any window says a word)
        apply_theme(app)
        win = MainWindow(settings)
    except Exception as e:  # noqa: BLE001 - said, not a silent exit
        log.exception("could not start")
        _fatal(APP_NAME, f"The app could not start: {e}\n\nThe log has the details: {logs_dir()}")
        return 1

    def show_window() -> None:
        conn = server.nextPendingConnection() if server is not None else None
        if conn is not None:
            conn.readAll()
        win.show_panel()

    if server is not None:
        server.newConnection.connect(show_window)
    win.launch()
    log.info("%s started", APP_NAME)
    code = int(app.exec())
    log.info("%s closed", APP_NAME)
    logging.shutdown()
    end(code)
    return code


# ---- one app at a time -----------------------------------------------------------------
def _first_instance(folder: Path) -> bool:
    """Is no other copy of the app running for this user? (A lock file: two starts at the
    same moment cannot both win; a crashed copy's lock is taken over, its process being
    gone.)"""
    global _instance_lock
    from PySide6.QtCore import QLockFile

    lock = QLockFile(str(folder / "app.lock"))
    lock.setStaleLockTime(0)  # (stale only when the process that took it is gone)
    if not lock.tryLock(0):
        return False
    _instance_lock = lock
    return True


def _channel_name(app_id: str) -> str:
    """The first app's wake-up channel: one per user (Windows' named pipes are seen
    machine-wide, Unix sockets lie in a shared temporary folder)."""
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = "user"
    return f"{app_id}-{''.join(c for c in user if c.isalnum()) or 'user'}"


def _listen(name: str):  # noqa: ANN202 - QLocalServer | None
    """The wake-up channel a second start knocks on. Called while holding the instance
    lock, so a channel left by a crashed copy is no one's and may go."""
    from PySide6.QtNetwork import QLocalServer

    QLocalServer.removeServer(name)
    server = QLocalServer()
    if not server.listen(name):
        log.warning("no wake-up channel (%s): a second start cannot unfold the panel", name)
        return None
    return server


def _wake(name: str) -> None:
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtNetwork import QLocalSocket

        _app = QCoreApplication.instance() or QCoreApplication(sys.argv)
        sock = QLocalSocket()
        sock.connectToServer(name)
        if sock.waitForConnected(WAKE_TIMEOUT_MS):
            sock.write(b"show")
            sock.flush()
            sock.waitForBytesWritten(WAKE_TIMEOUT_MS // 2)
    except Exception:  # noqa: BLE001 - the first one is running either way
        log.exception("could not wake the running app")


# ---- errors ----------------------------------------------------------------------------
def _log_errors(folder: Path) -> None:
    """Every error that would otherwise vanish (the packaged app has no console) goes to
    the log: uncaught ones on any thread, in Qt's event handlers too; a hard crash (in a
    native library) leaves its stack in crash.log."""
    global _crash_file

    def hook(kind, value, tb) -> None:
        if issubclass(kind, KeyboardInterrupt):
            return
        log.error("uncaught error", exc_info=(kind, value, tb))

    def thread_hook(args) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        log.error(
            "uncaught error on thread %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = hook
    threading.excepthook = thread_hook
    try:
        _crash_file = open(folder / "crash.log", "a", encoding="utf-8")  # noqa: SIM115
        faulthandler.enable(_crash_file, all_threads=True)
    except (OSError, RuntimeError, ValueError):
        pass


def _fatal(title: str, text: str) -> None:
    """Tell the player the app cannot start: a Qt message box when Qt works, else the
    system's own (Windows) or the console."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        _app = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, title, text)
        return
    except Exception:  # noqa: BLE001 - Qt may be what failed
        pass
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)  # MB_ICONERROR
    else:
        print(f"{title}: {text}", file=sys.stderr)


def end(code: int) -> None:
    """End the process now. A download, a translation or a text reader still busy on a
    background thread is not waited for (what they finished is saved). On Windows the
    DLLs' clean-up is skipped too: the OCR runtime's crashed while a reader thread was
    still in it."""
    if sys.stdout is not None:
        sys.stdout.flush()
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateProcess(k.GetCurrentProcess(), code)
    os._exit(code)


# ---- the package checks itself ------------------------------------------------------------
def self_test(out: str) -> int:
    """What the app needs, checked inside the package: secure connections, the database,
    the text reader with its bundled models (reading a test picture, no download), Qt, the
    credential store, the built-in profiles, the interface translations and fonts, and the
    game file readers. Writes {check: {ok, info}} as JSON to *out*; returns 0 when every
    check passed.

    Expectations per system: on Windows the reader must run on DirectML and keys must go to
    the Windows credential store (GTHUD_SELFTEST_CPU=1 accepts the CPU, for a build machine
    without a DirectX 12 GPU); elsewhere the CPU reads, and any credential store, or none,
    is fine (keys then go to a private file)."""
    import json

    results: dict[str, dict] = {}
    windows = sys.platform == "win32"
    cpu_ok = not windows or os.environ.get("GTHUD_SELFTEST_CPU") == "1"

    def check(name: str, fn) -> None:
        try:
            results[name] = {"ok": True, "info": str(fn())}
        except Exception as e:  # noqa: BLE001 - the result says what failed
            results[name] = {"ok": False, "info": f"{type(e).__name__}: {e}"}

    for name, fn in (
        ("tls", _check_tls), ("sqlite", _check_sqlite),
        ("onnxruntime", lambda: _check_onnx(cpu_ok)), ("ocr", lambda: _check_ocr(cpu_ok)),
        ("qt", _check_qt), ("credential_store", lambda: _check_keyring(windows)),
        ("files", _check_files), ("recognizers", _check_recognizers),
    ):  # fmt: skip
        check(name, fn)
    Path(out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["ok"] for r in results.values()) else 1


def _check_tls() -> str:
    import ssl

    import certifi

    ssl.create_default_context(cafile=certifi.where())
    want = os.environ.get("GTHUD_EXPECT_OPENSSL")  # (the build's: what should ship)
    if want and ssl.OPENSSL_VERSION != want:
        raise RuntimeError(f"{ssl.OPENSSL_VERSION} shipped, {want} expected")
    return ssl.OPENSSL_VERSION


def _check_sqlite() -> str:
    import sqlite3

    return sqlite3.sqlite_version


def _check_onnx(cpu_ok: bool) -> str:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if not cpu_ok and "DmlExecutionProvider" not in providers:
        raise RuntimeError(f"no DirectML: {providers}")
    return ", ".join(providers)


def _check_ocr(cpu_ok: bool) -> str:
    import cv2
    import numpy as np
    import rapidocr
    import yaml

    from app.paths import builtin_games_dir
    from capture.ocr import AUTO, DML, OcrEngine

    models = sorted(p.name for p in (Path(rapidocr.__file__).parent / "models").glob("*"))
    prof = builtin_games_dir() / "azur-promilia" / "profile.yaml"
    cfg = yaml.safe_load(prof.read_text(encoding="utf-8"))["ocr"]
    img = np.full((80, 360, 3), 255, np.uint8)
    cv2.putText(img, "READY 123", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    engine = OcrEngine.from_profile(cfg, device=AUTO if cpu_ok else DML)
    engine.warm_up()
    if not cpu_ok and engine.device != DML:
        raise RuntimeError(engine.fallback_reason or "OCR did not attach DirectML")
    text = " ".join(ln.text for ln in engine.read(img).lines)
    if "123" not in text:
        raise RuntimeError(f"read {text!r} (models: {models})")
    return f"read {text!r} on {engine.device} with {models}"


def _check_qt() -> str:
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication(sys.argv[:1])
    return QGuiApplication.platformName()


def _check_keyring(windows: bool) -> str:
    import keyring

    name = type(keyring.get_keyring()).__name__
    if windows and name != "WinVaultKeyring":  # (the API keys would go into a file instead)
        raise RuntimeError(f"keyring backend {name}, not the Windows credential store")
    return name


def _check_files() -> str:
    """The guide, the profiles, the interface's translations and (when the build had
    them, GTHUD_EXPECT_FONTS) the fonts the HUD draws with."""
    from app.config import Settings
    from app.library import Library
    from app.paths import bundle_dir

    root = bundle_dir()
    guide = root / "docs" / "USER_GUIDE.md"
    catalogs = sorted(p.stem for p in (root / "i18n" / "locale").glob("*.po"))
    if not catalogs:
        raise RuntimeError("no interface translations (i18n/locale/*.po) in the bundle")
    fonts = sorted(p.name for p in (root / "assets" / "fonts").glob("*.[ot]tf"))
    if os.environ.get("GTHUD_EXPECT_FONTS") == "1" and not fonts:
        raise RuntimeError("no fonts (assets/fonts) in the bundle")
    return (f"guide {guide.stat().st_size} bytes, profiles "
            f"{[p[0] for p in Library(Settings()).profiles()]}, translations {catalogs}, "
            f"fonts {fonts}")


def _check_recognizers() -> str:
    """The recognizers of every language a game may show (Japanese is _check_ocr's)."""
    import rapidocr

    models = {p.name for p in (Path(rapidocr.__file__).parent / "models").glob("*")}
    missing = [m for m in ("ch_PP-OCRv4_rec_mobile.onnx",
                           "chinese_cht_PP-OCRv3_rec_mobile.onnx",
                           "korean_PP-OCRv4_rec_mobile.onnx") if m not in models]
    if missing:
        raise RuntimeError(f"no recognizer for Chinese or Korean: {missing}")
    return f"{len(models)} models"


if __name__ == "__main__":
    sys.exit(main())
