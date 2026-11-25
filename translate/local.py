"""The translation model that runs on this computer: PLaMo 3 Translate (Preferred Networks),
run by llama.cpp's server.

Both are downloaded once, when the player asks for them, into the app's data folder
(``models/``): the server from llama.cpp's own release, the model from Preferred Networks'
page on Hugging Face. Each is pinned to one version and checked against its SHA-256
before it is used; a download that stops goes on where it stopped next time. The server
runs hidden, reachable from this computer only (127.0.0.1), while live translation is on,
and it is tied to the app: it ends when the app ends, however the app ends.

The models are under the PLaMo Community License (https://plamo.preferredai.jp/info/
plamo-community-license-en; the text the app ships is the one at the pinned models,
``app/assets/plamo_license.txt``): the player accepts it before the download, and a copy is
kept beside the models. "Built with PLaMo" is said where it is used, and a database it
translated says that the license applies to models trained on it (:data:`OUTPUT_NOTICE`).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from i18n import N_, _
from translate.base import StopError, TranslationError

log = logging.getLogger("translate")

USER_AGENT = "GameTranslationHUD (local model download)"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Download:
    url: str
    sha256: str
    size: int  # bytes


@dataclass(frozen=True)
class LocalModel:
    key: str
    name: str
    note: str  # what it is good for (N_: shown translated)
    file: Download

    @property
    def filename(self) -> str:
        return self.file.url.rsplit("/", 1)[-1]


def _hf(repo: str, revision: str, name: str, sha256: str, size: int) -> Download:
    return Download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", sha256, size)


MODELS: dict[str, LocalModel] = {
    "0.8b": LocalModel("0.8b", "PLaMo 3 Translate 0.8B", N_("Smallest and fastest, for weaker "
                                                         "computers"), _hf(
        "pfnet/plamo-3-translate-0.8b-GGUF", "7de4409f84a9d70e3a5e87b81e84883d49f77b58",
        "plamo-3-translate-0.8b-Q4_K_M.gguf",
        "70891ec70dc7ab4205745a38307867c2881b99239f20327927633abfe4bc9ec6", 573_126_848)),
    "2b": LocalModel("2b", "PLaMo 3 Translate 2B", N_("Recommended: fast, good dialogue"), _hf(
        "pfnet/plamo-3-translate-2b-GGUF", "caf0267797e8a4ae418e7d8c480746be2ea6040c",
        "plamo-3-translate-2b-Q4_K_M.gguf",
        "1ffd9073afcc70e35ccc942428aa2b73d7aa2b9ea779bf504b775369e0276c13", 1_807_203_040)),
    "8b": LocalModel("8b", "PLaMo 3 Translate 8B", N_("Best, for a graphics card with 8 GB or "
                                                      "more"), _hf(
        "pfnet/plamo-3-translate-8b-GGUF", "724c63780fbdbd821c86d7b3378a630b10176246",
        "plamo-3-translate-8b-Q4_K_M.gguf",
        "94e5d3f91b421f7c8900ddaac27b11a19f7441426c954bbb25a2fd60a1640187", 5_280_445_728)),
}
DEFAULT_LOCAL_MODEL = "2b"
LICENSE_FILE = "PLaMo Community License.txt"  # (the license, beside the models)
# what a database PLaMo translated says about itself (the license's Art. 5(4))
OUTPUT_NOTICE = ("Translated with PLaMo 3 Translate (Built with PLaMo). The PLaMo Community "
                 "License applies to any model trained using these translations.")

# llama.cpp's server, one pinned release (the Vulkan build: any graphics card, and the
# processor when there is none)
LLAMA_CPP = "b12359"
_RELEASE = (f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP}/"
            f"llama-{LLAMA_CPP}-bin-")
SERVERS: dict[str, Download] = {
    "win32": Download(_RELEASE + "win-vulkan-x64.zip",
                      "e8d18c6e4210cca87e7307c2ea78363ac8d158e4e1397358ec96ab70b53a0d5e",
                      38_090_643),
    "linux": Download(_RELEASE + "ubuntu-vulkan-x64.tar.gz",
                      "19b60f00395a3f4954e46a5037695bc1d843404fc9d3124f82a612ecd8fbe9f0",
                      36_153_377),
}
SERVER_EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"


class Cancelled(Exception):
    """The player stopped a download (what came in is kept for next time)."""


# ---- what is on this computer -----------------------------------------------------------
class LocalModels:
    """The models folder: what is installed, installing it, removing it."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    def model_path(self, key: str) -> Path:
        return self.folder / MODELS[key].filename

    def server_dir(self) -> Path:
        return self.folder / f"llama.cpp-{LLAMA_CPP}"

    def server_exe(self) -> Path | None:
        found = sorted(self.server_dir().rglob(SERVER_EXE)) if self.server_dir().is_dir() else []
        return found[0] if found else None

    def installed(self, key: str) -> bool:
        return key in MODELS and self.model_path(key).is_file() and self.server_exe() is not None

    def download_size(self, key: str) -> int:
        """Bytes still to download for *key* (the server too, when it is not there yet)."""
        size = 0 if self.model_path(key).is_file() else MODELS[key].file.size
        server = SERVERS.get(sys.platform)
        if self.server_exe() is None and server is not None:
            size += server.size
        return size

    def install(self, key: str, progress: Callable[[int, int], None] | None = None,
                cancel: threading.Event | None = None, license_file: Path | None = None
                ) -> None:
        """Download what *key* still needs: the server (unpacked), then the model; the
        model's *license_file* (the one the player accepted) is kept beside it. Raises
        :class:`Cancelled`, or TranslationError with what went wrong."""
        server = SERVERS.get(sys.platform)
        if server is None:
            raise StopError(_("The local model does not run on this system yet."))
        self.folder.mkdir(parents=True, exist_ok=True)
        if license_file is not None:
            shutil.copyfile(license_file, self.folder / LICENSE_FILE)
        total, base = self.download_size(key), 0

        def step(done: int) -> None:
            if progress is not None:
                progress(base + done, total)

        if self.server_exe() is None:
            archive = self.folder / server.url.rsplit("/", 1)[-1]
            fetch(server, archive, step, cancel)
            _unpack(archive, self.server_dir())
            archive.unlink(missing_ok=True)
            base += server.size
        if not self.model_path(key).is_file():
            fetch(MODELS[key].file, self.model_path(key), step, cancel)

    def remove(self, key: str) -> None:
        """Delete *key*'s model (the server goes with the last model)."""
        stop_servers()
        self.model_path(key).unlink(missing_ok=True)
        if not any(self.model_path(k).is_file() for k in MODELS):
            shutil.rmtree(self.server_dir(), ignore_errors=True)


def fetch(item: Download, dest: Path, progress: Callable[[int], None] | None = None,
          cancel: threading.Event | None = None, timeout: float = 30.0) -> None:
    """Download *item* to *dest* through ``<dest>.part`` (a part left by a download that
    stopped is continued), and keep it only if its SHA-256 is the pinned one."""
    import requests

    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    done = part.stat().st_size if part.is_file() else 0
    if done > item.size:
        part.unlink()
        done = 0
    if done:
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                digest.update(chunk)
    headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={done}-"} if done else {})}
    try:
        if done < item.size:
            with requests.get(item.url, stream=True, timeout=(10, timeout),
                              headers=headers) as resp:  # fmt: skip
                if resp.status_code == 200 and done:  # (the host sends it all again)
                    digest, done = hashlib.sha256(), 0
                elif resp.status_code not in (200, 206):
                    raise TranslationError(_("The download failed (HTTP {status}).").format(
                        status=resp.status_code))
                with open(part, "ab" if done else "wb") as f:
                    for chunk in resp.iter_content(CHUNK):
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        f.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done)
    except requests.RequestException as e:
        raise TranslationError(_("The download stopped ({error}): try again, it goes on "
                                 "where it stopped.").format(error=type(e).__name__)) from e
    if done != item.size or digest.hexdigest() != item.sha256:
        part.unlink(missing_ok=True)
        raise TranslationError(_("The downloaded file is not the one expected (it was damaged "
                                 "on the way): try again."))
    os.replace(part, dest)


def _unpack(archive: Path, dest: Path) -> None:
    """Unpack the server's archive (zip, or tar.gz) into *dest*: nothing may land outside
    it."""
    import tarfile

    tmp = dest.with_name(dest.name + ".unpack")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    root = tmp.resolve()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            if not all((tmp / n).resolve().is_relative_to(root) for n in z.namelist()):
                raise TranslationError(_("The server's archive is not as expected."))
            z.extractall(tmp)
    else:
        try:
            with tarfile.open(archive, "r:gz") as t:
                t.extractall(tmp, filter="data")  # (nothing outside it, no device, no link out)
        except tarfile.FilterError as e:
            raise TranslationError(_("The server's archive is not as expected.")) from e
    if sys.platform != "win32":
        for exe in tmp.rglob(SERVER_EXE):
            exe.chmod(0o755)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


# ---- the running server -----------------------------------------------------------------
class LocalServer:
    """llama.cpp's server running one model on 127.0.0.1: started on the first request,
    on the graphics card when it can be (else the processor), stopped by :meth:`stop`."""

    def __init__(self, exe: Path, model: Path, log_file: Path, context: int = 4096,
                 slots: int = 4) -> None:
        self.exe, self.model, self.log_file = exe, model, log_file
        self.context, self.slots = context, slots
        self.url = ""
        self.gpu = False
        self._proc: subprocess.Popen | None = None
        self._job = None  # (Windows: the job object that ends the server with the app)
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float = 120.0) -> str:
        """The server's URL, once it answers (started if it is not running). Raises
        TranslationError when it does not start."""
        with self._lock:
            if self.running and self.url:
                return self.url
            for gpu in (True, False):
                port = _free_port()
                self._spawn(port, gpu)
                url = f"http://127.0.0.1:{port}"
                if self._ready(url, timeout):
                    self.url, self.gpu = url, gpu
                    log.info("local model running on %s (%s)", url, "GPU" if gpu else "CPU")
                    return url
                self._end()
            raise TranslationError(_("The local model did not start. Its log is in {path}.")
                                   .format(path=self.log_file))

    def stop(self) -> None:
        with self._lock:
            self._end()

    def _spawn(self, port: int, gpu: bool) -> None:
        args = [str(self.exe), "--model", str(self.model), "--host", "127.0.0.1",
                "--port", str(port), "--ctx-size", str(self.context),
                "--parallel", str(self.slots), "--n-gpu-layers", "999" if gpu else "0",
                "--no-webui"]
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        out = open(self.log_file, "ab")  # noqa: SIM115 - kept open while the server runs
        try:
            kw: dict = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT,
                        "cwd": str(self.exe.parent)}
            if sys.platform == "win32":
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                kw["preexec_fn"] = _die_with_parent
            self._proc = subprocess.Popen(args, **kw)  # noqa: S603 - our own pinned program
        finally:
            out.close()  # (the server has its own handle)
        if sys.platform == "win32":
            self._job = _tie_to_app(self._proc)

    def _ready(self, url: str, timeout: float) -> bool:
        import requests

        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._proc is None or self._proc.poll() is not None:
                return False
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.25)
        return False

    def _end(self) -> None:
        proc, self._proc, self.url = self._proc, None, ""
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._job is not None:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(self._job)
            self._job = None


_SERVERS: dict[Path, LocalServer] = {}
_SERVERS_LOCK = threading.Lock()


def shared_server(models: LocalModels, key: str) -> LocalServer:
    """The one server for *key* (any other model's server is stopped first: one model at a
    time on the graphics card). Raises StopError when *key* is not installed."""
    exe = models.server_exe()
    if not models.installed(key) or exe is None:
        raise StopError(_("The local model is not installed yet: choose it under Live "
                          "translation and press Install."))
    path = models.model_path(key)
    with _SERVERS_LOCK:
        for other in [p for p in _SERVERS if p != path]:
            _SERVERS.pop(other).stop()
        server = _SERVERS.get(path)
        if server is None:
            server = _SERVERS[path] = LocalServer(exe, path, models.folder / "server.log")
        return server


def stop_servers() -> None:
    """Stop every local model server (live translation off, the app closing)."""
    with _SERVERS_LOCK:
        servers = list(_SERVERS.values())
        _SERVERS.clear()
    for s in servers:
        s.stop()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _die_with_parent() -> None:  # pragma: no cover - runs in the child, on Linux
    """The server gets SIGTERM when the app ends (Linux)."""
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def _tie_to_app(proc: subprocess.Popen):  # noqa: ANN202 - a job object handle, or None
    """Put the server in a job object that ends it when the app's handle to the job closes
    (Windows: also when the app is killed)."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                            wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) and \
        k32.AssignProcessToJobObject(job, int(proc._handle))  # noqa: SLF001
    if not ok:
        log.warning("the local model server is not tied to the app (error %d)",
                    ctypes.get_last_error())
    return job
