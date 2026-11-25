"""Which text reader the HUD uses, and installing the faster one.

* ``standard``: PP-OCRv4 on ONNX Runtime's DirectML build (any graphics card, or the
  processor), shipped with the app.
* ``meiki``: MeikiOCR, made for Japanese game text, on an NVIDIA graphics card through
  CUDA. About twice as fast; nothing of it ships with the app: ONNX Runtime's CUDA build,
  NVIDIA's libraries and the two models (about 1.9 GB) are downloaded once, after the
  player agrees, into the models folder (capture/cuda_helper.py runs them).

Every download is pinned: the exact file, its size and its SHA-256.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path

from i18n import N_
from translate.local import Download, fetch

STANDARD, MEIKI = "standard", "meiki"
READERS = {
    STANDARD: N_("Standard (any graphics card)"),
    MEIKI: N_("MeikiOCR (NVIDIA, faster)"),
}
FOLDER = "meiki-cuda"  # in the models folder
MEIKI_LICENSE = "https://huggingface.co/rtr46/meiki.text.detect.v0"

_PYPI = "https://files.pythonhosted.org/packages/"
# ONNX Runtime's CUDA build and the NVIDIA libraries it needs (the wheels are zip files:
# the parts that are kept are unpacked, the rest is dropped)
WHEELS: list[Download] = [
    Download(_PYPI + "d2/07/036825cbe30f91ea8574a18a759beccd0ea31b7b71e17f6a9ee9304b51d2/"
             "onnxruntime_gpu-1.24.4-cp311-cp311-win_amd64.whl",
             "1a799a16e5f1ff4d6a9e5f72d750849ab0fe534da8d323ae4a5d8d8bb7daeca8", 207_193_563),
    Download(_PYPI + "59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/"
             "nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
             "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891", 3_591_604),
    Download(_PYPI + "20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
             "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
             "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661", 553_162_896),
    Download(_PYPI + "c7/bd/1d6709d2223e99f04079f16005f8ae6e11a14ee16b5eab6833686cf72fe1/"
             "nvidia_cudnn_cu12-9.28.1.4-py3-none-win_amd64.whl",
             "3467aafa9b260e6adf5fbbfc0e21d180133305b42032aa1160be2d3142bb8b09", 696_300_805),
    Download(_PYPI + "20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/"
             "nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
             "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80", 200_067_309),
    Download(_PYPI + "e5/98/1bd66fd09cbe1a5920cb36ba87029d511db7cca93979e635fd431ad3b6c0/"
             "nvidia_curand_cu12-10.3.10.19-py3-none-win_amd64.whl",
             "e8129e6ac40dc123bd948e33d3e11b4aa617d87a583fa2f21b3210e90c743cde", 68_774_847),
    Download(_PYPI + "52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/"
             "nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
             "72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349", 76_408_187),
    Download(_PYPI + "dd/7e/2eecb277d8a98184d881fb98a738363fd4f14577a4d2d7f8264266e82623/"
             "nvidia_nvjitlink_cu12-12.9.86-py3-none-win_amd64.whl",
             "cc6fcec260ca843c10e34c936921a1c426b351753587fdd638e8cff7b16bb9db", 35_584_936),
]
_HF = "https://huggingface.co/rtr46/"
MODELS: dict[str, Download] = {
    "meiki.text.detect.v0.1.960x544.onnx": Download(
        _HF + "meiki.text.detect.v0/resolve/a9cffa4f60cbf72ddb87edf19c6f98a01cd042e6/"
        "meiki.text.detect.v0.1.960x544.onnx",
        "40b6a016667745cae7d3055929ae3b8b1e7716aac795f5904cd3c2c7c3b8404b", 14_503_825),
    "meiki.text.rec.v0.960x32.onnx": Download(
        _HF + "meiki.txt.recognition.v0/resolve/a28cf5874dc2438ebb1c86336be26bcec51e3375/"
        "meiki.text.rec.v0.960x32.onnx",
        "3e96bc772fbee9717e536a6353032bb944c3382dd2f6960ef4890decda43b000", 18_593_254),
}
NOTICE = """MeikiOCR for Game Translation HUD

models/: MeikiOCR's text detector and recognizer by rtr46, LGPL-3.0
  https://huggingface.co/rtr46/meiki.text.detect.v0
  https://huggingface.co/rtr46/meiki.txt.recognition.v0
  (loaded as they are; replace them with your own build of the same models if you like)
runtime/onnxruntime: ONNX Runtime (CUDA build) by Microsoft, MIT License
runtime/nvidia: NVIDIA CUDA, cuBLAS, cuFFT, cuRAND, NVRTC, nvJitLink and cuDNN runtime
  libraries, under NVIDIA's licenses (runtime/licenses)
Downloaded from PyPI and Hugging Face by the app, after you agreed.
"""


def nvidia_gpu() -> bool:
    """Is there an NVIDIA graphics card with its driver (CUDA's driver library loads)?"""
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        ctypes.WinDLL("nvcuda.dll")
    except OSError:
        return False
    return True


def reader_ready() -> bool:
    """Can MeikiOCR read: an NVIDIA card, and the reader installed?"""
    from app.paths import models_dir

    return nvidia_gpu() and MeikiReader(models_dir()).installed()


class MeikiReader:
    """The models folder's MeikiOCR: what is there, installing it, removing it."""

    def __init__(self, models_dir: Path) -> None:
        self.folder = Path(models_dir) / FOLDER

    @property
    def runtime(self) -> Path:
        return self.folder / "runtime"

    @property
    def models(self) -> Path:
        return self.folder / "models"

    def installed(self) -> bool:
        need = [self.runtime / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll",
                self.runtime / "nvidia" / "cudnn" / "bin" / "cudnn64_9.dll"]
        need += [self.models / name for name in MODELS]
        return all(p.is_file() for p in need) and (self.folder / "done").is_file()

    def download_size(self) -> int:
        """Bytes still to download (all of it, unless a run that stopped left some)."""
        size = sum(d.size for name, d in MODELS.items() if not (self.models / name).is_file())
        done = self._done_wheels()
        return size + sum(w.size for w in WHEELS if _name(w) not in done)

    def install(self, progress: Callable[[int, int], None] | None = None,
                cancel: threading.Event | None = None) -> None:
        """Download what is missing, unpack it, then mark the folder done. Raises
        translate.local.Cancelled, or TranslationError with what went wrong."""
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        (self.folder / "done").unlink(missing_ok=True)
        (self.folder / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
        total, base = self.download_size(), 0

        def step(done: int) -> None:
            if progress is not None:
                progress(base + done, total)

        for name, item in MODELS.items():
            if not (self.models / name).is_file():
                fetch(item, self.models / name, step, cancel)
                base += item.size
        done = self._done_wheels()
        for wheel in WHEELS:
            if _name(wheel) in done:
                continue
            archive = self.folder / _name(wheel)
            fetch(wheel, archive, step, cancel)
            _unpack_wheel(archive, self.runtime)
            archive.unlink(missing_ok=True)
            with open(self.folder / "wheels.txt", "a", encoding="utf-8") as f:
                f.write(_name(wheel) + "\n")
            base += wheel.size
        (self.folder / "done").write_text("ok", encoding="utf-8")

    def remove(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def _done_wheels(self) -> set[str]:
        p = self.folder / "wheels.txt"
        return set(p.read_text(encoding="utf-8").split()) if p.is_file() else set()


def _name(item: Download) -> str:
    return item.url.rsplit("/", 1)[-1]


def _unpack_wheel(archive: Path, runtime: Path) -> None:
    """What the reader needs from a wheel: ONNX Runtime's package, NVIDIA's DLLs, and the
    license files (nothing may land outside *runtime*)."""
    root = runtime.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("onnxruntime/") or (name.startswith("nvidia/") and "/bin/" in name):
                dest = runtime / name
            elif ".dist-info/" in name and ("LICENSE" in name.upper() or "NOTICE" in name.upper()):
                dest = runtime / "licenses" / archive.stem.split("-")[0] / Path(name).name
            else:
                continue
            if name.endswith("/") or not dest.resolve().is_relative_to(root):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
    if sys.platform != "win32":
        for exe in runtime.rglob("*.so*"):
            os.chmod(exe, 0o755)
