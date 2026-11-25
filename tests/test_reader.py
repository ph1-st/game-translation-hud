"""The text reader choice: MeikiOCR on CUDA, downloaded only after the player agrees (never
shipped), greyed out without an NVIDIA card; the helper process takes ONNX Runtime from the
downloaded folder, not the app's own build. Nothing here downloads anything."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from app.config import Settings


def _wheel(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def test_a_wheel_gives_only_what_the_reader_needs(tmp_path):
    from app.reader import _unpack_wheel

    w = _wheel(tmp_path / "nvidia_cudnn_cu12-9-py3-none-win_amd64.whl", {
        "nvidia/cudnn/bin/cudnn64_9.dll": b"dll",
        "nvidia/cudnn/include/cudnn.h": b"header",
        "nvidia_cudnn_cu12-9.dist-info/License.txt": b"terms",
        "nvidia_cudnn_cu12-9.dist-info/RECORD": b"record",
        "../../escaped.dll": b"no",
    })  # fmt: skip
    runtime = tmp_path / "runtime"
    _unpack_wheel(w, runtime)
    got = sorted(str(p.relative_to(runtime)).replace("\\", "/") for p in runtime.rglob("*")
                 if p.is_file())  # fmt: skip
    assert got == ["licenses/nvidia_cudnn_cu12/License.txt", "nvidia/cudnn/bin/cudnn64_9.dll"]
    assert not (tmp_path / "escaped.dll").exists()


def test_installing_fetches_each_part_once_and_goes_on_after_a_stop(tmp_path, monkeypatch):
    from app import reader
    from translate.local import Cancelled, Download

    wheels = [
        Download("https://x/onnxruntime_gpu-1-cp311-cp311-win_amd64.whl", "a", 10),
        Download("https://x/nvidia_cudnn_cu12-9-py3-none-win_amd64.whl", "b", 20),
    ]
    contents = {
        "onnxruntime_gpu-1-cp311-cp311-win_amd64.whl": {
            "onnxruntime/__init__.py": b"",
            "onnxruntime/capi/onnxruntime_providers_cuda.dll": b"x",
        },
        "nvidia_cudnn_cu12-9-py3-none-win_amd64.whl": {"nvidia/cudnn/bin/cudnn64_9.dll": b"x"},
    }  # fmt: skip
    fetched: list[str] = []
    stop_at = {"nvidia_cudnn_cu12-9-py3-none-win_amd64.whl"}

    def fake_fetch(item, dest, progress=None, cancel=None):  # noqa: ANN001, ANN202
        name = item.url.rsplit("/", 1)[-1]
        if name in stop_at:
            stop_at.clear()
            raise Cancelled()
        fetched.append(name)
        if name in contents:
            _wheel(dest, contents[name])
        else:
            dest.write_bytes(b"model")

    monkeypatch.setattr(reader, "WHEELS", wheels)
    monkeypatch.setattr(reader, "fetch", fake_fetch)
    r = reader.MeikiReader(tmp_path / "models")
    models = sum(m.size for m in reader.MODELS.values())
    assert not r.installed() and r.download_size() == 30 + models
    with pytest.raises(Cancelled):
        r.install()
    assert not r.installed() and r.download_size() == 20  # what came in is kept
    r.install()
    assert r.installed() and r.download_size() == 0
    assert fetched.count("onnxruntime_gpu-1-cp311-cp311-win_amd64.whl") == 1
    assert "LGPL" in (r.folder / "NOTICE.txt").read_text(encoding="utf-8")
    assert not list(r.folder.glob("*.whl"))  # (the wheels go once unpacked)


def test_the_reader_setting_takes_only_what_it_knows(tmp_path):
    p = tmp_path / "settings.json"
    s = Settings()
    s.ocr_reader = "quantum"
    s.save(p)
    assert Settings.load(p).ocr_reader == "standard"


def test_meiki_cannot_be_chosen_without_an_nvidia_card(tmp_path, monkeypatch):
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    monkeypatch.setattr("app.reader.nvidia_gpu", lambda: False)
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from app.panel.cards import ReaderCard

    settings = Settings()
    settings.ocr_reader = "meiki"  # (chosen on another computer)
    card = ReaderCard(type("Win", (), {"settings": settings})())
    card.refresh()
    i = card.reader.findData("meiki")
    assert not card.reader.model().item(i).isEnabled()
    assert card.reader.currentData() == "standard" and not card.install_btn.isVisible()
    assert "NVIDIA" in card.state.text()


def test_the_helper_takes_onnxruntime_from_the_downloaded_folder(tmp_path):
    fake = tmp_path / "runtime" / "onnxruntime"
    (fake / "capi").mkdir(parents=True)
    (fake / "__init__.py").write_text("WHERE = 'downloaded'\n", encoding="utf-8")
    (fake / "capi" / "__init__.py").write_text("", encoding="utf-8")
    code = (
        "import sys; from pathlib import Path; import onnxruntime;"  # (the app's own, loaded)
        "from capture.cuda_helper import use_runtime; use_runtime(Path(sys.argv[1]));"
        "import onnxruntime, onnxruntime.capi; print(onnxruntime.WHERE, onnxruntime.capi.__file__)"
    )
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "runtime")], cwd=root,
                         capture_output=True, text=True, timeout=120)  # fmt: skip
    assert out.returncode == 0, out.stderr
    where, capi = out.stdout.split()
    assert where == "downloaded" and Path(capi).is_relative_to(tmp_path)


def test_a_misreading_is_read_again_once_and_remembered(monkeypatch):
    import numpy as np

    from capture.ocr import OcrChar, OcrLine
    from capture.profile import Region
    from hud import pipeline

    calls = []

    def fake(engine, frame, boxes):  # noqa: ANN001, ANN202
        calls.append(len(boxes))
        return [("reread", 0.9) for _ in boxes]

    monkeypatch.setattr(pipeline, "recognize_boxes", fake)
    r = pipeline.Rereader(object())
    frame = np.zeros((100, 200, 3), np.uint8)
    line = OcrLine("CD", Region(0.1, 0.1, 0.2, 0.1), 0.6,
                   [OcrChar("C", 0.1, 0.2), OcrChar("D", 0.2, 0.3)])
    other = OcrLine("x", Region(0.5, 0.5, 0.1, 0.1), 0.6)
    got = r(frame, [line, other])
    assert calls == [2] and [g.text for g in got] == ["reread", "reread"]  # one batch
    assert got[0].chars is None  # (six characters for two cells: none kept)
    r(frame, [line])
    assert calls == [2]  # the same misreading in a box of the same size: remembered


def test_a_list_that_moved_takes_its_whole_column():
    from types import SimpleNamespace

    from hud.board import _list_mates

    def cover(x, y):  # noqa: ANN001, ANN202
        return SimpleNamespace(geom=SimpleNamespace(box=(x, y, 120, 26)))

    gone = [cover(320, 183), cover(312, 255), cover(320, 323), cover(319, 763)]
    stayed = cover(319, 395)  # its strokes still passed, over the row that scrolled in
    tab = cover(1023, 973)  # another column: never part of the list
    below = cover(320, 900)  # the same column, but below what moved
    assert _list_mates(gone, gone + [stayed, tab, below]) == [stayed]
    assert _list_mates(gone[:2], gone[:2] + [stayed]) == []  # two covers: no list
