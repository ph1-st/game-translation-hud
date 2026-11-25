"""The model that runs on this computer: downloaded once, kept only when it is the pinned
file, continued where a download stopped, unpacked inside its folder; nothing runs here."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import zipfile

import pytest

from tests.test_sync import server  # noqa: F401 - (the fixture)
from translate import StopError, TranslationError


def pinned(root, name: str, data: bytes):  # noqa: ANN001, ANN201
    """A file on the test server, and what pins it."""
    from translate.local import Download

    (root / name).write_bytes(data)
    return Download(f"/{name}", hashlib.sha256(data).hexdigest(), len(data))


def test_a_download_is_kept_only_when_it_is_the_pinned_file(server, tmp_path):  # noqa: F811
    from translate.local import Download, fetch

    root, base = server
    item = pinned(root, "model.gguf", os.urandom(300_000))
    item = Download(base + item.url, item.sha256, item.size)
    seen: list[int] = []
    fetch(item, tmp_path / "model.gguf", seen.append)
    assert (tmp_path / "model.gguf").read_bytes() == (root / "model.gguf").read_bytes()
    assert seen[-1] == item.size
    wrong = Download(item.url, "0" * 64, item.size)
    with pytest.raises(TranslationError, match="not the one expected"):
        fetch(wrong, tmp_path / "other.gguf")
    assert not (tmp_path / "other.gguf").exists() and not (tmp_path / "other.gguf.part").exists()


def test_a_stopped_download_goes_on_where_it_stopped(server, tmp_path):  # noqa: F811
    from translate.local import Cancelled, Download, fetch

    root, base = server
    data = os.urandom(3 << 20)
    item = pinned(root, "big.gguf", data)
    item = Download(base + item.url, item.sha256, item.size)
    stop = threading.Event()

    def halfway(done: int) -> None:
        if done >= len(data) // 2:
            stop.set()

    with pytest.raises(Cancelled):
        fetch(item, tmp_path / "big.gguf", halfway, stop)
    part = tmp_path / "big.gguf.part"
    assert 0 < part.stat().st_size < len(data)
    fetch(item, tmp_path / "big.gguf")  # (this server ignores the range: all comes again)
    assert (tmp_path / "big.gguf").read_bytes() == data and not part.exists()


def test_the_servers_archive_unpacks_inside_its_folder(tmp_path):
    from translate.local import _unpack

    good = tmp_path / "server.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("llama-server.exe", b"MZ")
        z.writestr("ggml-vulkan.dll", b"MZ")
    _unpack(good, tmp_path / "llama.cpp")
    assert (tmp_path / "llama.cpp" / "llama-server.exe").read_bytes() == b"MZ"
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../outside.exe", b"MZ")
    with pytest.raises(TranslationError, match="not as expected"):
        _unpack(bad, tmp_path / "other")
    assert not (tmp_path / "outside.exe").exists()


def test_install_brings_the_server_and_the_model_once(server, tmp_path, monkeypatch):  # noqa: F811
    import translate.local as local

    root, base = server
    exe = local.SERVER_EXE
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(exe, b"MZ")
    srv = pinned(root, "llama.zip", archive.getvalue())
    mdl = pinned(root, "tiny.gguf", b"GGUF" + os.urandom(1000))
    monkeypatch.setattr(local, "SERVERS", {sys.platform: local.Download(
        base + srv.url, srv.sha256, srv.size)})
    monkeypatch.setattr(local, "MODELS", {"tiny": local.LocalModel(
        "tiny", "Tiny", "", local.Download(base + mdl.url, mdl.sha256, mdl.size))})
    models = local.LocalModels(tmp_path / "models")
    assert not models.installed("tiny") and models.download_size("tiny") == srv.size + mdl.size
    with pytest.raises(StopError, match="not installed yet"):
        local.shared_server(models, "tiny")
    totals: list[tuple[int, int]] = []
    terms = tmp_path / "license.txt"
    terms.write_text("PLaMo Community License Agreement", encoding="utf-8")
    models.install("tiny", lambda done, total: totals.append((done, total)), license_file=terms)
    assert models.installed("tiny") and models.download_size("tiny") == 0
    assert (models.folder / local.LICENSE_FILE).read_text(encoding="utf-8") == terms.read_text(
        encoding="utf-8")  # (the license the player accepted, kept beside the model)
    assert totals[-1] == (srv.size + mdl.size, srv.size + mdl.size)
    assert models.server_exe().name == exe and not list(models.folder.glob("*.zip"))
    models.remove("tiny")
    assert not models.installed("tiny") and models.server_exe() is None
