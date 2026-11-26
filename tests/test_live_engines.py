"""Live translation's engines: the one the settings ask for (PLaMo on this computer, or the
player's own key for Claude or ChatGPT), and the panel that chooses it."""

from __future__ import annotations

import pytest

from app.config import Settings


def test_the_engine_the_settings_ask_for(tmp_path, monkeypatch) -> None:
    import app.keys as keys
    from app.session import make_backend
    from translate import ClaudeBackend, OpenAIBackend, StopError, TranslationError

    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    store = {"openai": "sk-test", "claude": "sk-ant-test"}
    monkeypatch.setattr(keys, "get_key", lambda service="claude": store.get(service, ""))
    s = Settings()
    s.live.openai_model, s.live.claude_model = "gpt-test", "claude-sonnet-5-5"
    chatgpt = make_backend(s, "openai")
    assert isinstance(chatgpt, OpenAIBackend) and chatgpt.model == "gpt-test"
    claude = make_backend(s, "claude")
    assert isinstance(claude, ClaudeBackend) and claude.model == "claude-sonnet-5-5"
    store.pop("openai")
    with pytest.raises(TranslationError, match="no OpenAI API key"):
        make_backend(s, "openai")
    from translate import CompatibleBackend, DeepLBackend

    with pytest.raises(TranslationError, match="no DeepL API key"):
        make_backend(s, "deepl")
    store["deepl"] = "dl-key:fx"
    deepl = make_backend(s, "deepl")
    assert isinstance(deepl, DeepLBackend) and deepl.url.startswith("https://api-free")
    with pytest.raises(TranslationError, match="no server address"):
        make_backend(s, "custom")
    s.live.custom_url, s.live.custom_model = "http://localhost:11434/v1/", "qwen3:8b"
    other = make_backend(s, "custom")  # (no key: a server on this computer takes none)
    assert isinstance(other, CompatibleBackend) and other.api_key == ""
    assert (other.url, other.model) == ("http://localhost:11434/v1", "qwen3:8b")
    with pytest.raises(StopError, match="not installed yet"):  # (nothing downloaded here)
        make_backend(s, "plamo")


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from app.panel import MainWindow
    from app.theme import apply_theme

    apply_theme(app)
    s = Settings()
    s.live.engine = "claude"  # (any engine: live translation is off)
    win = MainWindow(s, desktop=False)
    for w in (win, win.handle):
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    yield win
    win.close_down()
    win.deleteLater()


def test_the_panel_chooses_chatgpt_and_keeps_its_key(panel, monkeypatch) -> None:
    import app.panel.cards as mw

    keys: dict[str, str] = {}
    monkeypatch.setattr(mw, "get_key", lambda service="claude": keys.get(service, ""))
    monkeypatch.setattr(mw, "set_key", lambda value, service="claude": keys.__setitem__(
        service, value) or "the test store")
    live = panel.controls.live
    assert live.engine.currentData() == "off" and not panel.settings.live.enabled
    assert live.key_row.isHidden() and live.local_row.isHidden()
    live.engine.setCurrentIndex(live.engine.findData("openai"))
    live._engine_changed()
    assert panel.settings.live.enabled and panel.settings.live.engine == "openai"
    assert not live.key_row.isHidden() and "OpenAI" in live.key.placeholderText()
    assert live.model.isEditable() and live.model.currentText() == panel.settings.live.openai_model
    live.key.setText("sk-secret")
    live._save_key()
    assert keys == {"openai": "sk-secret"} and "Key saved" in live.state.text()
    assert live.key.text() == ""  # (never shown again)
    live.engine.setCurrentIndex(live.engine.findData("off"))
    live._engine_changed()
    assert not panel.settings.live.enabled and keys == {"openai": "sk-secret"}


def test_the_panel_offers_plamo_with_one_install_button(panel) -> None:
    from translate.local import MODELS

    live = panel.controls.live
    live.engine.setCurrentIndex(live.engine.findData("plamo"))
    live._engine_changed()
    assert panel.settings.live.engine == "plamo" and panel.settings.live.plamo_model == "2b"
    assert not live.local_row.isHidden() and live.key_row.isHidden()
    assert not live.install_btn.isHidden() and live.test_btn.isHidden()  # nothing to test yet
    assert "Built with PLaMo" in live.plamo_note.text()
    assert "Not installed yet" in live.state.text() and "GB" in live.state.text()
    assert [live.local_model.itemData(i) for i in range(live.local_model.count())] == list(
        MODELS)
    live.local_model.setCurrentIndex(live.local_model.findData("0.8b"))
    live._local_model_changed()
    assert panel.settings.live.plamo_model == "0.8b" and "MB" in live.state.text()


def test_installing_plamo_asks_to_accept_its_license_first(panel, monkeypatch) -> None:
    """Install shows PLaMo's own license text (the age and the business terms said above
    it); declined, nothing is downloaded."""
    from app.dialogs import LicenseDialog

    shown: list[LicenseDialog] = []

    def declined(dialog: LicenseDialog) -> int:
        shown.append(dialog)
        return LicenseDialog.DialogCode.Rejected

    monkeypatch.setattr(LicenseDialog, "exec", declined)
    live = panel.controls.live
    live.engine.setCurrentIndex(live.engine.findData("plamo"))
    live._engine_changed()
    live._install()
    (dialog,) = shown
    from PySide6.QtWidgets import QLabel, QPlainTextEdit

    text = dialog.findChild(QPlainTextEdit).toPlainText()
    intro = dialog.findChild(QLabel).text()
    assert text.startswith("PLaMo Community License Agreement") and "Article 5" in text
    assert "18 or older" in intro and "registration" in intro
    assert live.task is None  # declined: nothing downloads


def test_the_panel_plugs_in_another_model_by_its_servers_address(panel, monkeypatch) -> None:
    from translate import SERVERS

    live = panel.controls.live
    live.engine.setCurrentIndex(live.engine.findData("custom"))
    live._engine_changed()
    s = panel.settings.live
    assert s.enabled and s.engine == "custom"
    assert not live.server_row.isHidden() and not live.key_row.isHidden()
    assert live.local_row.isHidden() and live.model.isEditable()
    assert "usually needs no key" in live.state.text()
    live._server_chosen(SERVERS["Ollama"])
    assert s.custom_url == "http://localhost:11434/v1"
    live.model.setEditText("qwen3:8b")
    live._model_changed()
    assert s.custom_model == "qwen3:8b"
    live._listed["custom"] = ["gemma3:12b", "qwen3:8b"]
    live.refresh()
    assert [live.model.itemText(i) for i in range(live.model.count())] == [
        "qwen3:8b", "gemma3:12b"]
    live._server_chosen(SERVERS["LM Studio"])  # another server: its own models
    assert "custom" not in live._listed and s.custom_model == "qwen3:8b"


def test_the_panel_offers_deepl_with_a_key_and_no_model(panel) -> None:
    live = panel.controls.live
    live.engine.setCurrentIndex(live.engine.findData("deepl"))
    live._engine_changed()
    assert panel.settings.live.engine == "deepl"
    assert not live.key_row.isHidden() and live.model.isHidden()
    assert live.server_row.isHidden() and "DeepL" in live.key.placeholderText()
