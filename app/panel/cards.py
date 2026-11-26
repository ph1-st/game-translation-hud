"""The boxes the HUD panel is made of: what is translated, the dialogue's glass and area,
live translation, diagnostics, and (folded away) the rest of the settings.

Every card reads the settings in :meth:`refresh` and writes them when the player changes
something; the panel's window (``win``: :class:`app.panel.window.MainWindow`) restarts or
reloads the overlay as needed."""

from __future__ import annotations

import json
import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QWidget,
)

from app.config import TEXT_SIZES, GameEntry
from app.keys import get_key, mask, set_key
from app.library import parse_region, region_text
from app.panel.hotkeys import HOTKEYS, HotkeyButton
from app.paths import assets_dir, models_dir
from app.widgets import Task, button, card, combo, link_button, note, tinted
from i18n import N_, _
from translate.backends.compatible import SERVERS
from translate.languages import native_name
from translate.local import MODELS

ENGINES = {  # live translation: what the player picks -> its name ("off": none)
    "off": N_("Off"),
    "plamo": N_("PLaMo, on this computer (free)"),
    "claude": N_("Claude (your API key)"),
    "openai": N_("ChatGPT (your API key)"),
    "deepl": N_("DeepL (your API key)"),
    "custom": N_("Another model (a server you name)"),
}
KEY_HINTS = {
    "claude": N_("Claude API key (sk-ant-…)"),
    "openai": N_("OpenAI API key (sk-…)"),
    "deepl": N_("DeepL API key"),
    "custom": N_("API key, if the server asks for one"),
}
LISTED = ("openai", "custom")  # engines whose models Test lists (the name can be typed)
PLAMO_LICENSE = "https://plamo.preferredai.jp/info/plamo-community-license-en"
GLASS_STYLES = {"grow": N_("Grows with the text"), "window": N_("One window")}
READING_MODES = {
    "balanced": N_("Balanced (recommended)"),
    "instant": N_("Instant (may flicker on hard text)"),
}


def _set_checked(box: QCheckBox, on: bool) -> None:
    box.blockSignals(True)
    box.setChecked(on)
    box.blockSignals(False)


class Card(QWidget):
    """A titled box of the panel."""

    def __init__(self, win, title: str = "") -> None:  # noqa: ANN001 - MainWindow
        super().__init__()
        self.win = win
        frame, self.body = card(title)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

    @property
    def game(self) -> GameEntry | None:
        return self.win.settings.current_game

    def refresh(self) -> None:
        raise NotImplementedError


# ---- what is translated -----------------------------------------------------------------
class TranslateCard(Card):
    def __init__(self, win) -> None:  # noqa: ANN001
        super().__init__(win, _("Translate"))
        row = QHBoxLayout()
        self.story = QCheckBox(_("Dialogue"))
        self.story.setToolTip(_("Dialogue, subtitles, narration and choices"))
        self.menus = QCheckBox(_("Menus"))
        self.menus.setToolTip(_("Menus, labels and other on-screen text"))
        for b in (self.story, self.menus):
            b.toggled.connect(self._changed)
            row.addWidget(b)
        row.addStretch(1)
        self.body.addLayout(row)
        trow = QHBoxLayout()
        trow.addWidget(QLabel(_("Text size")))
        self.size = combo()
        for pct in TEXT_SIZES:
            self.size.addItem(_("Game's size") if pct == 100 else f"{pct} %", pct)
        self.size.setToolTip(_("The translation's size next to the game's text (smaller "
                               "text is squeezed less)"))
        self.size.activated.connect(lambda _i: self._looks_changed())
        trow.addWidget(self.size)
        self.white = QCheckBox(_("White text"))
        self.white.setToolTip(_("Every translation white with a black outline (off: in the "
                                "colours of the game's text)"))
        self.white.toggled.connect(lambda _on: self._looks_changed())
        trow.addWidget(self.white)
        trow.addStretch(1)
        self.body.addLayout(trow)

    def refresh(self) -> None:
        s = self.win.settings
        _set_checked(self.story, s.translate_dialogue)
        _set_checked(self.menus, s.translate_ui)
        _set_checked(self.white, s.white_text)
        self.size.blockSignals(True)
        self.size.setCurrentIndex(max(0, self.size.findData(s.text_size)))
        self.size.blockSignals(False)

    def _looks_changed(self) -> None:
        s = self.win.settings
        size, white = self.size.currentData(), self.white.isChecked()
        if (size, white) == (s.text_size, s.white_text) or size not in TEXT_SIZES:
            return
        s.text_size, s.white_text = size, white
        s.save()
        self.win.restart_overlay()

    def _changed(self) -> None:
        s = self.win.settings
        story, menus = self.story.isChecked(), self.menus.isChecked()
        if not (story or menus):  # one of them stays: the box just unticked comes back
            _set_checked(self.sender(), True)
            story, menus = self.story.isChecked(), self.menus.isChecked()
        s.translate_dialogue, s.translate_ui = story, menus
        s.save()
        self.win.session.apply_content()


# ---- the dialogue -----------------------------------------------------------------------
class DialogueCard(Card):
    def __init__(self, win) -> None:  # noqa: ANN001
        super().__init__(win, _("Dialogue"))
        self._picker = None  # the area picker, while it is open
        self.glass_box = QCheckBox(_("Dialogue on frosted glass"))
        self.glass_box.setToolTip(_("Show dialogue on frosted glass in the dialogue area "
                                    "(off: covered where it is, like other text)"))
        self.glass_box.toggled.connect(self._region_changed)
        self.body.addWidget(self.glass_box)
        grow = QHBoxLayout()
        self.glass = QButtonGroup(self)
        for key, text in GLASS_STYLES.items():
            rb = QRadioButton(_(text))
            rb.setProperty("glass", key)
            self.glass.addButton(rb)
            grow.addWidget(rb)
        grow.addStretch(1)
        self.glass.buttonToggled.connect(self._glass_changed)
        self.body.addLayout(grow)
        orow = QHBoxLayout()
        self.keep_box = QCheckBox(_("Stays on screen"))
        self.type_box = QCheckBox(_("Types in"))
        self.type_box.setToolTip(_("The translation types in with the game's text (off: "
                                   "shown whole at once)"))
        for b in (self.keep_box, self.type_box):
            b.toggled.connect(self._options)
            orow.addWidget(b)
        orow.addStretch(1)
        self.body.addLayout(orow)
        self.area_label = note()
        self.body.addWidget(self.area_label)
        arow = QHBoxLayout()
        self.area_btn = button(_("Set area…"), _("Drag a box over the game's dialogue box"))
        self.area_btn.clicked.connect(lambda: self._choose_area(name=False))
        self.name_btn = button(_("Name area…"), _("Where the speaker's name is (optional)"))
        self.name_btn.clicked.connect(lambda: self._choose_area(name=True))
        self.reset_btn = button(_("Reset"), _("Back to the preset's own areas"))
        self.reset_btn.clicked.connect(self._use_builtin)
        for b in (self.area_btn, self.name_btn, self.reset_btn):
            arow.addWidget(b)
        arow.addStretch(1)
        self.body.addLayout(arow)

    def _widgets(self) -> tuple:
        return (self.glass_box, self.area_btn, self.name_btn, self.reset_btn, self.keep_box,
                self.type_box, *self.glass.buttons())

    def refresh(self) -> None:
        g, lib = self.game, self.win.library
        for w in self._widgets():
            w.setEnabled(g is not None)
            w.blockSignals(True)
        try:
            if g is None:
                self.area_label.setText("")
                return
            own = parse_region(g.region)
            builtin = lib.builtin_region(g)
            self.glass_box.setChecked(g.region_on)
            if own is not None:
                text = _("Area: yours") + (", " + _("with a name area")
                                           if parse_region(g.name_region) else "")
            elif builtin:
                text = _("Area: the preset's")
            else:
                text = _("No area yet: open a conversation in the game, then Set area…")
            self.area_label.setText(text)
            self.reset_btn.setVisible(builtin and own is not None)
            self.name_btn.setEnabled(own is not None)
            for b in self.glass.buttons():
                b.setChecked(b.property("glass") == g.glass_style)
            self.keep_box.setChecked(g.keep_glass)
            self.type_box.setChecked(g.type_in)
            active = g.region_on and lib.has_dialogue_mode(g)
            for w in (*self.glass.buttons(), self.keep_box, self.type_box):
                w.setEnabled(active)
        finally:
            for w in self._widgets():
                w.blockSignals(False)

    def _changed(self, g: GameEntry) -> None:
        self.win.settings.save()
        self.refresh()
        self.win.restart_overlay(game=g)

    def _region_changed(self, on: bool) -> None:
        g = self.game
        if g is not None:
            g.region_on = on
            self._changed(g)

    def _glass_changed(self, b, checked: bool) -> None:  # noqa: ANN001
        g = self.game
        if checked and g is not None:
            g.glass_style = b.property("glass")
            self._changed(g)

    def _options(self) -> None:
        g = self.game
        if g is not None:
            g.keep_glass, g.type_in = self.keep_box.isChecked(), self.type_box.isChecked()
            self._changed(g)

    def _use_builtin(self) -> None:
        g = self.game
        if g is not None:
            g.region, g.name_region = "", ""
            self._changed(g)

    def _choose_area(self, name: bool) -> None:
        """Drag a box over the game: its dialogue area, or where the speaker's name is."""
        from app.region_picker import RegionPicker, picture_of

        g = self.game
        if g is None:
            return
        found = self.win.library.find_window(g)
        if found is None:
            self.win.status(_("Open the game with a conversation on screen first."))
            return
        prompt = (_("Drag a box over where the speaker's name appears") if name else
                  _("Drag a box over the game's dialogue box: everywhere its text can appear"))
        current = parse_region(g.name_region if name else g.region)
        aspect = self.win.library.profile(g).window.aspect
        picker = RegionPicker(found.client, picture_of(found, aspect), current, prompt)

        def chosen(rect) -> None:  # noqa: ANN001
            self._picker = None
            if rect is None:
                return
            if name:
                g.name_region = region_text(rect)
            else:
                first = not g.region
                g.region, g.region_on = region_text(rect), True
                if first:  # (a message box the player chose: it is always there)
                    g.keep_glass = True
            self._changed(g)
            self.win.status(_("Name area saved.") if name else _("Dialogue area saved."))

        picker.done.connect(chosen)
        picker.show()
        self._picker = picker  # (kept until it closes)


# ---- live translation -------------------------------------------------------------------
class LiveCard(Card):
    """Live translation: off, PLaMo on this computer (installed with one button), the
    player's own key for Claude, ChatGPT or DeepL, or a model behind an OpenAI-compatible
    server the player names (nothing installed for it), with the model to use."""

    def __init__(self, win) -> None:  # noqa: ANN001
        super().__init__(win, _("Live translation"))
        self.task: Task | None = None  # a test, or the local model's download
        self._listed: dict[str, list[str]] = {}  # the models Test found, per engine
        erow = QHBoxLayout()
        self.engine = combo()
        for key, text in ENGINES.items():
            self.engine.addItem(_(text), key)
        self.engine.setToolTip(_("Translates text no database has, as it appears (only the "
                                 "text is sent, never pictures; PLaMo sends nothing at all)"))
        self.engine.activated.connect(lambda _i: self._engine_changed())
        erow.addWidget(self.engine, 1)
        self.test_btn = button(_("Test"), _("Check that the engine works"))
        self.test_btn.clicked.connect(self._test)
        erow.addWidget(self.test_btn)
        self.body.addLayout(erow)
        self.into = note()
        self.body.addWidget(self.into)
        # PLaMo: which model, and installing it
        self.local_row = QWidget()
        lrow = QHBoxLayout(self.local_row)
        lrow.setContentsMargins(0, 0, 0, 0)
        self.local_model = combo()
        for key, m in MODELS.items():
            self.local_model.addItem(f"{m.name} · {_(m.note)} ({_size(m.file.size)})", key)
        self.local_model.activated.connect(lambda _i: self._local_model_changed())
        self.install_btn = button(_("Install"), _("Download the model and the program that "
                                                  "runs it, once"))
        self.install_btn.clicked.connect(self._install)
        lrow.addWidget(self.local_model, 1)
        lrow.addWidget(self.install_btn)
        self.body.addWidget(self.local_row)
        self.plamo_note = note(_("Built with PLaMo · {license}").format(license=_license()))
        self.body.addWidget(self.plamo_note)
        # another model: its server's address (a list of known ones fills it in)
        self.server_row = QWidget()
        srow = QHBoxLayout(self.server_row)
        srow.setContentsMargins(0, 0, 0, 0)
        self.server = QLineEdit()
        self.server.setPlaceholderText(_("Server address (http://localhost:11434/v1)"))
        self.server.setToolTip(_("An OpenAI-compatible server: one on this computer (Ollama, "
                                 "LM Studio, llama.cpp…) or a service (OpenRouter, Gemini…)"))
        self.server.editingFinished.connect(self._server_changed)
        srow.addWidget(self.server, 1)
        known = button(_("Servers"), _("Fill in a known server's address"))
        menu = QMenu(known)
        for name, url in SERVERS.items():
            menu.addAction(f"{name}  ·  {url}").triggered.connect(
                lambda _c=False, u=url: self._server_chosen(u))
        known.setMenu(menu)
        srow.addWidget(known)
        self.body.addWidget(self.server_row)
        # Claude, ChatGPT, DeepL, another model: the key and the model
        self.key_row = QWidget()
        krow = QHBoxLayout(self.key_row)
        krow.setContentsMargins(0, 0, 0, 0)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.returnPressed.connect(self._save_key)
        save = button(_("Save"))
        save.clicked.connect(self._save_key)
        krow.addWidget(self.key, 1)
        krow.addWidget(save)
        self.body.addWidget(self.key_row)
        self.model = combo()
        self.model.activated.connect(lambda _i: self._model_changed())
        self.body.addWidget(self.model)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.body.addWidget(self.bar)
        self.state = note()
        self.body.addWidget(self.state)
        self.usage = note()
        self.usage.setVisible(False)
        self.body.addWidget(self.usage)

    def _engine_key(self) -> str:
        s = self.win.settings.live
        return s.engine if s.enabled else "off"

    def refresh(self) -> None:
        from translate import CLAUDE_MODELS

        s = self.win.settings.live
        engine = self._engine_key()
        self.engine.blockSignals(True)
        self.engine.setCurrentIndex(max(0, self.engine.findData(engine)))
        self.engine.blockSignals(False)
        local, keyed = engine == "plamo", engine in KEY_HINTS
        self.local_row.setVisible(local)
        self.plamo_note.setVisible(local)
        self.server_row.setVisible(engine == "custom")
        if not self.server.hasFocus():
            self.server.setText(s.custom_url)
        self.key_row.setVisible(keyed)
        self.model.setVisible(keyed and engine != "deepl")
        if keyed:
            self.key.setPlaceholderText(_(KEY_HINTS[engine]))
        self.model.blockSignals(True)
        self.model.clear()
        self.model.setEditable(engine in LISTED)
        if engine == "claude":
            for mid, text in CLAUDE_MODELS.items():
                self.model.addItem(_(text), mid)
            self.model.setCurrentIndex(max(0, self.model.findData(s.claude_model)))
        elif engine in LISTED:
            chosen = s.openai_model if engine == "openai" else s.custom_model
            for mid in dict.fromkeys([chosen, *self._listed.get(engine, [])]):
                if mid:
                    self.model.addItem(mid, mid)
            self.model.setCurrentIndex(0 if chosen else -1)
            if engine == "custom":
                self.model.lineEdit().setPlaceholderText(_("Model name (Test lists them)"))
            edit = self.model.lineEdit()
            if edit is not None and not edit.property("hooked"):  # (once per line edit)
                edit.editingFinished.connect(self._model_changed)
                edit.setProperty("hooked", True)
        self.model.blockSignals(False)
        self.local_model.blockSignals(True)
        self.local_model.setCurrentIndex(max(0, self.local_model.findData(s.plamo_model)))
        self.local_model.blockSignals(False)
        installed = local and self._models().installed(s.plamo_model)
        self.install_btn.setVisible(local and not installed)
        self.test_btn.setVisible(engine != "off" and (installed or not local))
        self._show_into(engine)
        if self.task is not None:
            return  # (the task says how it goes)
        if engine == "off":
            self._say("")
        elif local:
            self._say(_("Runs on this computer: free, and nothing you read leaves it.")
                      if installed else _("Not installed yet: {size} to download, once.")
                      .format(size=_size(self._models().download_size(s.plamo_model))))
        else:
            k = get_key(engine)
            if k:
                self._say(_("Key saved: {key}").format(key=mask(k)))
            elif engine == "custom":
                self._say(_("Only the text is sent, to the server above. A server on this "
                            "computer usually needs no key."))
            else:
                self._say(_("No key saved yet"))

    @staticmethod
    def _models():  # noqa: ANN205 - translate.local.LocalModels
        from translate.local import LocalModels

        return LocalModels(models_dir())

    def _show_into(self, engine: str) -> None:
        from translate.backends.plamo import plamo_code

        g = self.game
        if g is None or engine == "off":
            self.into.setVisible(False)
            return
        lang = self.win.library.target_lang(g)
        text = _("Into {language}").format(language=native_name(lang))
        if engine == "plamo" and not plamo_code(lang):
            text += " — " + tinted(_("PLaMo does not offer it"), "problem")
        self.into.setText(text)
        self.into.setVisible(True)

    def _say(self, text: str) -> None:
        self.state.setText(text)
        self.state.setVisible(bool(text))

    def _engine_changed(self) -> None:
        s = self.win.settings.live
        engine = self.engine.currentData()
        before = (s.enabled, s.engine)
        s.enabled = engine != "off"
        if engine != "off":
            s.engine = engine
        self.win.settings.save()
        self.refresh()
        if before != (s.enabled, s.engine):
            self.win.live_changed()

    def _local_model_changed(self) -> None:
        s = self.win.settings.live
        key = self.local_model.currentData()
        if key in MODELS and key != s.plamo_model and self.task is None:
            s.plamo_model = key
            self.win.settings.save()
            self.refresh()
            self.win.live_changed()

    def _model_changed(self) -> None:
        s = self.win.settings.live
        before = (s.claude_model, s.openai_model, s.custom_model)
        if s.engine == "claude":
            s.claude_model = self.model.currentData() or s.claude_model
        elif s.engine == "openai":
            s.openai_model = self.model.currentText().strip() or s.openai_model
        elif s.engine == "custom":
            s.custom_model = self.model.currentText().strip() or s.custom_model
        if before != (s.claude_model, s.openai_model, s.custom_model):
            self.win.settings.save()
            self.win.live_changed()

    def _server_chosen(self, url: str) -> None:
        self.server.setText(url)
        self._server_changed()

    def _server_changed(self) -> None:
        s = self.win.settings.live
        url = self.server.text().strip()
        if url != s.custom_url:
            s.custom_url = url
            self._listed.pop("custom", None)  # (another server: other models)
            self.win.settings.save()
            self.refresh()
            self.win.live_changed()

    def _save_key(self) -> None:
        engine = self._engine_key()
        key = self.key.text().strip()
        if engine not in KEY_HINTS or not key:  # (an empty field never removes a key)
            return
        try:
            set_key(key, engine)
        except OSError as e:
            QMessageBox.warning(self, _("API key"), _("The key could not be saved: {why}")
                                .format(why=e))
            return
        self.key.clear()
        self.refresh()
        self.win.keys_changed()
        self.win.live_changed()

    def _test(self) -> None:
        from app.session import make_backend

        if self.task is not None:
            return
        if self.key.text().strip():
            self._save_key()
        engine = self._engine_key()
        if engine == "off":
            return
        self._say(_("Starting the model…") if engine == "plamo" else _("Testing…"))
        settings = self.win.settings

        if engine == "custom":
            self._server_changed()  # (typed, not confirmed yet)
            self._model_changed()

        def work(_task: Task):  # noqa: ANN202
            backend = make_backend(settings, engine)
            models = None
            if engine in LISTED:
                try:
                    models = backend.models()
                except Exception:  # noqa: BLE001 - the list is a nicety
                    models = None
            if engine == "custom" and not backend.model and models:
                return _("Choose one of the server's models."), models
            return backend.check(), models

        def done(res) -> None:  # noqa: ANN001
            msg, models = res
            if models:
                self._listed[engine] = models
            self._worked(msg)

        self._run(work, None, done)

    def _install(self) -> None:
        if self.task is not None:
            self.task.cancel.set()  # (during a download the button stops it)
            return
        from app.dialogs import LicenseDialog

        s = self.win.settings.live
        key, models = s.plamo_model, self._models()
        name, terms = MODELS[key].name, assets_dir() / "plamo_license.txt"
        intro = _(
            "Download {name} and the program that runs it ({size})? It translates on this "
            "computer, on the graphics card when it can: free, and nothing you read leaves "
            "this computer.").format(name=name, size=_size(models.download_size(key)))
        intro += "<br><br>" + _(
            "PLaMo is Preferred Networks' translation model. To use it you accept its license, "
            "below: among other things you must be 18 or older (or have a parent's or "
            "guardian's consent), and using it for business needs their free registration.")
        ask = LicenseDialog(_("Install {name}").format(name=name), intro,
                            terms.read_text(encoding="utf-8"), _("Accept and install"), self)
        if ask.exec() != LicenseDialog.DialogCode.Accepted:
            return
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.install_btn.setText(_("Stop"))
        self._say(_("Downloading…"))

        def work(task: Task) -> str:
            from translate.local import Cancelled

            try:
                models.install(key, task.report, task.cancel, license_file=terms)
            except Cancelled:
                return _("Download stopped: Install goes on where it stopped.")
            return _("{name} is installed.").format(name=name)

        def progress(done, total) -> None:  # noqa: ANN001
            if total:
                self.bar.setRange(0, 1000)
                self.bar.setValue(int(1000 * done / total))
                self.bar.setFormat(f"{_size(done)} / {_size(total)}")

        def finished(msg: str) -> None:
            self._worked(msg)
            if s.enabled and s.engine == "plamo" and models.installed(s.plamo_model):
                self.win.live_changed()

        self._run(work, progress, finished)

    def _run(self, work, progress, finished) -> None:  # noqa: ANN001
        self.task = Task(work)
        if progress is not None:
            self.task.progress.connect(progress)
        self.task.done.connect(finished)
        self.task.failed.connect(lambda why: self._done(f"✘ {why}"))
        self.task.start()

    def _done(self, text: str) -> None:
        self.task = None
        self.bar.setVisible(False)
        self.install_btn.setText(_("Install"))
        self.refresh()
        self._say(text)

    def _worked(self, msg: str) -> None:
        """A test or an install worked: live translation, if it stopped (the key was
        refused, the model missing), starts again."""
        self._done(f"✔ {msg}")
        stats = self.win.session.live_stats()
        if stats is not None and stats["halted"]:
            self.win.live_changed()

    def show_usage(self, stats: dict | None) -> None:
        """What live translation did since the overlay started (while it runs)."""
        if stats is None:
            self.usage.setVisible(False)
            return
        text = _("{n} lines translated live").format(n=f"{stats['translated']:,}")
        if stats["failed"]:
            text += ", " + _("{n} failed").format(n=f"{stats['failed']:,}")
        if stats["halted"]:
            text += " · " + _("stopped: {why}").format(why=stats["halted"])
        elif stats["last_error"]:
            text += " · " + _("last problem: {why}").format(why=stats["last_error"])
        self.usage.setText(text)
        self.usage.setVisible(True)


class ReaderCard(Card):
    """The text reader: the standard one, or MeikiOCR on an NVIDIA graphics card (faster;
    nothing of it ships with the app: installed with one button, after the player agrees
    to the download). MeikiOCR cannot be chosen without an NVIDIA card, nor yet on Linux."""

    def __init__(self, win) -> None:  # noqa: ANN001
        from app.reader import MEIKI, READERS, meiki_system, nvidia_gpu

        super().__init__(win, _("Text reader"))
        self.task: Task | None = None
        self.system = meiki_system()
        self.nvidia = nvidia_gpu()
        row = QHBoxLayout()
        self.reader = combo()
        for key, text in READERS.items():
            self.reader.addItem(_(text), key)
        if not self.nvidia:
            item = self.reader.model().item(self.reader.findData(MEIKI))
            item.setEnabled(False)
            item.setToolTip(_("Needs an NVIDIA graphics card") if self.system
                            else _("Windows only for now"))
        self.reader.activated.connect(lambda _i: self._changed())
        row.addWidget(self.reader, 1)
        self.install_btn = button(_("Install"), _("Download MeikiOCR and what it needs, once"))
        self.install_btn.clicked.connect(self._install)
        row.addWidget(self.install_btn)
        self.body.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.body.addWidget(self.bar)
        self.state = note()
        self.body.addWidget(self.state)

    @staticmethod
    def _installed() -> bool:
        from app.reader import MeikiReader

        return MeikiReader(models_dir()).installed()

    def refresh(self) -> None:
        from app.reader import MEIKI, MeikiReader

        s = self.win.settings
        chosen = s.ocr_reader if self.nvidia else "standard"
        self.reader.blockSignals(True)
        self.reader.setCurrentIndex(max(0, self.reader.findData(chosen)))
        self.reader.blockSignals(False)
        installed = self._installed()
        self.install_btn.setVisible(self.task is not None or (chosen == MEIKI and not installed))
        if self.task is not None:
            return
        if not self.system:
            text = _("MeikiOCR runs on Windows only for now: this computer reads with the "
                     "standard reader.")
        elif not self.nvidia:
            text = _("MeikiOCR needs an NVIDIA graphics card: this computer reads with the "
                     "standard reader.")
        elif chosen != MEIKI:
            text = _("This computer has an NVIDIA graphics card: MeikiOCR reads about twice as "
                     "fast.")
        elif not installed:
            text = _("Not installed yet: {size} to download, once.").format(
                size=_size(MeikiReader(models_dir()).download_size()))
        else:
            text = _("MeikiOCR reads Japanese games; others use the standard reader.")
        self.state.setText(text)

    def _changed(self) -> None:
        key = self.reader.currentData()
        s = self.win.settings
        if key == s.ocr_reader:
            return
        s.ocr_reader = key
        s.save()
        self.refresh()
        if key == "standard" or self._installed():
            self.win.restart_overlay()

    def _install(self) -> None:
        if self.task is not None:
            self.task.cancel.set()  # (during a download the button stops it)
            return
        from PySide6.QtWidgets import QMessageBox

        from app.reader import MEIKI_LICENSE, MeikiReader

        reader = MeikiReader(models_dir())
        ask = QMessageBox(self)
        ask.setWindowTitle(_("Install MeikiOCR"))
        ask.setTextFormat(Qt.TextFormat.RichText)
        ask.setText(_(
            "Download MeikiOCR and NVIDIA's CUDA libraries ({size})? They stay in the app's "
            "models folder, and the text reader runs on your graphics card with them.<br><br>"
            "From PyPI: ONNX Runtime's CUDA build (MIT) and NVIDIA's CUDA and cuDNN runtime "
            "libraries (NVIDIA's licenses). From Hugging Face: MeikiOCR's models ({license}, "
            "LGPL-3.0).").format(
            size=_size(reader.download_size()),
            license=f'<a href="{MEIKI_LICENSE}">rtr46</a>'))
        ask.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ask.button(QMessageBox.StandardButton.Yes).setText(_("Download"))
        if ask.exec() != QMessageBox.StandardButton.Yes:
            return
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.install_btn.setText(_("Stop"))
        self.state.setText(_("Downloading…"))

        def work(task: Task) -> str:
            from translate.local import Cancelled

            try:
                reader.install(task.report, task.cancel)
            except Cancelled:
                return _("Download stopped: Install goes on where it stopped.")
            return _("MeikiOCR is installed.")

        def progress(done, total) -> None:  # noqa: ANN001
            if total:
                self.bar.setRange(0, 1000)
                self.bar.setValue(int(1000 * done / total))
                self.bar.setFormat(f"{_size(done)} / {_size(total)}")

        def finished(msg: str) -> None:
            self._done(f"✔ {msg}")
            if self._installed():
                self.win.restart_overlay()

        self.task = Task(work)
        self.task.progress.connect(progress)
        self.task.done.connect(finished)
        self.task.failed.connect(lambda why: self._done(f"✘ {why}"))
        self.task.start()

    def _done(self, text: str) -> None:
        self.task = None
        self.bar.setVisible(False)
        self.install_btn.setText(_("Install"))
        self.refresh()
        self.state.setText(text)


def _size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _license() -> str:
    """A link to PLaMo's license, for rich text."""
    return f'<a href="{PLAMO_LICENSE}">' + _("PLaMo Community License") + "</a>"


# ---- diagnostics ------------------------------------------------------------------------
GPU_PROVIDERS = ("CUDAExecutionProvider", "OpenVINOExecutionProvider")  # (off Windows)
WINDOWS = sys.platform == "win32"  # (its ONNX Runtime has DirectML)


class DiagnosticsCard(Card):
    """How the text reader runs, and what keeps the HUD from working; shown once the game
    window is found (or when there is a problem)."""

    def __init__(self, win) -> None:  # noqa: ANN001
        super().__init__(win)
        self._data: dict = {}
        self._alert = ""
        self.reader = QLabel()
        self.reader.setWordWrap(True)
        self.body.addWidget(self.reader)
        self.warning = note()
        self.body.addWidget(self.warning)
        self.gpu_btn = button(_("Try the GPU again"))
        self.gpu_btn.clicked.connect(win.retry_gpu)
        self.body.addWidget(self.gpu_btn)
        self.timing = note()
        self.body.addWidget(self.timing)
        self.cache = note()
        self.body.addWidget(self.cache)
        copy = button(_("Copy report"), _("Copies these diagnostics for a bug report"))
        copy.clicked.connect(self._copy)
        self.body.addWidget(copy)

    def refresh(self) -> None:
        pass  # (updated by the panel's clock: update())

    def update_from(self, data: dict | None, problem: str = "") -> bool:
        """Show the HUD's diagnostics. Returns whether there is something the player should
        see (a problem, or the slow CPU reader)."""
        data = data or {}
        self._data = dict(data)
        ocr = data.get("ocr") if isinstance(data.get("ocr"), dict) else {}
        device = str(ocr.get("device") or "").lower()
        cpu = device == "cpu"
        reader = {"dml": _("GPU (DirectML)"), "cuda": _("GPU (CUDA)"),
                  "openvino": "OpenVINO", "cpu": _("CPU")}.get(device, _("starting"))
        self.reader.setText(_("Text reader: {reader} · Capture: {capture}").format(
            reader=f"<b>{reader}</b>", capture=data.get("capture", "—")))
        notes = []
        if problem:
            notes.append(tinted("⚠ " + problem, "problem"))
        # can this build read on a GPU at all? (on Linux its ONNX Runtime is the CPU build)
        gpu_built = WINDOWS or any(
            p in GPU_PROVIDERS for p in ocr.get("providers") or [])
        if cpu and not gpu_built:
            notes.append(tinted("⚠ " + _("The CPU reader is slow."), "wait") + "<br>" + _(
                "On this system the text reader runs on the CPU: the GPU reader is for "
                "Windows only for now."))
        elif cpu:
            why = str(ocr.get("fallback_reason") or "") or _(
                "The GPU reader did not start; see the app's log.")
            notes.append(tinted("⚠ " + _("The CPU reader is slow."), "wait") + "<br>"
                         + why)
        self.warning.setText("<br>".join(notes))
        self.warning.setVisible(bool(notes))
        self.gpu_btn.setVisible(cpu and gpu_built)
        timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}

        def ms(part: str) -> str:
            t = timings.get(part) if isinstance(timings.get(part), dict) else {}
            return f"{t.get('median', 0):g} / {t.get('p95', 0):g} ms"

        self.timing.setText(_("Reading time (median / slowest 5 %): dialogue {d} · menus "
                              "{u} · all {t}").format(d=ms("dialogue"), u=ms("ui"),
                                                      t=ms("tick")))
        cache = data.get("cover_cache") if isinstance(data.get("cover_cache"), dict) else {}
        recall = data.get("recall") if isinstance(data.get("recall"), dict) else {}
        lookup = data.get("lookup") if isinstance(data.get("lookup"), dict) else {}
        self.cache.setText(_("Remembered menus: {recalled} · cache {hits} hits / {misses} "
                             "misses · lines no database had: {unknown}").format(
            recalled=recall.get("hits", 0), hits=cache.get("hits", 0),
            misses=cache.get("misses", 0), unknown=lookup.get("misses", 0)))
        alert = "problem" if problem else "cpu" if cpu else ""
        news = bool(alert) and alert != self._alert
        self._alert = alert
        return news

    def _copy(self) -> None:
        QApplication.clipboard().setText(json.dumps(self._data, ensure_ascii=False, indent=2))
        self.win.status(_("Diagnostics copied."))


# ---- the rest ---------------------------------------------------------------------------
class StreamCard(Card):
    """The overlay for streaming: OBS cannot see the overlay window (kept out of every
    capture, so the HUD never reads its own translations), so the app serves it as a page
    on this computer that OBS shows as a Browser source above the game."""

    def __init__(self, win) -> None:  # noqa: ANN001
        super().__init__(win)
        self.box = QCheckBox(_("Show the translations to OBS"))
        self.box.setToolTip(_("A page on this computer only, with the translations on a "
                              "clear background, for a Browser source above the game"))
        self.box.toggled.connect(self._changed)
        self.body.addWidget(self.box)
        urow = QHBoxLayout()
        self.url = QLabel()
        self.url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        urow.addWidget(self.url, 1)
        self.copy = button(_("Copy"), _("Copy the address"))
        self.copy.clicked.connect(self._copy)
        urow.addWidget(self.copy)
        self.body.addLayout(urow)
        self.how = note()
        self.body.addWidget(self.how)

    def refresh(self) -> None:
        s, session = self.win.settings, self.win.session
        _set_checked(self.box, s.stream_source)
        url = session.stream_url
        self.url.setText(url)
        for w in (self.url, self.copy, self.how):
            w.setVisible(bool(url))
        size = session.picture_size()
        if size is not None:
            where = _("width {w} and height {h}").format(w=size[0], h=size[1])
        else:
            where = _("the size of the game's picture (its resolution)")
        self.how.setText(_(
            "In OBS: add a Browser source with this address and {size}, above your capture "
            "of the game (Game or Window capture: the game without the overlay).").format(
            size=where))

    def _changed(self, on: bool) -> None:
        self.win.settings.stream_source = on
        self.win.settings.save()
        self.win.session.apply_stream()
        self.refresh()

    def _copy(self) -> None:
        url = self.win.session.stream_url
        if url:
            QApplication.clipboard().setText(url)
            self.win.status(_("Address copied."))


class MoreCard(Card):
    """Settings that are set once: the panel's side and handle, the hotkeys, how the reader
    weighs speed against care, and the miss log."""

    def __init__(self, win) -> None:  # noqa: ANN001
        super().__init__(win)
        srow = QHBoxLayout()
        srow.addWidget(QLabel(_("Panel side")))
        self.sides = QButtonGroup(self)
        for side, text in (("left", _("Left")), ("right", _("Right"))):
            rb = QRadioButton(text)
            rb.setProperty("side", side)
            self.sides.addButton(rb)
            srow.addWidget(rb)
        srow.addStretch(1)
        self.sides.buttonToggled.connect(self._side_changed)
        self.body.addLayout(srow)
        self.handle_box = QCheckBox(_("Handle on the edge when folded"))
        self.handle_box.toggled.connect(self._handle_changed)
        self.body.addWidget(self.handle_box)
        form = QFormLayout()
        self.keys: dict[str, HotkeyButton] = {}
        for attr, label in HOTKEYS.items():
            b = HotkeyButton(getattr(win.settings, attr))
            b.changed.connect(lambda spec, a=attr: self._hotkey_changed(a, spec))
            form.addRow(_(label), b)
            self.keys[attr] = b
        self.reading = combo()
        for key, text in READING_MODES.items():
            self.reading.addItem(_(text), key)
        self.reading.activated.connect(lambda _i: self._reading_changed())
        form.addRow(_("Reading"), self.reading)
        self.body.addLayout(form)
        self.misses_box = QCheckBox(_("Write down lines no database has"))
        self.misses_box.setToolTip(_("Kept on this computer only: what the next "
                                     "database should add"))
        self.misses_box.toggled.connect(self._misses_changed)
        self.body.addWidget(self.misses_box)
        mrow = QHBoxLayout()
        self.misses = note()
        mrow.addWidget(self.misses, 1)
        show = link_button(_("Open folder"))
        show.clicked.connect(self._open_misses)
        mrow.addWidget(show, 0, Qt.AlignmentFlag.AlignTop)
        self.body.addLayout(mrow)

    def refresh(self) -> None:
        s = self.win.settings
        for b in self.sides.buttons():
            b.blockSignals(True)
            b.setChecked(b.property("side") == s.panel_side)
            b.blockSignals(False)
        _set_checked(self.handle_box, s.panel_handle)
        _set_checked(self.misses_box, s.log_misses)
        for attr, b in self.keys.items():
            b.set_spec(getattr(s, attr))
        self.reading.blockSignals(True)
        self.reading.setCurrentIndex(max(0, self.reading.findData(s.confirmation_mode)))
        self.reading.blockSignals(False)
        self.show_misses()

    def show_misses(self) -> None:
        n = self.win.session.misses_count()
        self.misses.setText(_("{n} written down this session and before").format(n=f"{n:,}")
                            if n else "")

    def _open_misses(self) -> None:
        g = self.game
        if g is not None:
            d = self.win.library.misses_dir(g)
            d.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _misses_changed(self, on: bool) -> None:
        self.win.settings.log_misses = on
        self.win.settings.save()
        self.win.restart_overlay()

    def _reading_changed(self) -> None:
        mode = self.reading.currentData()
        if mode in READING_MODES and mode != self.win.settings.confirmation_mode:
            self.win.settings.confirmation_mode = mode
            self.win.settings.save()
            self.win.restart_overlay()

    def _hotkey_changed(self, attr: str, spec: str) -> None:
        from hud.hotkey import available, pretty

        s = self.win.settings
        if not spec:
            self.win.status(_("Those keys cannot be a hotkey: add Ctrl, Alt or Win."))
            return
        if any(getattr(s, a).lower() == spec.lower() for a in HOTKEYS if a != attr):
            self.win.status(_("{keys} is one of the app's other hotkeys already.").format(
                keys=pretty(spec)))
            return
        if spec.lower() == getattr(s, attr).lower():
            return
        ok = self.win.set_panel_hotkey(spec) if attr == "panel_hotkey" else available(spec)
        if not ok:
            self.win.status(_("Another program has {keys}: choose other keys.").format(
                keys=pretty(spec)))
            return
        setattr(s, attr, spec)
        s.save()
        self.refresh()
        if attr != "panel_hotkey":  # (the overlay takes its hotkeys when it starts)
            self.win.restart_overlay()
        self.win.status(f"{_(HOTKEYS[attr])}: {pretty(spec)}.")

    def _side_changed(self, b, checked: bool) -> None:  # noqa: ANN001
        if checked and b.property("side") != self.win.settings.panel_side:
            self.win.settings.panel_side = b.property("side")
            self.win.settings.save()
            self.win.dock()

    def _handle_changed(self, on: bool) -> None:
        self.win.settings.panel_handle = on
        self.win.settings.save()
