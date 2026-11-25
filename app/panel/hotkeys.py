"""The button a hotkey is set with: click it, press the new keys."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton

from hud.hotkey import pretty, spec_for_key
from i18n import N_, _

HOTKEYS = {  # setting -> what it does
    "toggle_hotkey": N_("Hide / show translations"),
    "stop_hotkey": N_("Stop the overlay"),
    "panel_hotkey": N_("Fold the panel"),
}


class HotkeyButton(QPushButton):
    """A hotkey: click it and press the new keys (Esc keeps the old ones). ``changed``
    gives the new hotkey, or "" for keys that cannot be one."""

    changed = Signal(str)
    MODIFIERS = {Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta,
                 Qt.Key.Key_AltGr}  # fmt: skip

    def __init__(self, spec: str) -> None:
        super().__init__()
        self.spec = spec
        self._listening = False
        self.clicked.connect(self._listen)
        self._show()

    def set_spec(self, spec: str) -> None:
        self.spec = spec
        self._show()

    def _show(self) -> None:
        self.setText(_("Press the keys… (Esc: keep)") if self._listening
                     else pretty(self.spec))

    def _listen(self) -> None:
        self._listening = True
        self._show()
        self.setFocus()
        self.grabKeyboard()

    def _done(self) -> None:
        if self._listening:
            self.releaseKeyboard()
            self._listening = False
            self._show()

    def keyPressEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        if not self._listening:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key in self.MODIFIERS:
            return  # (the key itself is still to come)
        mods = event.modifiers()
        self._done()
        if key == Qt.Key.Key_Escape and not mods:
            return
        self.changed.emit(spec_for_key(key, mods) or "")

    def focusOutEvent(self, event) -> None:  # noqa: N802, ANN001 - Qt API
        self._done()
        super().focusOutEvent(event)
