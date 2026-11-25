"""Global hotkeys, spelled as the settings keep them: "ctrl+shift+F8".

A spec is modifiers (ctrl, alt, shift, win) and one key name: F1-F24, A-Z, 0-9, NUM0-NUM9
(the keypad), SPACE, TAB, ESC, PAUSE, SCROLL, HOME, END, INSERT, DELETE, PAGEUP, PAGEDOWN,
LEFT, UP, RIGHT, DOWN. :func:`spec_for_key` spells one from a Qt key event (any system);
the platform turns it into its own key codes (hud.platform).
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt

from hud.platform import current

MODIFIERS = ("ctrl", "alt", "shift", "win")  # in the order a spec writes them
_MOD_ALIASES = {"control": "ctrl", "meta": "win", "super": "win"}
FUNCTION_KEYS = {f"F{i}" for i in range(1, 25)}
KEYS = (
    FUNCTION_KEYS
    | {chr(c) for c in range(ord("A"), ord("Z") + 1)}
    | {str(d) for d in range(10)}
    | {f"NUM{d}" for d in range(10)}
    | {"SPACE", "TAB", "ESC", "PAUSE", "SCROLL", "HOME", "END", "INSERT", "DELETE",
       "PAGEUP", "PAGEDOWN", "LEFT", "UP", "RIGHT", "DOWN"}
)  # fmt: skip
ALONE = FUNCTION_KEYS | {"PAUSE", "SCROLL"}  # keys a game rarely needs: fine without modifiers

_QT_KEYS = {
    int(k): name for k, name in (
        (Qt.Key.Key_Space, "SPACE"), (Qt.Key.Key_Tab, "TAB"), (Qt.Key.Key_Escape, "ESC"),
        (Qt.Key.Key_Pause, "PAUSE"), (Qt.Key.Key_ScrollLock, "SCROLL"),
        (Qt.Key.Key_Home, "HOME"), (Qt.Key.Key_End, "END"), (Qt.Key.Key_Insert, "INSERT"),
        (Qt.Key.Key_Delete, "DELETE"), (Qt.Key.Key_PageUp, "PAGEUP"),
        (Qt.Key.Key_PageDown, "PAGEDOWN"), (Qt.Key.Key_Left, "LEFT"), (Qt.Key.Key_Up, "UP"),
        (Qt.Key.Key_Right, "RIGHT"), (Qt.Key.Key_Down, "DOWN"),
    )
}  # fmt: skip
_F1, _F24 = int(Qt.Key.Key_F1), int(Qt.Key.Key_F24)
_QT_MODS = (
    (Qt.KeyboardModifier.ControlModifier, "ctrl"),
    (Qt.KeyboardModifier.AltModifier, "alt"),
    (Qt.KeyboardModifier.ShiftModifier, "shift"),
    (Qt.KeyboardModifier.MetaModifier, "win"),
)


def split_spec(spec: str) -> tuple[list[str], str]:
    """'ctrl+shift+F8' -> (["ctrl", "shift"], "F8"); ValueError for an unknown key or a
    spec without exactly one key."""
    mods: list[str] = []
    key = None
    for part in spec.replace(" ", "").split("+"):
        low = _MOD_ALIASES.get(part.lower(), part.lower())
        if low in MODIFIERS:
            if low not in mods:
                mods.append(low)
        elif part.upper() in KEYS and key is None:
            key = part.upper()
        else:
            raise ValueError(f"unknown key in hotkey: {part!r}")
    if key is None:
        raise ValueError(f"hotkey has no key: {spec!r}")
    return sorted(mods, key=MODIFIERS.index), key


def make_spec(mods: list[str], key: str) -> str | None:
    """The spec of a key with modifiers; None for a combination that cannot be a hotkey:
    an unknown key, or a typing or moving key without ctrl, alt or win (it would be taken
    from the game)."""
    if key not in KEYS:
        return None
    mods = sorted(set(mods), key=MODIFIERS.index)
    if not ({"ctrl", "alt", "win"} & set(mods)) and key not in ALONE:
        return None
    return "+".join([*mods, key])


def spec_for_key(key: int, modifiers: Qt.KeyboardModifier) -> str | None:
    """The spec of a Qt key press (QKeyEvent.key(), .modifiers()); None when it cannot be
    a hotkey (make_spec), or is a modifier alone."""
    k = int(key)
    keypad = bool(modifiers & Qt.KeyboardModifier.KeypadModifier)
    if _F1 <= k <= _F24:
        name = f"F{k - _F1 + 1}"
    elif ord("A") <= k <= ord("Z"):
        name = chr(k)
    elif ord("0") <= k <= ord("9"):
        name = f"NUM{chr(k)}" if keypad else chr(k)
    else:
        name = _QT_KEYS.get(k)
    if name is None:
        return None
    return make_spec([m for flag, m in _QT_MODS if modifiers & flag], name)


def pretty(spec: str) -> str:
    """A spec as people read it: "ctrl+shift+F8" -> "Ctrl+Shift+F8"."""
    try:
        mods, key = split_spec(spec)
    except ValueError:
        return spec
    names = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}
    return "+".join([names[m] for m in mods] + [key])


class GlobalHotkeys:
    """Hotkeys that work while another program (the game) has the focus; callbacks run on
    the GUI thread. One object per owner (the panel, the overlay)."""

    def __init__(self) -> None:
        self._keys = current().hotkeys()

    def install(self, app: object) -> None:
        """Hook into the application (once, before register)."""
        self._keys.install(app)

    def register(self, spec: str, callback: Callable[[], None]) -> bool:
        """Take *spec*; False when another program has it or it is no valid spec."""
        return self._keys.register(spec, callback)

    def unregister_all(self) -> None:
        self._keys.unregister_all()

    def uninstall(self) -> None:
        self._keys.unregister_all()
        self._keys.uninstall()


def available(spec: str) -> bool:
    """Could *spec* be registered now (no program, this one included, has it)?"""
    return current().hotkey_available(spec)
