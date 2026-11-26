"""The app's parts that need no window: settings, the game library, profiles, and the live
translation filter of the overlay session (what is sent, and when)."""

from __future__ import annotations

import json

import pytest

from app.config import GameEntry, Settings


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path / "data"))
    return tmp_path / "data"


def test_settings_round_trip_and_tolerate_old_files(data):
    s = Settings()
    s.games.append(GameEntry("my-game", "My Game", window_title="Game"))
    s.current = "my-game"
    s.confirmation_mode = "instant"
    s.live.enabled = True
    s.live.engine, s.live.plamo_model = "openai", "0.8b"
    s.save()
    t = Settings.load()
    assert t.current_game.name == "My Game" and t.confirmation_mode == "instant"
    assert t.live.enabled and (t.live.engine, t.live.plamo_model) == ("openai", "0.8b")
    # an older or newer file: unknown keys ignored, missing ones defaulted, bad values fixed
    raw = json.loads(Settings.path().read_text(encoding="utf-8"))
    raw["future_option"] = 1
    raw["games"][0]["unknown"] = "x"
    raw["confirmation_mode"] = "sparkles"
    raw["ocr_reader"] = "quantum"  # a reader this app does not know: the default
    del raw["live"]
    Settings.path().write_text(json.dumps(raw), encoding="utf-8")
    u = Settings.load()
    assert u.games[0].id == "my-game" and u.confirmation_mode == "balanced"
    assert not u.live.enabled and u.ocr_reader == "standard"


def test_the_window_picked_becomes_the_profiles_window(data):
    from app.library import GENERIC, Library

    lib = Library(Settings())
    ids = {pid: dlg for pid, _name, dlg in lib.profiles()}
    assert ids["azur-promilia"] is True and ids[GENERIC] is False  # dialogue mode or not
    g = GameEntry("x", "X", profile=GENERIC, window_title="My Game (1.0) [x]", process_name="g.exe")
    prof = lib.profile(g)
    assert not prof.dialogue_layouts
    assert prof.window.title_patterns == [r"My\ Game\ \(1\.0\)\ \[x\]"]  # its exact title
    assert prof.window.process_names == ["g.exe"]
    ap = lib.profile(GameEntry("ap", "AP", profile="azur-promilia"))
    assert ap.dialogue_layouts and ap.window.title_patterns  # the profile's own window
    assert lib.has_dialogue_mode(GameEntry("ap", "AP", profile="azur-promilia"))
    assert not lib.has_dialogue_mode(g)


def test_a_game_added_gets_its_own_folder_and_an_empty_database_folder(data):
    from app.library import Library

    s = Settings()
    lib = Library(s)
    g = lib.add_game("My Japanese RPG", window_title="RPG")
    assert g.id == "my-japanese-rpg" and s.current == g.id
    assert lib.db_dir(g).is_dir() and not any(lib.db_dir(g).iterdir())
    assert lib.info(g)["lines"] == 0 and lib.info(g)["files"] == 0
    g2 = lib.add_game("My Japanese RPG")
    assert g2.id == "my-japanese-rpg-2"  # never the same folder
    assert Settings.load().game(g.id) is not None  # saved
    lib.remove_game(g, delete_files=True)
    assert not lib.folder(g2).parent.joinpath(g.id, "database.sqlite").exists()
    assert s.game(g.id) is None


def test_the_first_start_adds_the_games_with_a_built_in_profile(data):
    from app.library import Library

    s = Settings()
    Library(s).ensure_defaults()
    assert [g.profile for g in s.games] == ["azur-promilia"]


def _session(monkeypatch, clock):
    """A session's live translation filter alone, no window: what it sends."""
    import threading

    from app.session import HudSession, LiveReadings

    sent = []
    sess = HudSession.__new__(HudSession)
    sess.live = type("L", (), {"request": lambda self, item: sent.append(item)})()
    sess._asked, sess.game, sess.misses = {}, None, None
    sess.readings = LiveReadings()
    sess._lock = threading.Lock()
    monkeypatch.setattr("app.session.time.monotonic", lambda: clock[0])
    return sess, sent


def test_live_translation_sends_settled_japanese_text_at_once(data, monkeypatch):
    clock = [100.0]
    sess, sent = _session(monkeypatch, clock)
    box = (10, 10, 100, 20)
    # (the core reports only text that held still while it was read)
    assert not sess._unmatched("いまなら朝ごはんにも間に合う！", "dialogue", "Terara", 0.95, box)
    assert len(sent) == 1 and sent[0].text.startswith("いまなら") and sent[0].kind == "dialogue"
    assert not sess._unmatched("装備を変更", "ui", None, 0.9, box)  # a label: at once too
    assert [i.text for i in sent[1:]] == ["装備を変更"]
    for junk in ("HP", "123/456", "の", "L-SHIFT"):  # no Japanese to speak of
        assert not sess._unmatched(junk, "ui", None, 0.99, box)  # (never read again for it)
    assert not sess._unmatched("装備", "ui", None, 0.5, box)  # read unsure
    assert len(sent) == 2


def test_a_number_on_screen_is_one_text_for_live_translation(data, monkeypatch):
    clock = [0.0]
    sess, sent = _session(monkeypatch, clock)
    for n in (12, 11, 10):  # a timer counting down: one text, the number a placeholder
        sess._unmatched(f"残り{n}秒です", "ui", None, 0.95, (0, 0, 50, 20))
        clock[0] += 1.0
    assert sent and all(i.text == "残り{n}秒です" for i in sent)
    assert sess._asked[sent[0].text][0] == "残り10秒です"  # what was read last


def test_a_line_pausing_at_a_comma_is_not_sent_half_typed(data, monkeypatch):
    clock = [0.0]
    sess, sent = _session(monkeypatch, clock)
    for _ in range(3):  # the typing stops at the comma: read again, not sent
        assert sess._unmatched("おお、テララ", "dialogue", None, 0.95, (0, 0, 50, 20))
        clock[0] += 0.25
    assert not sent
    # the whole sentence held still: sent at once
    assert not sess._unmatched("おお、テララ。任務はもう終わったかの？", "dialogue", None, 0.95,
                               (0, 0, 9, 9))
    assert len(sent) == 1 and sent[0].text.endswith(("？", "?"))
    clock[0] += 1.0  # a line that ends without a stop, the same a moment later: sent
    assert not sess._unmatched("おお、テララ", "dialogue", None, 0.95, (0, 0, 50, 20))
    assert len(sent) == 2


def test_settings_survive_hand_edits_and_damage(tmp_path):
    import json

    from app.config import Settings

    p = tmp_path / "settings.json"
    p.write_text(
        "﻿"
        + json.dumps(
            {
                "log_misses": "yes",  # wrong type: the default
                "ui_language": "es",
                "games": [
                    {"id": "ok-game", "name": "OK"},
                    {"id": "ok-game", "name": "twice"},  # a duplicate: dropped
                    {"id": "..\\..\\Windows", "name": "escape"},  # not a folder name
                    {"id": "no-name"},  # a name is made up
                    "junk",
                ],
                "live": [1, 2],
                "text_size": 95,  # not one of the sizes: the game's own
            }
        ),
        encoding="utf-8",
    )
    s = Settings.load(p)
    assert s.log_misses is True and s.ui_language == "es"
    assert [(g.id, g.name) for g in s.games] == [("ok-game", "OK"), ("no-name", "no-name")]
    assert s.live.enabled is False and s.text_size == 100 and not s.white_text
    p.write_text("{not json", encoding="utf-8")
    assert Settings.load(p).games == []
    assert not p.exists() and list(tmp_path.glob("settings.*.bad.json"))  # kept aside


def test_the_game_window_is_never_a_browser_tab_named_after_it():
    from capture.profile import WindowMatch
    from hud.platform import PART, WHOLE, WindowRef, best_window

    wins = [
        WindowRef(1, "Azur Promilia wiki - Google Chrome", 10, "chrome.exe", (0, 0, 2560, 1400)),
        WindowRef(2, "Azur Promilia", 20, "azurpromilia.exe", (0, 0, 1920, 1080)),
        WindowRef(3, "BlueStacks App Player", 30, "hd-player.exe", (0, 0, 1280, 720)),
    ]

    def find(pats, exact=(), procs=()):
        return best_window(list(wins), WindowMatch(list(procs), list(pats), list(exact)),
                           own_pid=-1)  # fmt: skip

    pats = ["Azur Promilia", "BlueStacks"]
    got = find(pats)
    assert (got.id, got.score) == (2, WHOLE)  # the game, not the larger browser
    del wins[1]  # the game closed: the emulator, by a part of its title
    got = find(pats)
    assert (got.id, got.score) == (3, PART)
    # a browser's window counts only as the player picked it, never by a pattern
    title = "Azur Promilia wiki - Google Chrome"
    assert find([r"Azur Promilia wiki \- Google Chrome"]) is None
    assert find([], exact=[title]).id == 1
    assert find(["Azur"]) is None  # only the browser's title has it
    # a folder named after the game (File Explorer shows just its name) is not the game
    wins.append(WindowRef(4, "Azur Promilia", 40, "explorer.exe", (0, 0, 1600, 900)))
    assert find(pats).id == 3
    # the program alone finds it (the release client: AzurPromiliaJP.exe, any title)
    wins.append(WindowRef(5, "Some title", 50, "azurpromiliajp.exe", (0, 0, 1280, 720)))
    assert find([], procs=["AzurPromiliaJP.exe"]).id == 5


def test_a_picked_window_of_a_shared_program_is_known_by_its_title():
    from app.windows import Choice

    assert Choice(1, "Azur Promilia", "HD-Player.exe", (1280, 720)).own_program == ""
    assert Choice(1, "Endfield", "Endfield.exe", (1920, 1080)).own_program == "Endfield.exe"


def test_hotkeys_are_spelled_from_the_keys_pressed():
    from PySide6.QtCore import Qt

    from hud.hotkey import make_spec, pretty, spec_for_key, split_spec

    mod = Qt.KeyboardModifier
    assert spec_for_key(Qt.Key.Key_F8, mod.ControlModifier | mod.ShiftModifier) == "ctrl+shift+F8"
    assert spec_for_key(Qt.Key.Key_A, mod.AltModifier) == "alt+A"
    assert spec_for_key(Qt.Key.Key_F13, mod.NoModifier) == "F13"  # a function key alone
    assert spec_for_key(Qt.Key.Key_A, mod.ShiftModifier) is None  # a typing key
    assert spec_for_key(Qt.Key.Key_5, mod.KeypadModifier | mod.MetaModifier) == "win+NUM5"
    assert spec_for_key(Qt.Key.Key_Control, mod.ControlModifier) is None  # a modifier alone
    for spec in ("ctrl+shift+F8", "alt+A", "F13", "ctrl+PAGEUP", "win+NUM5"):
        assert make_spec(*split_spec(spec)) == spec
    assert split_spec("Shift+Ctrl+f8") == (["ctrl", "shift"], "F8")
    assert pretty("ctrl+shift+F8") == "Ctrl+Shift+F8"
    with pytest.raises(ValueError):
        split_spec("ctrl+shift")


@pytest.mark.windows_only
def test_windows_hotkeys_use_virtual_key_codes():
    from hud.platform.windows import MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, parse_spec

    assert parse_spec("ctrl+shift+F8") == (MOD_NOREPEAT | MOD_CONTROL | MOD_SHIFT, 0x77)
    assert parse_spec("alt+A")[1] == ord("A") and parse_spec("NUM5")[1] == 0x65


@pytest.mark.windows_only
def test_the_build_tells_signed_program_files_from_unsigned(tmp_path):
    import importlib.util
    from pathlib import Path

    import onnxruntime

    spec = importlib.util.spec_from_file_location("build_exe", "packaging/build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    signed = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"  # Microsoft's
    assert build.has_signature(signed)
    plain = tmp_path / "plain.dll"
    data = bytearray(signed.read_bytes()[:4096])
    pe = int.from_bytes(data[0x3C:0x40], "little")
    entry = pe + 24 + 112 + 4 * 8  # (PE32+) the security directory: no signature
    data[entry : entry + 8] = bytes(8)
    plain.write_bytes(bytes(data))
    assert not build.has_signature(plain)
    (tmp_path / "notes.txt").write_text("not a program")
    assert not build.has_signature(tmp_path / "notes.txt")


def test_the_self_test_finds_the_files_the_app_ships():
    from app.__main__ import _check_files

    info = _check_files()  # (what a build's self-test checks in the bundle)
    assert "azur-promilia" in info and "'es'" in info and "NotoSans" in info
