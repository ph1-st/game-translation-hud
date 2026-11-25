"""Set-region mode: a dialogue area the player chose for any game (here drawn over a
subtitle of the footage), its glass as one window or growing with the text, kept on
screen, the English left-aligned like a message box's or centred, typing in or not; and
how the app turns a game's settings into that profile."""

from __future__ import annotations

import numpy as np
import pytest

from capture.profile import Profile
from hud.covercache import CoverCache
from hud.covers import px
from hud.dialogue import Scene
from hud.dialogue.panes import grow, symmetric
from hud.dialogue.scene import align_of, lay_out
from hud.pipeline import HudCore
from hud.style import get_style
from tools.replay import RecordedEngine, load_fixture

AREA = (0.25, 0.86, 0.50, 0.12)  # around the two subtitle rows of 004636 (1600x900)


def region_profile(profile: Profile, **flags) -> Profile:
    from app.config import GameEntry
    from app.library import with_region

    g = GameEntry("t", "T", keep_glass=flags.get("keep", True), type_in=flags.get("typing", True))
    return Profile.from_dict(with_region(dict(profile.raw), AREA, None, g))


class AnyCrop(RecordedEngine):
    """The fixture's recorded readings for any crop: those inside it (they are in the
    whole frame's coordinates whatever crop they were read from)."""

    def read(self, frame, region=None, min_conf: float = 0.0, fine=False):  # noqa: ANN001, ANN201
        from capture.ocr import OcrLine, OcrResult

        seen: dict[tuple, OcrLine] = {}
        for calls in self.calls.values():
            for d in calls:
                ln = OcrLine.from_dict(d)
                cx, cy = ln.box.x + ln.box.w / 2, ln.box.y + ln.box.h / 2
                if region is None or region.contains_point(cx, cy):
                    seen.setdefault((ln.text, round(cx, 2), round(cy, 2)), ln)
        return OcrResult([ln for ln in seen.values() if ln.conf >= min_conf], 0.0)


def core_on(profile: Profile, lookup, glossary, mode: str, **flags):  # noqa: ANN001, ANN201
    frame, meta = load_fixture("azur-promilia", "004636_subtitle")
    core = HudCore(
        region_profile(profile, **flags),
        lookup,
        glossary,
        CoverCache(None),
        AnyCrop(meta["ocr"]),
        get_style("glow"),
        cover_mode=mode,
    )
    core.tick(frame, lambda: frame, 0.0)
    return core, frame


def panes(core) -> list:  # noqa: ANN001
    return [it for it in core.items() if it.id.startswith("dlg:pane:")]


def test_a_window_over_the_area_that_stays(profile, lookup, glossary) -> None:
    core, frame = core_on(profile, lookup, glossary, "fixed")
    fh, fw = frame.shape[:2]
    sc = core.dialogue.scene
    assert sc is not None and sc.line is not None and sc.layout.all_dialogue
    (pane,) = panes(core)
    x, y, w, h = AREA
    assert pane.frost == (round(x * fw), round(y * fh), round(w * fw), round(h * fh))
    # the text goes (another moment of the game): the window stays, for good
    empty = np.zeros_like(frame)
    for t in np.arange(0.1, 8.0, 0.1):
        core.watch(empty, float(t))
    assert core.dialogue.scene is not None and len(panes(core)) == 1
    assert core.dialogue.scene.line is None  # (the English went with its Japanese)


def test_without_keep_the_glass_goes_with_the_dialogue(profile, lookup, glossary) -> None:
    core, frame = core_on(profile, lookup, glossary, "fixed", keep=False)
    empty = np.zeros_like(frame)
    for t in np.arange(0.1, 8.0, 0.1):
        core.watch(empty, float(t))
    assert core.dialogue.scene is None and not panes(core)


def test_the_glass_grows_with_the_text_and_never_shrinks(profile, lookup, glossary) -> None:
    core, frame = core_on(profile, lookup, glossary, "grow")
    sc = core.dialogue.scene
    (pane,) = panes(core)  # one pane over everything shown so far
    rows = (515, 798, 575, 75)  # the subtitle's two rows
    x, y, w, h = pane.frost
    assert x <= rows[0] and y <= rows[1] and x + w >= rows[0] + rows[2] and y + h >= rows[1] + 70
    before = pane.frost
    assert not grow(sc, (600, 820, 100, 20))  # inside: nothing changes
    assert grow(sc, (380, 800, 80, 30))  # beyond its left edge: it grows
    after = sc.panes[0].rect
    assert after[0] < before[0] and after[0] + after[2] >= before[0] + before[2]
    assert len(sc.panes) == 1 and pane.id.endswith(f":{sc.panes[0].key}")  # the same pane:
    # it widens, never fades out


def test_kept_glass_is_up_from_the_start(profile, lookup, glossary) -> None:
    for mode, shown in (("fixed", 1), ("grow", 0)):
        core, frame = core_on(profile, lookup, glossary, mode)
        core.dialogue.scene = None  # (as at the next start: the cache knows the area)
        core.watch(np.zeros_like(frame), 0.1)
        assert core.dialogue.scene is not None, mode
        assert len(panes(core)) == shown, mode  # the window at once; the pane with the text


def test_the_english_types_in_or_is_there_at_once(profile, lookup, glossary) -> None:
    core, _f = core_on(profile, lookup, glossary, "fixed", typing=True)
    line = next(it for it in core.items() if it.id.startswith("dlg:line:"))
    assert line.reveal is not None
    core, _f = core_on(profile, lookup, glossary, "fixed", typing=False)
    line = next(it for it in core.items() if it.id.startswith("dlg:line:"))
    assert line.reveal is None


@pytest.mark.parametrize(
    ("rows", "want"),
    [
        ([(410, 790, 600, 30), (410, 830, 420, 30)], "left"),  # flush at the box's left edge
        ([(560, 790, 480, 30), (600, 830, 400, 30)], "center"),  # centred rows
        ([(410, 790, 780, 30)], "center"),  # one row across the whole box: centred is safe
    ],
)
def test_the_rows_tell_how_the_game_aligns_them(rows, want) -> None:
    from capture.profile import DialogueLayout, Region

    lay = DialogueLayout("r", (0.86, 0.98), Region(0.25, 0.86, 0.5, 0.12), align="auto")
    assert align_of(lay, rows, 1600, 900) == want


def test_left_aligned_glass_and_english_start_at_the_box_edge(profile, lookup, glossary) -> None:
    core, frame = core_on(profile, lookup, glossary, "grow")
    fh, fw = frame.shape[:2]
    sc: Scene = core.dialogue.scene
    sc.align = "left"
    rect = symmetric(sc, (420, 800, 300, 30), fw, fh)
    assert rect[0] <= 420 and rect[0] >= px(sc.layout.glass, fw, fh)[0]
    assert rect[0] + rect[2] < fw / 2 + 100  # not mirrored about the middle
    lay_out(sc, "A line of English that is rather long, to wrap.", fw, fh)
    assert sc.text_box[0] < 450 and sc.text_box[0] >= px(sc.layout.panel, fw, fh)[0] - 10


def test_a_games_settings_make_its_profile(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GTHUD_DATA", str(tmp_path))
    from app.config import GameEntry, Settings
    from app.library import Library, parse_region, region_text

    lib = Library(Settings())
    g = GameEntry("vn", "A Visual Novel", "_generic")
    assert lib.cover_mode(g) == "natural"  # no area yet
    g.region = region_text((0.05, 0.72, 0.9, 0.25))
    g.name_region = region_text((0.05, 0.66, 0.25, 0.05))
    g.keep_glass, g.glass_style = True, "window"
    assert parse_region(g.region) == pytest.approx((0.05, 0.72, 0.9, 0.25))
    assert lib.cover_mode(g) == "fixed"
    prof = lib.profile(g)
    (lay,) = prof.dialogue_layouts.values()
    assert lay.all_dialogue and lay.keep and lay.align == "auto" and lay.glass is not None
    assert lay.speaker is not None and lay.glass.y == pytest.approx(0.72)
    assert prof.dialogue is not None and prof.dialogue.h == pytest.approx(0.25)
    g.region_on = False
    assert lib.cover_mode(g) == "natural"
    # a built-in area (Azur Promilia) takes the player's keep / type-in choices
    ap = GameEntry("ap", "AP", "azur-promilia", keep_glass=True, type_in=False)
    layouts = lib.profile(ap).dialogue_layouts.values()
    assert layouts and all(la.keep and not la.reveal and not la.all_dialogue for la in layouts)
    assert lib.cover_mode(ap) == "grow"
