"""The cover protocol's rules on synthetic frames and small inputs."""

from __future__ import annotations

import cv2
import numpy as np

from capture.change import ChangeTracker
from db.lookup import Lookup
from db.store import Line
from db.templates import templatize
from hud.anchors import HARD_FAIL, HOLDS, anchor_state, make_anchor, settled
from hud.board import Board, Cover
from hud.covercache import CoverCache, place
from hud.extent import align_span
from hud.layout import MIN_STRETCH, PAD_X, Geom, drawn_size, layout_block, layout_line, widen
from hud.looks import Looks
from hud.matching import match_reading
from hud.pipeline import BAND_STEP, bands
from hud.style import get_style
from hud.textwidth import STRETCH, text_width


def line(jp: str, en: str | None, speaker: str | None = None) -> Line:
    t = templatize(jp)
    return Line(t.key, jp, speaker, "ui", t.is_template, en, "human", "fixture", None, 1, None)


# ---- only the matched text --------------------------------------------------------------
def test_alignment_leaves_out_what_the_line_does_not_have():
    assert align_span("日物理防御力", "物理防御力") == (1, 6)  # an icon read as 日
    assert align_span("世界見聞・1NEW", "世界見聞・2") == (0, 6)  # a glued badge
    assert align_span("レまなら朝こはん", "いまなら朝ごはん") == (0, 8)  # misreads are text
    assert align_span("みなさん、優", "みなさん、優しい方ばかりです。") == (0, 6)  # typing out


def test_an_icon_the_database_stored_with_the_label_is_not_covered():
    # the capture stored the tab as 金詳細 (its icon read as 金) next to 詳細
    lk = Lookup([line("金詳細", "Details"), line("詳細", "Details")])
    hit = match_reading(lk, "金詳細")
    assert hit is not None and hit.span == (1, 3) and hit.translation == "Details"
    # a number is part of its label: 2キャベッコ is not trimmed to キャベッコ
    lk = Lookup([line("2キャベッコ", "{n} Kyabekko"), line("キャベッコ", "Kyabekko")])
    assert match_reading(lk, "2キャベッコ").span == (0, 6)


def test_untranslated_and_unknown_text_is_not_matched():
    lk = Lookup([line("ボーナス", None), line("詳細", "Details")])
    assert match_reading(lk, "ボーナス") is None
    assert match_reading(lk, "まったく関係のない文章") is None


# ---- size: the Japanese size, one per look ---------------------------------------------
def test_sizes_snap_to_the_games_text_sizes():
    looks = Looks()
    assert looks.snap(13.6, 900) == 13.6
    assert looks.snap(14.3, 900) == 13.6  # the same size, measured a little off
    assert looks.snap(16.0, 900) == 16.0  # another size
    assert looks.snap(26.0, 1800) == 2 * 13.6 or looks.snap(27.2, 1800) == 27.2


def test_sizes_and_covers_persist(tmp_path):
    c = CoverCache(tmp_path / "covers.json")
    c.looks.snap(13.6, 900)
    g = Geom((100, 100, 80, 18), (100, 100, 120, 18), 19, "left", ["Details"], 102, 14, 0)
    assert c.put("Details", g, 1600, 900)
    c.save()
    again = CoverCache(tmp_path / "covers.json")
    assert abs(again.looks.snap(14.0, 900) - 13.6) < 1e-3
    assert again.find("Details", (101, 100, 79, 18), 1600, 900) is not None


# ---- shape: never taller, wider only by what the English needs ---------------------------
def test_shorter_english_keeps_the_japanese_width():
    cover = (500, 300, 200, 20)
    assert widen(cover, 150, "center", [], 1600) == (cover, 0)


def test_longer_english_grows_sideways_the_way_the_text_is_laid_out():
    cover = (500, 300, 100, 20)
    box, short = widen(cover, 160, "center", [], 1600)
    assert box == (470, 300, 160, 20) and short == 0  # both ways, never taller
    assert widen(cover, 160, "left", [], 1600)[0] == (500, 300, 160, 20)
    assert widen(cover, 160, "right", [], 1600)[0] == (440, 300, 160, 20)


def test_growth_never_covers_a_neighbour_and_goes_to_the_free_side():
    cover = (500, 300, 100, 20)
    badge = (610, 302, 16, 16)  # a key badge right of the label
    box, short = widen(cover, 160, "center", [badge], 1600)
    assert box[0] + box[2] <= badge[0] - 2 and box[2] == 160 and short == 0
    # boxed in on both sides: as wide as it can be, the rest is reported short
    box, short = widen(cover, 160, "center", [badge, (470, 300, 10, 20)], 1600)
    assert box[0] >= 482 and box[0] + box[2] <= 608 and short > 0


def test_boxed_in_text_is_condensed_then_made_smaller_never_past_its_box():
    cover = (500, 300, 100, 20)
    walls = [(490, 300, 8, 20), (604, 300, 8, 20)]
    for text in ("Call back Kibo", "Call back your Kibo", "Call back your Kibo to your side"):
        g = layout_line(cover, text, 21, "center", walls, 1600, 302, 16)
        inner = g.box[2] - 2 * PAD_X
        assert g.box[3] == cover[3]  # never taller
        assert text_width(g.lines[0], drawn_size(g.font_px, g.stretch)) <= inner + 0.5
        if text_width(text, drawn_size(21, MIN_STRETCH)) <= inner:
            assert g.font_px == 21 and g.stretch < STRETCH, text  # condensing is enough
        else:
            assert g.stretch == MIN_STRETCH and g.font_px < 21, text  # then smaller


def test_rows_of_two_colours_are_two_texts_not_a_paragraph():
    from capture.ocr import OcrLine
    from capture.profile import Region
    from hud.covers import Row, group_blocks
    from hud.extent import Extent
    from hud.matching import Hit

    def row(text: str, y: int, fill) -> Row:
        db = line(text, "English " + text)
        ink = (46, y, 150, 15)
        ext = Extent((44, y - 2, 154, 19), ink, 15.0, [], True, [], fill, (30, 30, 30))
        ocr = OcrLine(text, Region(46 / 1600, y / 900, 150 / 1600, 15 / 900), 0.95)
        return Row(ocr, Hit(db.key, db.translation, (0, len(text)), db, "exact", 100.0), ext)

    title = row("第一章シャルル村の外れにて", 223, (200, 180, 70))  # gold
    objective = row("村長の家へ向かう道を進む", 243, (196, 196, 191))  # grey, right below
    alike = row("村長の家へ向かう道を進む", 243, (205, 186, 80))  # the same gold
    assert group_blocks([title, objective], [15.0, 15.0], [], 1600, 900) == [[0], [1]]
    assert group_blocks([title, alike], [15.0, 15.0], [], 1600, 900) == [[0, 1]]


POPUP = "ローカルリソースの整合性チェックを行います。チェック中はゲームをプレイできません。"


def popup_row(f: np.ndarray, text: str, x: int, y: int):
    """A row of dark text on a light popup, 15 px a character; its reading."""
    from capture.ocr import OcrLine
    from capture.profile import Region

    for i in range(len(text)):
        cv2.rectangle(f, (x + 15 * i + 2, y + 3), (x + 15 * i + 12, y + 16), (40, 40, 40), -1)
    return OcrLine(text, Region(x / 1600, y / 900, 15 * len(text) / 1600, 20 / 900), 0.95)


def test_centred_rows_are_one_paragraph():
    from hud.covers import paragraph_stacks

    f = np.full((900, 1600, 3), 235, np.uint8)
    first = popup_row(f, POPUP[:33], 300, 400)
    centred = popup_row(f, POPUP[33:], 300 + 15 * (33 - 8) // 2, 426)  # a popup's message
    aside = popup_row(f, POPUP[33:], 700, 426)  # neither its left edge nor its middle
    assert paragraph_stacks([first, centred], 1600, 900) == [[first, centred]]
    assert paragraph_stacks([first, aside], 1600, 900) == []


def test_a_line_that_wraps_is_matched_across_its_rows():
    from capture.profile import load_profile
    from hud.covers import CoverMaker

    lk = Lookup([line(POPUP, "Checking local resources. You can't play meanwhile.")])
    maker = CoverMaker(lk, CoverCache(None), load_profile("azur-promilia"))
    x = 300 + 15 * (33 - 8) // 2
    # the first row is most of the line (its last kana misread), the second nothing alone
    f = np.full((900, 1600, 3), 235, np.uint8)
    first = popup_row(f, POPUP[:32] + "フ", 300, 400)
    rest = popup_row(f, POPUP[33:], x, 426)
    assert match_reading(lk, first.text, whole=True).partial
    rows, other = maker.rows(f, [first, rest])
    assert [r.line for r in rows] == [first, rest] and other == []
    assert rows[0].hit is rows[1].hit and rows[0].hit.score > 0.95
    # an unrelated row below a partial one: the row keeps its own match, alone
    f = np.full((900, 1600, 3), 235, np.uint8)
    first = popup_row(f, POPUP[:32] + "フ", 300, 400)
    unrelated = popup_row(f, "設定を保存しました", x, 426)
    rows, other = maker.rows(f, [first, unrelated])
    assert [r.line for r in rows] == [first] and other == [unrelated]


def test_a_paragraph_wraps_into_its_rows_and_gets_wider_not_taller():
    text = "A girl whose origins are shrouded in mystery. She can be a little dense."
    cover = (1080, 480, 200, 40)
    g = layout_block(cover, text, 19, 2, 22.0, "left", [], 1600, 482, 14)
    assert len(g.lines) <= 2 and g.box[3] == cover[3] and g.box[0] == cover[0]
    assert g.box[2] > cover[2]
    need = max(text_width(t, 19) for t in g.lines)
    assert g.box[2] <= need + 2 * 3 + 2  # no white space beyond what the English needs


# ---- the cache: the same line at the same place is the same block -------------------------
def test_the_cached_block_is_reused_exactly_and_never_reshaped(tmp_path):
    c = CoverCache(tmp_path / "covers.json")
    g = Geom((1061, 501, 76, 17), (1061, 501, 105, 17), 20, "left", ["Physical DEF"], 503, 13, 0)
    c.put("Physical DEF", g, 1600, 900)
    assert not c.put("Physical DEF", g.moved(1, 1), 1600, 900)  # written once, never changed
    rec = c.find("Physical DEF", (1062, 502, 74, 16), 1600, 900)  # read a pixel off
    assert place(rec, (1062, 502, 74, 16), ["Physical DEF"], 1600, 900).box == g.box
    rec = c.find("Physical DEF", (1066, 506, 76, 17), 1600, 900)  # the menu moved 5 px
    moved = place(rec, (1066, 506, 76, 17), ["Physical DEF"], 1600, 900)
    assert moved.box == (1066, 506, 105, 17) and moved.font_px == 20  # same shape
    assert c.find("Physical DEF", (1061, 531, 76, 17), 1600, 900) is None  # the next row


def test_a_cache_from_another_hud_version_is_not_used(tmp_path):
    p = tmp_path / "covers.json"
    p.write_text('{"Details": [{"cover": [0.1, 0.1, 0.05, 0.02]}]}', encoding="utf-8")
    assert CoverCache(p).covers == {}


# ---- the watch: pixels remove a cover, nothing else ---------------------------------------
def glyphs(text: str, bg: np.ndarray, at=(40, 60)) -> np.ndarray:
    """White glyphs with a dark outline (like the game's own text) drawn on *bg*."""
    f = bg.copy()
    cv2.putText(f, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.putText(f, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2, cv2.LINE_AA)
    return f


def scene(seed: int, size=(120, 400)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(30, 220, (size[0] // 8, size[1] // 8, 3)).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(small, (size[1], size[0])), (0, 0), 3)


RECT = (34, 34, 260, 36)


def test_the_text_holds_whatever_happens_behind_it():
    f = glyphs("Kibo Star Link", scene(1))
    a = make_anchor(f, RECT, light=True)
    assert a is not None and anchor_state(f, a) == HOLDS
    # the camera cuts to another shot behind the same subtitle
    assert anchor_state(glyphs("Kibo Star Link", scene(2)), a) == HOLDS
    # a hover tint over everything
    assert anchor_state(cv2.convertScaleAbs(f, alpha=0.9, beta=20), a) == HOLDS


def test_the_cover_goes_when_its_text_goes_or_changes():
    f = glyphs("Kibo Star Link", scene(1))
    a = make_anchor(f, RECT, light=True)
    assert anchor_state(scene(1), a) != HOLDS  # gone
    assert anchor_state(glyphs("Call back Kibo", scene(1)), a) != HOLDS  # another text
    assert anchor_state(glyphs("Kibo Star Link", scene(1), at=(40, 80)), a) != HOLDS  # moved


def char_cells(text: str, at=(40, 60)) -> list[tuple[int, int]]:
    """The columns of each character of *text* as glyphs() draws it."""
    out = []
    for i in range(len(text)):
        x0 = at[0] + cv2.getTextSize(text[:i], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        x1 = at[0] + cv2.getTextSize(text[: i + 1], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        out.append((x0, x1))
    return out


def test_one_changed_character_is_a_changed_line():
    # a counter ticks under its cover: most strokes stay, one digit's do not
    f = glyphs("Remaining: 13", scene(1))
    a = make_anchor(f, RECT, light=True, cells=char_cells("Remaining: 13"))
    assert anchor_state(f, a) == HOLDS
    assert anchor_state(glyphs("Remaining: 13", scene(2)), a) == HOLDS  # the scene moved
    assert anchor_state(glyphs("Remaining: 14", scene(1)), a) != HOLDS


def test_only_settled_text_is_placed():
    f = glyphs("Kibo Star Link", scene(1))
    a = make_anchor(f, RECT, light=True)
    assert settled(glyphs("Kibo Star Link", scene(2)), a)  # static text, moving scene
    fading = cv2.addWeighted(f, 0.75, scene(1), 0.25, 0)  # still fading in
    assert not settled(fading, a)


def test_the_board_removes_on_the_first_frame_the_text_is_plainly_gone():
    f = glyphs("Kibo Star Link", scene(1))
    a = make_anchor(f, RECT, light=True)
    b = Board("ui", get_style("glow"))
    g = Geom(RECT, RECT, 20, "center", ["Kibo Star Link"], 40, 16, 0)
    assert b.add_if_free(Cover("Kibo Star Link", "Kibo Star Link", g, a))
    assert b.watch(f) == [] and len(b) == 1
    empty = np.full_like(f, 90)
    assert anchor_state(empty, a) == HARD_FAIL
    assert b.watch(empty) == [RECT] and len(b) == 0


def test_the_ocr_never_removes_a_cover(run):
    # a tick that no longer reads a covered line leaves its cover; only the watch decides
    core, frame = run("004720_list")
    n = len(core.ui_board)
    blank = frame.copy()
    blank[:, :] = 128
    core.engine.calls = {k: [] for k in core.engine.calls}  # the OCR now reads nothing
    core.tick(blank, lambda: blank, 1.0)
    assert len(core.ui_board) == n
    core.watch(blank, 1.0)
    assert len(core.ui_board) == 0


# ---- what the OCR reads: what changed, and what the watch freed --------------------------
def test_only_what_changed_is_read_again():
    f = scene(3, (900, 1600))
    t = ChangeTracker()
    assert t.changed(f).all()  # nothing read yet
    t.mark_read(f, None)
    assert not t.changed(f).any()
    g = f.copy()
    cv2.putText(g, "NEW TEXT", (600, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    rows = t.changed(g).any(axis=1)
    plan = bands(rows, [], 1600, 900)
    assert plan is not None and len(plan) == 1
    x, y, w, h = plan[0]
    assert x == 0 and w == 1600 and y < 395 and y + h > 425  # a full-width band...
    assert h % BAND_STEP == 0 and h <= 2 * BAND_STEP  # ...in steps the detector reuses
    assert bands(np.ones_like(rows), [], 1600, 900) is None  # everything: the whole frame
    t.free([(100, 700, 200, 20)])
    assert t.take_freed() == [(100, 700, 200, 20)] and t.take_freed() == []


def test_a_counter_gets_its_new_number_in_place():
    from capture.profile import load_profile
    from hud.covers import CoverMaker, Placement

    def frame_with(digits: str) -> np.ndarray:
        f = np.full((120, 400, 3), 30, np.uint8)
        cv2.putText(f, digits, (110, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 3)
        return f

    maker = CoverMaker(Lookup([]), CoverCache(None), load_profile("azur-promilia"))
    board = Board("ui", get_style("glow"))
    rect = (100, 40, 200, 50)

    def placement(en: str) -> Placement:
        g = Geom(rect, rect, 20, "left", [en], 45, 30, 0)
        return Placement("{n} seconds left", en, g, False, 30.0, True)

    a = frame_with("13")
    assert maker.commit([placement("13 seconds left")], a, a, board, 0.0) == 1
    (c,) = board.snapshot()
    b = frame_with("18")  # the watch may not see it (the 3's strokes are in the 8): the OCR does
    maker.commit([placement("18 seconds left")], b, b, board, 1.0)
    (c2,) = board.snapshot()
    assert c2.id == c.id and c2.text == "18 seconds left" and c2.geom.lines == ["18 seconds left"]
