"""The cover protocol on footage frames (tests/frames/azur-promilia), through the production
path with the OCR results recorded when the fixtures were made.

Reference positions (icons, arrows, badges) were measured on the frames independently of
the HUD (a plain grey threshold over each element) and checked by eye.
"""

from __future__ import annotations

import pytest

from hud.dialogue.scene import lay_out
from hud.extent import PAD
from hud.layout import PAD_X, intersects
from hud.textwidth import text_width

STAT_ICON_RIGHT = 1050  # the stat icons span x 1037-1050 in every row
STAT_TEXT_LEFT = 1063  # 攻撃力 / 物理防御力 / 魔法防御力 start here
LIST_ARROW_RIGHT = 1095  # the ▶ / ▼ arrows span x 1088-1095
LIST_TEXT_LEFT = 1111
NEW_BADGE_LEFT = 1400  # the red NEW badges span x 1400-1435
KEY_BADGE_LEFT = 1532  # the G and Z key badges span x 1532-1550
pytestmark = pytest.mark.footage


def fills(core):
    return [it for it in core.items() if it.style == "label"]


def by_text(core, text):
    found = [it for it in core.items() if it.lines and " ".join(it.lines) == text]
    assert found, f"no cover with {text!r}: {[it.lines for it in core.items()]}"
    return found[0]


def right(rect):
    return rect[0] + rect[2]


# ---- what is covered: the matched Japanese, nothing else -------------------------------
def test_stat_icons_are_never_covered(run):
    core, _ = run("004659_stats")
    for text in ("ATK", "Physical DEF", "Magic DEF"):
        it = by_text(core, text)
        assert it.rect[0] > STAT_ICON_RIGHT + 1, (text, it.rect)
        # ...while the Japanese is covered from its first stroke, within a couple of pixels
        assert STAT_TEXT_LEFT - PAD - 1 <= it.rect[0] <= STAT_TEXT_LEFT - 1, (text, it.rect)


def test_list_rows_leave_arrows_and_new_badges_bare(run):
    for name in ("004715_list_new", "004720_list"):
        core, _ = run(name)
        rows = [
            it for it in fills(core) if it.lines[0].split()[0] in ("Details", "World", "Ultimate")
        ]
        rows = [it for it in rows if it.rect[0] > 1000]
        assert len(rows) == 6, [it.lines for it in rows]
        for it in rows:
            assert it.rect[0] > LIST_ARROW_RIGHT + 1, (name, it.lines, it.rect)
            assert it.rect[0] <= LIST_TEXT_LEFT - 1, (name, it.lines, it.rect)
            assert right(it.rect) < NEW_BADGE_LEFT - 1, (name, it.lines, it.rect)


def test_key_hints_leave_badges_bare_and_grow_away_from_them(run):
    core, _ = run("004653_walk")
    a = by_text(core, "Kibo Star Link")
    b = by_text(core, "Call back your Kibo")
    for it in (a, b):
        assert right(it.rect) < KEY_BADGE_LEFT - 1, it.rect
        assert it.align == "right"
    assert a.font_px == b.font_px


def test_a_reading_with_an_icon_read_as_a_character_covers_only_the_text(run):
    # the tab icon is read as 金 and the database even stores 金詳細: only 詳細 is covered
    core, frame = run("004659_stats")
    it = by_text(core, "Details")
    assert it.rect[0] >= 126  # the icon spans x 111-124
    # (the locked tab's lock icon, read as 6 at full size, is read as 台 by the whole frame's
    # quicker read: that tab is not matched, the price of reading the screen once)


def test_unmatched_text_is_not_covered(run):
    # the paragraph's last row "た。" is not a database line: it stays as it is
    core, _ = run("004711_paragraph")
    ta = (1084, 641, 27, 20)
    assert not any(intersects(it.rect, ta) for it in core.items())
    # the NEW badges are read (as NLW, NET...) but are not translated text
    core, _ = run("004715_list_new")
    badges = (NEW_BADGE_LEFT, 355, 40, 200)
    assert not any(intersects(it.rect, badges) for it in core.items())


# ---- size and shape ---------------------------------------------------------------------
def test_english_capitals_are_four_fifths_of_the_japanese(run):
    from hud.textwidth import METRICS, TEXT_SCALE

    core, _ = run("004720_list")
    it = by_text(core, "Details 1")
    cap = METRICS["cap"] * it.font_px
    # the glyphs are 14 px tall (13.7-14.3 measured): capitals of 11 px, 20 % smaller
    assert abs(cap - TEXT_SCALE * it.row_h) <= 1.0
    assert 10.0 <= cap <= 12.0


def test_equal_looking_text_gets_equal_size(run):
    core, _ = run("004720_list")
    rows = [it for it in fills(core) if it.rect[0] > 1080 and it.rect[1] < 560]
    assert len({it.font_px for it in rows}) == 1, [(it.lines, it.font_px) for it in rows]
    core, _ = run("004659_stats")
    stats = [by_text(core, t) for t in ("ATK", "Physical DEF", "Magic DEF")]
    assert len({it.font_px for it in stats}) == 1
    tabs = [it for it in fills(core) if it.rect[1] > 850]
    assert len(tabs) >= 3 and len({it.font_px for it in tabs}) == 1, [
        (it.lines, it.font_px) for it in tabs
    ]


def test_covers_are_never_taller_and_never_wider_than_the_english_needs(run):
    for name in ("004653_walk", "004659_stats", "004711_paragraph", "004720_list"):
        core, _ = run(name)
        for c in core.ui_board.snapshot() + core.dialogue.board.snapshot():
            g = c.geom
            assert g.box[3] == g.cover[3], (name, g.lines)  # never taller
            if g.box[2] > g.cover[2]:  # widened: by exactly what the English needs
                need = max(text_width(line, g.font_px * g.stretch / 90) for line in g.lines)
                assert g.box[2] <= need + 2 * PAD_X + 2, (name, g.lines, g.box, need)
            assert g.box[0] <= g.cover[0] and right(g.box) >= right(g.cover)


def test_centred_title_card_grows_both_ways(run):
    # three centred rows of different sizes share their centres (x 800) and, roughly,
    # their left edges: they are a centred stack, not a left-aligned column
    core, _ = run("004646_title")
    for text in ("Budding Valley, Chapter 1", "Starbound Beginnings", "Unlocked"):
        it = by_text(core, text)
        assert it.align == "center", text
        jp_centre = it.under[0] + it.under[2] / 2
        assert abs((it.rect[0] + it.rect[2] / 2) - jp_centre) <= 1.5, (text, it.rect, it.under)
        assert abs(jp_centre - 800) <= 4, text


def test_left_aligned_column_grows_right(run):
    core, _ = run("004659_stats")
    it = by_text(core, "Physical DEF")
    assert it.rect[0] == by_text(core, "ATK").rect[0]  # flush, like the Japanese column
    assert it.align == "left"


def test_paragraph_rows_are_one_block(run):
    core, _ = run("004711_paragraph")
    blocks = [it for it in fills(core) if len(it.lines) > 1]
    assert len(blocks) == 1
    b = blocks[0]
    assert len(b.lines) <= 7  # the seven Japanese rows, never more
    assert b.rect[1] <= 485 and right(b.rect) > 1400  # covers the first row ...
    assert b.rect[1] + b.rect[3] >= 633  # ... to the seventh
    assert b.rect[1] + b.rect[3] < 641  # and not the unmatched last row
    assert " ".join(b.lines).startswith("A girl whose origins")
    assert " ".join(b.lines).endswith("gathered around her.")


def _union(rects):
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[0] + r[2] for r in rects), max(r[1] + r[3] for r in rects)
    return x0, y0, x1 - x0, y1 - y0


class _Union:
    """The panes of a dialogue line, as one: the area they cover."""

    def __init__(self, rects) -> None:
        assert rects, "no panes"
        self.rect = _union(rects)


# ---- the dialogue band ------------------------------------------------------------------
def test_dialogue_band_has_one_owner(run, profile):
    for name in ("004636_subtitle", "004653_walk"):
        core, frame = run(name)
        fh, fw = frame.shape[:2]
        territory = core.dialogue.territory_px(fw, fh)
        for c in core.ui_board.snapshot():
            assert not any(intersects(c.geom.box, t) for t in territory), (name, c.text)
        panels = [it for it in core.items() if it.id.startswith(("dlg:panel:", "dlg:pane:"))]
        texts = [it for it in core.items() if it.id.startswith("dlg:line:")]
        assert panels and len(texts) == 1
        # nothing else is drawn on the panel
        area = _union([p.rect for p in panels])
        for it in core.items():
            if it.style == "label" and intersects(it.rect, area):
                raise AssertionError((name, it.lines))


def test_cutscene_subtitle_is_one_centred_line_on_the_panel(run):
    core, _ = run("004636_subtitle")
    panel = _Union([it.rect for it in core.items() if it.id.startswith("dlg:pane:")])
    (text,) = [it for it in core.items() if it.id.startswith("dlg:line:")]
    assert text.align == "center"
    assert " ".join(text.lines).startswith("Everyone here is kind.")
    # the panel covers both Japanese rows (y 796-871, x 510-1090)
    x, y, w, h = panel.rect
    assert x <= 505 and x + w >= 1095 and y <= 794 and y + h >= 873


def test_npc_line_and_speaker_name(run):
    core, _ = run("004653_walk")
    panel = _Union([it.rect for it in core.items() if it.id.startswith("dlg:pane:")])
    (text,) = [it for it in core.items() if it.id.startswith("dlg:line:")]
    assert text.lines == ["If we hurry, we might even make it in time for breakfast!"]
    name = by_text(core, "Terara")
    assert name.rect[1] + name.rect[3] <= panel.rect[1]  # a fixed box above the panel
    assert name.font_px > text.font_px  # a size up


# ---- permanent standardization ----------------------------------------------------------
def test_same_line_same_place_gets_the_cached_block(run, tmp_path):
    from hud.covercache import CoverCache

    cache = CoverCache(tmp_path / "covers.json")
    first, _ = run("004715_list_new", cache, settle=True)
    cache.save()
    # five seconds later, after a restart: the rows that stayed (one of them now reads a
    # pixel higher) get exactly their blocks back
    again, _ = run("004720_list", CoverCache(tmp_path / "covers.json"))
    for text in ("Details 1", "Details 2"):
        a, b = by_text(first, text), by_text(again, text)
        assert (a.rect, a.font_px, a.lines) == (b.rect, b.font_px, b.lines), text
    # a row the list moved to another place is another place
    assert by_text(first, "World Lore 2").rect[1] == 478
    assert by_text(again, "World Lore 2").rect[1] == 426


def test_a_lines_own_dash_is_no_obstacle(run):
    # the quote's rows end in a dash the stored lines lack ("世界の意思よ——", read as 一):
    # the English stays in the rows' column and runs on over the dash
    core, _ = run("004720_list")
    rows = [by_text(core, t) for t in ("Gathered in the azure sky,", "O will of the world,")]
    rows.append(by_text(core, "lend me your strength!"))
    assert len({it.rect[0] for it in rows}) == 1, [it.rect for it in rows]


def test_dialogue_english_has_one_place_per_layout(run):
    # the cutscene layout: rows grow upwards; the English (two lines here) ends on the last
    # row and its lines lie on the rows, centred on the panel, whatever typed out so far
    core, _ = run("004636_subtitle")
    d = core.dialogue
    (text,) = [it for it in core.items() if it.id.startswith("dlg:line:")]
    panel = _Union([it.rect for it in core.items() if it.id.startswith("dlg:pane:")])
    assert len(text.lines) == 2
    assert abs(text.pitch - 44) <= 1  # the layout's own row pitch (rows at y 802 and 846)
    assert abs(text.row_top - 802) <= 4  # so the first line is on the first row
    px, _py, pw, _ph = panel.rect
    assert abs((text.rect[0] + text.rect[2] / 2) - (px + pw / 2)) <= 1  # centred
    # the same line read while it is still typing out goes exactly the same place
    sc = d.scene
    before = sc.text_box
    fh, fw = _.shape[:2]
    lay_out(sc, sc.line.text, fw, fh)
    assert sc.text_box == before
    # the NPC layout: the English on the one row the Japanese is on, not below it
    core, _ = run("004653_walk")
    (text,) = [it for it in core.items() if it.id.startswith("dlg:line:")]
    assert len(text.lines) == 1 and abs(text.row_top - 687) <= 1


def test_the_scene_stays_while_the_next_line_comes_in(run):
    import cv2

    core, frame = run("004636_subtitle")
    x, y, w, h = 500, 790, 600, 90  # the subtitle rows
    empty = frame.copy()  # the band without text
    empty[y : y + h, x : x + w] = cv2.medianBlur(frame[y : y + h, x : x + w], 31)
    d = core.dialogue
    d.watch(empty, 1.0)  # the line's Japanese goes
    d.watch(empty, 1.05)
    assert d.scene is not None and d.scene.line is None
    d.watch(frame, 2.0)  # a second later the next line is in the band, not read yet:
    assert d.scene is not None  # the panel stays and paints it over (no flash of Japanese)
    d.watch(empty, 3.0)  # the band stays empty: the scene is over
    assert d.scene is None


def test_the_players_text_size_scales_every_translation(profile, lookup, glossary):
    from hud.covercache import CoverCache
    from hud.pipeline import HudCore
    from hud.style import get_style
    from tools.replay import RecordedEngine, load_fixture

    frame, meta = load_fixture("azur-promilia", "004720_list")
    sizes = {}
    for scale in (1.0, 0.8, 1.3):
        core = HudCore(profile, lookup, glossary, CoverCache(None), RecordedEngine(meta["ocr"]),
                       get_style("glow"), text_scale=scale)  # fmt: skip
        core.tick(frame, lambda: frame, 0.0)
        sizes[scale] = by_text(core, "Details 1").font_px
    assert sizes[0.8] < sizes[1.0] < sizes[1.3]
    assert abs(sizes[0.8] - 0.8 * sizes[1.0]) <= 1.0


def test_white_text_keeps_the_japanese_colours_for_painting_it_over(profile, lookup, glossary):
    from hud.covercache import CoverCache
    from hud.pipeline import HudCore
    from hud.style import get_style
    from tools.replay import RecordedEngine, load_fixture

    frame, meta = load_fixture("azur-promilia", "004720_list")
    core = HudCore(profile, lookup, glossary, CoverCache(None), RecordedEngine(meta["ocr"]),
                   get_style("glow"), plain_text=True)  # fmt: skip
    core.tick(frame, lambda: frame, 0.0)
    labels = fills(core)
    assert labels and all(it.plain for it in labels)
    assert all(it.fg is not None for it in labels)  # (what the Japanese is erased by)
