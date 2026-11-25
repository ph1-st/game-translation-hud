from db.lookup import Lookup
from db.store import Line


def mk(key_text: str, seq: int, en: str | None = "en") -> Line:
    from db.templates import templatize

    t = templatize(key_text)
    return Line(
        key=t.key,
        source_text=key_text,
        speaker=None,
        screen_type="dialogue",
        is_template=t.is_template,
        translation=en,
        tier="machine",
        source="batch",
        chapter=None,
        seq=seq,
        game_version=None,
    )


LINES = [
    mk("あなたは変わらないね。", 1, "You haven't changed."),
    mk("あなたは変わってしまった。", 2, "You have changed."),
    mk("ここは原初の海。", 3, "This is the primordial sea."),
    mk("今度こそ悔いのないように、ね。", 4, "This time, no regrets, okay?"),
    mk("残り30秒", 5, "{n} seconds left"),
]


def test_exact_and_fuzzy():
    lk = Lookup(LINES)
    assert lk.match("ここは原初の海。").kind == "exact"
    m = lk.match("こは.原初の海。")  # OCR glitch
    assert m is not None and m.kind == "fuzzy" and m.line.seq == 3
    assert lk.match("まったく関係のない文章") is None


def test_prefix_ambiguity_resolved_by_sequence():
    lk = Lookup(LINES)
    # "あなたは変わ" is a prefix of two lines: ambiguous without context
    assert lk.match("あなたは変わ", remember=False) is None
    lk.last_seq = 1
    m = lk.match("あなたは変わ")
    assert m is not None and m.kind == "prefix" and m.line.seq == 2
    # unambiguous prefix works without context
    lk.reset_sequence()
    m = lk.match("今度こそ悔い")
    assert m is not None and m.kind == "prefix" and m.line.seq == 4


def test_template_fill():
    lk = Lookup(LINES)
    m = lk.match("残り45秒")
    assert m is not None and m.kind == "exact" and m.translation == "45 seconds left"


def test_a_placeholder_without_a_value_goes():
    from db.templates import fill

    # a reading matched to a line with more numbers ("3/5" read as "35"): no "{n}" shown
    assert fill("Harvest crops {n} times {n}/{n}", ["5", "35"], None) == "Harvest crops 5 times 35"
    assert fill("Unripe Starfull Fruit ×{n}", [], None) == "Unripe Starfull Fruit"
    assert fill("{n}/{n}", ["0", "1"], None) == "0/1"


def test_a_misread_voiced_mark_still_finds_its_line_exactly():
    lk = Lookup([mk("キボを呼び戻す", 1, "Call back your Kibo"), mk("ボイス", 2, "Voice")])
    for reading in ("キポを呼び戻す", "キホを呼び戻す", "ポイス"):  # ボ read as ポ / ホ
        m = lk.match(reading)
        assert m is not None and m.kind == "exact", reading
    assert lk.match("ソールを起こす") is None  # nothing like it


def test_look_alike_lines_are_never_confused():
    lk = Lookup([mk("パン", 1, "Bread"), mk("バン", 2, "Van")])
    assert lk.match("パン").translation == "Bread" and lk.match("バン").translation == "Van"
    assert lk.match("ハン") is None  # either of them: neither is taken


def test_a_reading_is_looked_up_once_until_the_database_changes():
    lk = Lookup([mk("ここは原初の海。", 1, "This is the primordial sea.")])
    assert lk.match("見張りの交代時間") is None
    assert "見張りの交代時間" in lk._memo
    lk.add(mk("見張りの交代時間", 2, "The watch rotation"))  # a live translation
    m = lk.match("見張りの交代時間")
    assert m is not None and m.translation == "The watch rotation"


def test_the_players_name_is_a_word_of_its_own():
    from db.templates import templatize

    assert templatize("アイテムを使う", "アイ").key == "アイテムを使う"  # not the アイ of アイテム
    assert templatize("アイ、こっちだ", "アイ").key == "{player}、こつちだ"
    assert templatize("ルルカが来た", "ルカ").key == "ルルカが来た"
    assert templatize("Wrenさん", "Wren").key == "{player}さん"


def test_a_number_with_a_thousands_separator_is_one_number():
    from db.templates import fill, templatize

    t = templatize("1,000ゴールド")
    assert t.key == "{n}ゴールド" and t.values == ["1000"]
    assert fill("{n} Gold", t.values, None) == "1000 Gold"
    assert fill("Max {n}", [], None) == "Max"  # (a letter x is no "×")


def test_without_a_name_set_the_reading_tells_the_players_name():
    lk = Lookup([mk("はじめまして、{player}です。", 1, "Nice to meet you, I'm {player}.")])
    m = lk.match("はじめまして、アリスです。")
    assert m is not None and m.translation == "Nice to meet you, I'm アリス."


def test_the_start_of_lines_that_differ_waits_for_more():
    lines = [
        mk("ありがとう、助かったよ。", 10, "Thanks, you saved me."),
        mk("ありがとうございます！", 20, "Thank you so much!"),
    ]
    lk = Lookup(lines)
    assert lk.match("ありがとう", remember=False) is None  # either line: no guess
    m = lk.match("ありがとう", remember=False, after_seq=19)  # what comes next in the script
    assert m is not None and m.translation == "Thank you so much!"
    assert lk.match("ありがとうござ", remember=False).translation == "Thank you so much!"


def test_an_untranslated_misreading_gives_way_to_its_translated_line():
    lk = Lookup([mk("メイソストーリー", 1, None), mk("メインストーリー", 2, "Main Story")])
    assert lk.match("メイソストーリー").translation == "Main Story"


def test_a_label_read_without_its_full_stop_still_matches():
    lk = Lookup([mk("ない。", 1, "None.")])
    assert lk.match("ない").translation == "None."


def test_a_big_database_answers_unknown_text_quickly():
    import random
    import time

    rnd = random.Random(1)
    kana = [chr(c) for c in range(0x30A1, 0x30F6)] + [chr(c) for c in range(0x3041, 0x3094)]
    lines = [
        mk("".join(rnd.choice(kana) for _ in range(rnd.randint(4, 30))), i, "x")
        for i in range(100_000)
    ]
    lk = Lookup(lines)

    def per_lookup() -> float:
        t0 = time.perf_counter()
        for i in range(20):
            lk.match("".join(rnd.choice(kana) for _ in range(10 + i % 10)), remember=False)
        return (time.perf_counter() - t0) / 20

    # the best of three rounds (a busy machine is no failure); about 25 ms on a desktop,
    # while a Python loop over every key would take a second
    per = min(per_lookup() for _ in range(3))
    assert per < 0.25, f"{per * 1000:.0f} ms a lookup"


# ---- words the game fills in, lines that keep their numbers ------------------------------
def line(key: str, translation: str | None, seq: int = 1) -> Line:
    return Line(key=key, source_text=key, speaker=None, screen_type="ui",
                is_template="{" in key, translation=translation, tier="machine",
                source="batch", chapter=None, seq=seq, game_version=None)


def test_a_word_the_game_fills_in_is_translated_when_the_database_knows_it():
    from db.templates import templatize

    lk = Lookup([line(templatize("{s}を手に入れた！").key, "Got {s}!", 1),
                 line(templatize("回復薬").key, "Potion", 2)])
    assert lk.match("回復薬を手に入れた！").translation == "Got Potion!"
    assert lk.match("星の剣を手に入れた！").translation == "Got 星の剣!"  # unknown: as read


def test_a_line_whose_translation_has_no_numbers_matches_only_its_own():
    from db.templates import keyed

    t, translation = keyed("第3章", "Chapter Three")
    assert t.key == "第3章" and not t.is_template and translation == "Chapter Three"
    t2, templated = keyed("残り30秒", "30 seconds left")
    assert t2.key == "残り{n}秒" and templated == "{n} seconds left"
    lk = Lookup([line(t.key, translation), line(t2.key, templated, 2)])
    assert lk.match("第3章").translation == "Chapter Three"
    assert lk.match("第4章") is None
    assert lk.match("残り5秒").translation == "5 seconds left"


def test_translations_take_the_placeholders_their_numbers_allow():
    from db.templates import keyed, template_translation, templatize, with_player

    t = templatize("所持金1,000G 残り3回")
    assert template_translation("1000 G, 3 left", t) == "{n} G, {n} left"  # 1,000 is 1000
    assert template_translation("3 left, 1000 G", t) is None  # reordered: no template
    assert template_translation("{n} G, {n} left", t) == "{n} G, {n} left"
    assert template_translation("{n} left", t) is None
    hello, en = keyed("ようこそ、リバーV。", "Welcome, リバーV.", player_name="リバーV")
    assert hello.key == "ようこそ、{player}。" and en == "Welcome, {player}."
    assert with_player("リバーVさん、アイテムを", "リバーV") == "{player}さん、アイテムを"
    assert with_player("アイテム", "アイ") == "アイテム"  # a word of its own only


def test_a_glossary_file_serves_every_language_it_holds(tmp_path):
    from db.glossary import Glossary
    from db.store import Name

    path = tmp_path / "glossary.yaml"
    path.write_text(
        "- ja: ルミィ\n  en: Lumie\n  es: Lumi\n  type: character\n"
        "- jp: 原初の海\n  en: Primordial Sea\n  type: place\n"
        "- ko: 루미\n  translation: Lumie\n  type: character\n"
        "- ja: 星臨者\n  type: title\n", encoding="utf-8")
    en = Glossary.load(path, "en")
    assert dict(en.pairs()) == {"ルミィ": "Lumie", "原初の海": "Primordial Sea", "루미": "Lumie"}
    assert [n.source for n in en.characters()] == ["ルミィ", "루미"]
    es = Glossary.load(path, "es-419")  # Latin American Spanish takes the es entries
    assert dict(es.pairs())["ルミィ"] == "Lumi" and "原初の海" not in dict(es.pairs())
    names = Glossary.from_names([Name("テララ", "Terara", "character")])
    assert len(Glossary.combine(en, names)) == 4
    assert en.translate("原初の海") == "Primordial Sea"



def test_a_digit_glued_to_a_label_is_an_icon_only_when_the_rest_is_a_line():
    lk = Lookup([mk("星祝", 1, "Star Blessing"), mk("大地の祝福を受けし者", 2, "Earth-Blessed")])
    m = lk.match("6星祝")  # the lock icon beside the tab, read as a digit
    assert m is not None and m.kind == "exact" and m.translation == "Star Blessing"
    assert lk.match("5大地の祝福") is None  # never the start of a line without its number
