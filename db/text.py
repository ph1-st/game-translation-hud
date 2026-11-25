"""What kind of text a string is: the writing systems of the games the app reads (Japanese,
Chinese, Korean), sentences, and texts that need no translation.

One module for every reader (the extraction engines, the table analysis, the database's
normalization and importers), so they all agree on what counts as CJK text or a sentence.
"""

from __future__ import annotations

import re

# kana, CJK ideographs (with extension A and the compatibility block), Hangul syllables
# and jamo: the text of the games the app reads
CJK = re.compile(
    "[぀-ヿ㐀-䶿一-鿿豈-﫿"
    "가-힣ᄀ-ᇿ㄰-㆏]"
)
KANA = re.compile("[ぁ-ゖァ-ヺ]")  # hiragana and katakana letters
HAN = re.compile("[一-鿿]")  # CJK unified ideographs
HANGUL = re.compile("[가-힣]")  # Hangul syllables
LETTER = re.compile(r"[^\W\d_]")  # a letter of any script

# Where a sentence ends: Japanese and Chinese punctuation, the ASCII forms, and a period
# after a Hangul syllable (Korean ends sentences with "."; a period after a digit or a
# Latin letter is a decimal point or an abbreviation as often as not).
SENTENCE_END = re.compile("[。！？…!?．｡]|(?<=[가-힣])\\.")
SENTENCE_MARKS = "。！？…!?"  # 。！？…!? (a line ending with one is prose)

# A developer's text, never shown to players: a test marker in brackets or followed by a
# separator ([test] ..., 【テスト】..., test_01, テスト: ...), a line starting with Chinese
# 测试 ("test": the Chinese builds' developers write it that way), a Japanese note that a
# line is not in the game (【実装対象外／...】 "not to be implemented", 【CBT未実装...】 "not
# in the CBT"), or one of the Chinese placeholder phrases. "テストの結果" or "Testament" are
# ordinary words.
_TEST_WORD = "(?:test|测试|テスト|테스트)"  # test 测试 テスト 테스트
_DEV = re.compile(
    rf"^\s*(?:[\[【(（]\s*{_TEST_WORD}"
    rf"|{_TEST_WORD}(?:\s*[:：_#\-\d]|\s*$)"
    "|测试"
    "|【(?:実装対象外|[^】\n]{0,12}未実装))",
    re.IGNORECASE,
)
# 测试文本 (test text), 测试占位 (test placeholder), 标准临时文本 (standard temporary text)
_DEV_ANYWHERE = re.compile("测试文本|测试占位|"
                           "标准临时文本")


def cjk_count(text: str) -> int:
    """Characters of the CJK writing systems (kana, ideographs, Hangul) in *text*."""
    return len(CJK.findall(text))


def has_cjk(text: str, at_least: int = 1) -> bool:
    return cjk_count(text) >= at_least


def kana_count(text: str) -> int:
    return len(KANA.findall(text))


def hangul_count(text: str) -> int:
    return len(HANGUL.findall(text))


def is_sentence(text: str) -> bool:
    """Whether *text* holds sentence punctuation (prose rather than a label)."""
    return SENTENCE_END.search(text) is not None


def skip_reason(text: str) -> str | None:
    """Why a text needs no translation: "dev" for a developer's text, "symbols" for one
    without letters (…, ？？？, 98012901); None for a text to translate."""
    if _DEV.search(text) or _DEV_ANYWHERE.search(text):
        return "dev"
    if not LETTER.search(text):
        return "symbols"
    return None


def wanted(text: str) -> bool:
    """Worth keeping from a game's files: it holds CJK text and is not a developer's."""
    text = text.strip()
    return bool(text) and has_cjk(text) and skip_reason(text) != "dev"
