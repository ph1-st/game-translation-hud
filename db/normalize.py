"""Text normalization shared by the database keys and the HUD lookup.

The goal is that two OCR readings of the same on-screen line produce the same key, and that
the key is stable against the OCR's most common confusions:

* NFKC (full-width ASCII, half-width katakana, compatibility forms)
* all whitespace removed (Japanese has no inter-word spaces; OCR inserts random ones)
* ellipsis / leader dots collapsed to one ``…`` (the OCR reads ``……`` as 1-6 dots)
* small kana folded to their large form (っ/つ, ぃ/い ... are the OCR's top confusion)
* punctuation variants unified (brackets, dashes, exclamation/question marks)

``normalize_key`` is lossy on purpose. Keep the raw text next to the key.
"""

from __future__ import annotations

import re
import unicodedata

from db.text import cjk_count

_SMALL_KANA = str.maketrans(
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ",
    "あいうえおつやゆよわアイウエオツヤユヨワカケ",
)
_PUNCT = str.maketrans(
    {
        "『": "「",
        "』": "」",
        "“": "「",
        "”": "」",
        "‘": "「",
        "’": "」",
        "【": "「",
        "】": "」",
        "[": "「",
        "]": "」",
        "［": "「",
        "］": "」",
        "（": "(",
        "）": ")",
        "〜": "ー",
        "～": "ー",
        "~": "ー",  # (NFKC turns the full-width tilde into this one)
        "―": "ー",
        "—": "ー",
        "–": "ー",
        "‐": "ー",
        "-": "ー",
        "－": "ー",
        "!": "！",
        "?": "？",
        ",": "、",
        "．": "。",
    }
)
# Kanji / katakana look-alikes the OCR swaps freely. Folding them makes keys stable; it is
# harmless because both the capture and the HUD go through the same normalization.
_CONFUSABLES = str.maketrans(
    {
        "力": "カ",
        "口": "ロ",
        "二": "ニ",
        "工": "エ",
        "卜": "ト",
        "夕": "タ",
        "千": "チ",
        "八": "ハ",
        "才": "オ",
        "へ": "ヘ",
        "ぺ": "ペ",
        "べ": "ベ",
        "一": "ー",
    }
)
# Characters the OCR mixes up that also tell real words apart (パン/バン, シール/ツール), so
# they stay apart in keys: folded only for the lookup's second look (db/lookup.py), which
# takes a folded match only when it is unambiguous.
_LOOKALIKES = str.maketrans(
    {
        "ソ": "ン", "ゾ": "ン", "シ": "ツ", "ジ": "ツ", "ヅ": "ツ", "り": "リ",
        "ガ": "カ", "ギ": "キ", "グ": "ク", "ゲ": "ケ", "ゴ": "コ",
        "ザ": "サ", "ズ": "ス", "ゼ": "セ", "ダ": "タ", "ヂ": "チ", "デ": "テ", "ド": "ト",
        "バ": "ハ", "パ": "ハ", "ビ": "ヒ", "ピ": "ヒ", "ブ": "フ", "プ": "フ",
        "ベ": "ヘ", "ペ": "ヘ", "ボ": "ホ", "ポ": "ホ", "ヴ": "ウ",
        "が": "か", "ぎ": "き", "ぐ": "く", "げ": "け", "ご": "こ",
        "ざ": "さ", "じ": "し", "ず": "す", "ぜ": "せ", "ぞ": "そ",
        "だ": "た", "ぢ": "ち", "づ": "つ", "で": "て", "ど": "と",
        "ば": "は", "ぱ": "は", "び": "ひ", "ぴ": "ひ", "ぶ": "ふ", "ぷ": "ふ",
        "ぼ": "ほ", "ぽ": "ほ",
    }
)  # fmt: skip
_ELLIPSIS_RE = re.compile(r"[.。・‥…·‧∙]{2,}|…+|‥+")
# Any run of leader dots that is not part of a number / latin token (x1.0, v2.3) is an ellipsis.
_DOTS_RE = re.compile(r"(?<![0-9A-Za-z])[.・‥…·‧∙]+|[.・‥…·‧∙]+(?![0-9A-Za-z])")
_WS_RE = re.compile(r"\s+")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    """Light normalization that keeps the text readable (for display / translation input)."""
    t = unicodedata.normalize("NFKC", text)
    t = _WS_RE.sub("", t)
    t = _ELLIPSIS_RE.sub("…", t)
    return t


def normalize_key(text: str) -> str:
    """Lossy normalization used as the database key and for same-line comparisons."""
    t = normalize_text(text)
    t = t.translate(_PUNCT)
    t = t.translate(_SMALL_KANA)
    t = t.translate(_CONFUSABLES)
    t = _DOTS_RE.sub("…", t)
    return t


def fold_key(key: str) -> str:
    """A key with the look-alikes of :data:`_LOOKALIKES` folded (voiced marks, ソ/ン,
    シ/ツ): what the OCR may have read for it."""
    return key.translate(_LOOKALIKES)


def normalize_char(ch: str) -> str:
    """One character folded like :func:`normalize_key` folds it ("" for whitespace), for
    aligning a raw reading with a stored line character by character."""
    if ch.isspace():
        return ""
    t = unicodedata.normalize("NFKC", ch)
    return _WS_RE.sub("", t).translate(_PUNCT).translate(_SMALL_KANA).translate(_CONFUSABLES)


def looks_like_text(text: str, min_jp: int = 2, min_latin: int = 3) -> bool:
    """Reject OCR noise such as lone digits, dashes or stray brackets."""
    return cjk_count(text) >= min_jp or len(_LATIN_RE.findall(text)) >= min_latin
