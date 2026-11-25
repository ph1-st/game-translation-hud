"""Which language a game shows, from its window: the picture is read once with the
recognizer of each language the HUD knows, and the one that reads text of its own script
best wins. Kana means Japanese, Hangul Korean; Han characters without either, Chinese
(Simplified or Traditional by the characters read).

It takes a few seconds (four recognizers load): it runs when the player asks, on a
background task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from translate.languages import OCR

_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
_HANGUL = re.compile(r"[가-힣]")
_HAN = re.compile(r"[一-鿿]")
MIN_CONF = 0.75
MIN_CHARS = 3  # a script read fewer times than this is no finding


MIN_SIDE = 32  # a smaller window has no text to read


@dataclass
class Finding:
    lang: str | None  # ja | zh-hans | zh-hant | ko, None when no text was found
    counts: dict[str, int]  # characters of each script read confidently
    sample: str  # some of the text read, to show


def grab(window) -> np.ndarray | None:  # noqa: ANN001 - a hud.platform.WindowRef
    """The game window's picture (BGR), or None when it cannot be taken."""
    from hud.platform import current

    if window.client[2] < MIN_SIDE or window.client[3] < MIN_SIDE:
        return None
    grabber = current().grabber()  # (a grabber of this thread's own)
    try:
        frame, _ok = grabber.grab(window.client, window.id)
    finally:
        grabber.close()
    return frame


_SIMPLIFIED = set("们这说时国会来对个为门见话经还发东长么书车马鸟龙气语读学万与两业丽义乐习")
_TRADITIONAL = set("們這說時國會來對個為門見話經還發東長麼書車馬鳥龍氣語讀學萬與兩業麗義樂習")


def chinese_script(texts: list[str]) -> str:
    """"zh-hant" when the texts are written in traditional characters, else "zh-hans"."""
    simp = trad = 0
    for s in texts:
        for ch in s:
            simp += ch in _SIMPLIFIED
            trad += ch in _TRADITIONAL
    return "zh-hant" if trad > simp else "zh-hans"


def detect(frame: np.ndarray, ocr_cfg: dict, device: str = "auto") -> Finding:
    """The language of the text in *frame*, read with each recognizer."""
    from capture.ocr import OcrEngine

    texts: dict[str, list[str]] = {}
    for code, lang in OCR.items():
        if code == "zh-hant":  # (the Simplified recognizer reads both: one pass for Chinese)
            continue
        engine = OcrEngine.from_profile({**ocr_cfg, "lang": lang}, device=device)
        res = engine.read(frame)
        texts[code] = [ln.text for ln in res.lines if ln.conf >= MIN_CONF]
    kana = sum(len(_KANA.findall(t)) for t in texts.get("ja", []))
    hangul = sum(len(_HANGUL.findall(t)) for t in texts.get("ko", []))
    han = sum(len(_HAN.findall(t)) for t in texts.get("zh-hans", []))
    counts = {"kana": kana, "hangul": hangul, "han": han}
    if kana >= MIN_CHARS and kana >= hangul:
        lang, used = "ja", texts["ja"]
    elif hangul >= MIN_CHARS:
        lang, used = "ko", texts["ko"]
    elif han >= MIN_CHARS:
        used = texts["zh-hans"]
        lang = chinese_script(used)
    else:
        return Finding(None, counts, "")
    return Finding(lang, counts, " ".join(used)[:60])
