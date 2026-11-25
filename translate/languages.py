"""Languages: the one a game shows (what the HUD reads: Japanese, Chinese or Korean) and the
ones translations are made in.

Codes are lowercase BCP 47 tags: ``en``, ``es-419``, ``pt-br``, ``zh-hans``. One table holds
every language the app offers, with its own name and how its text is laid out. A language
a person types that is not in the table gets a code made from its name: databases can be
created in any language the model writes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---- the game's language ----------------------------------------------------------------
SOURCES = {
    "ja": "Japanese",
    "zh-hans": "Chinese (Simplified)",
    "zh-hant": "Chinese (Traditional)",
    "ko": "Korean",
}
SOURCE_NATIVE = {
    "ja": "日本語", "zh-hans": "中文（简体）", "zh-hant": "中文（繁體）", "ko": "한국어",
}
# the OCR's recognizer for each (rapidocr's language names)
OCR = {"ja": "japan", "zh-hans": "ch", "zh-hant": "chinese_cht", "ko": "korean"}


@dataclass(frozen=True)
class Language:
    code: str
    name: str  # in English
    native: str  # in the language itself (what players look for in a list)
    you: str = "you"  # stands for the player's name when the HUD knows none
    cased: bool = True  # written with capital letters
    rtl: bool = False  # written right to left
    spaced: bool = True  # words separated by spaces (lines break at spaces)

    @property
    def primary(self) -> str:
        return self.code.split("-")[0]


_UNSPACED = dict(cased=False, spaced=False)
LANGUAGES: tuple[Language, ...] = (
    Language("en", "English", "English", "you"),
    Language("en-gb", "English (UK)", "English (UK)", "you"),
    Language("es-419", "Spanish (Latin America)", "Español (Latinoamérica)", "tú"),
    Language("es", "Spanish (Spain)", "Español (España)", "tú"),
    Language("pt-br", "Portuguese (Brazil)", "Português (Brasil)", "você"),
    Language("pt-pt", "Portuguese (Portugal)", "Português (Portugal)", "tu"),
    Language("fr", "French", "Français", "toi"),
    Language("de", "German", "Deutsch", "du"),
    Language("it", "Italian", "Italiano", "tu"),
    Language("nl", "Dutch", "Nederlands", "jij"),
    Language("pl", "Polish", "Polski", "ty"),
    Language("ru", "Russian", "Русский", "ты"),
    Language("uk", "Ukrainian", "Українська", "ти"),
    Language("tr", "Turkish", "Türkçe", "sen"),
    Language("id", "Indonesian", "Bahasa Indonesia", "kamu"),
    Language("vi", "Vietnamese", "Tiếng Việt", "bạn"),
    Language("th", "Thai", "ไทย", "คุณ", **_UNSPACED),
    Language("ar", "Arabic", "العربية", "أنت", cased=False, rtl=True),
    Language("he", "Hebrew", "עברית", "אתה", cased=False, rtl=True),
    Language("bg", "Bulgarian", "Български", "ти"),
    Language("cs", "Czech", "Čeština", "ty"),
    Language("da", "Danish", "Dansk", "du"),
    Language("el", "Greek", "Ελληνικά", "εσύ"),
    Language("et", "Estonian", "Eesti", "sina"),
    Language("fi", "Finnish", "Suomi", "sinä"),
    Language("hu", "Hungarian", "Magyar", "te"),
    Language("lt", "Lithuanian", "Lietuvių", "tu"),
    Language("lv", "Latvian", "Latviešu", "tu"),
    Language("nb", "Norwegian", "Norsk bokmål", "du"),
    Language("ro", "Romanian", "Română", "tu"),
    Language("sk", "Slovak", "Slovenčina", "ty"),
    Language("sl", "Slovenian", "Slovenščina", "ti"),
    Language("sv", "Swedish", "Svenska", "du"),
    Language("ja", "Japanese", "日本語", "あなた", **_UNSPACED),
    Language("ko", "Korean", "한국어", "당신", cased=False),
    Language("zh-hans", "Chinese (Simplified)", "中文（简体）", "你", **_UNSPACED),
    Language("zh-hant", "Chinese (Traditional)", "中文（繁體）", "你", **_UNSPACED),
    Language("hi", "Hindi", "हिन्दी", "आप", cased=False),
    Language("ms", "Malay", "Bahasa Melayu", "awak"),
    Language("fil", "Filipino", "Filipino", "ikaw"),
)
BY_CODE = {lang.code: lang for lang in LANGUAGES}
TARGETS = {lang.code: lang.name for lang in LANGUAGES}  # code -> English name
# other spellings of a code people (and older settings) use
_ALIASES = {
    "en-us": "en", "eng": "en", "english": "en",
    "es-es": "es", "es-mx": "es-419", "es-la": "es-419", "spanish": "es-419",
    "pt": "pt-br", "portuguese": "pt-br",
    "zh": "zh-hans", "zh-cn": "zh-hans", "zh-sg": "zh-hans", "chinese": "zh-hans",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant",
    "jp": "ja", "kr": "ko", "no": "nb", "iw": "he", "tl": "fil",
}  # fmt: skip


def get(code: str) -> Language | None:
    """The language of a code (any case, aliases allowed); None for one not in the table."""
    return BY_CODE.get(normalize(code))


def normalize(text: str) -> str:
    """The code of a language given by code, alias or name (English or its own, any case);
    a language not in the table gets a code made from its name ("Swahili" -> "swahili")."""
    s = (text or "").strip()
    low = s.lower().replace("_", "-")
    if low in BY_CODE:
        return low
    if low in _ALIASES:
        return _ALIASES[low]
    for lang in LANGUAGES:
        if low in (lang.name.lower(), lang.native.lower()):
            return lang.code
    for code, name in SOURCES.items():
        if low == name.lower():
            return code
    # a name without its variant ("Español", "Português"): the first variant in the table
    for lang in LANGUAGES:
        if low in (_base(lang.name), _base(lang.native)):
            return lang.code
    return re.sub(r"[^\w]+", "-", low).strip("-_") or "en"


def _base(name: str) -> str:
    """A language's name without its variant: "español" for "Español (España)", "中文" for
    "中文（简体）"."""
    return re.split(r" ?[(（]", name)[0].strip().lower()


def primary(code: str) -> str:
    """The language without its variant: es-419 -> es, zh-hans -> zh."""
    return normalize(code).split("-")[0]


def match(have: str, want: str) -> int:
    """How well a translation into *have* serves a player who wants *want*: 2 the same
    language, 1 the same language in another variant (es-419 for es), 0 another one."""
    if not have or not want:
        return 0
    have, want = normalize(have), normalize(want)
    if have == want:
        return 2
    return 1 if primary(have) == primary(want) else 0


def target_code(name_or_code: str) -> str:
    """The code of a translation language given by code or name (:func:`normalize`)."""
    return normalize(name_or_code)


def target_name(code: str) -> str:
    """The English name of a translation language (prompts, logs)."""
    lang = get(code)
    return lang.name if lang else code.replace("-", " ").title()


def native_name(code: str) -> str:
    """The language's own name, for lists players pick from."""
    lang = get(code)
    return lang.native if lang else target_name(code)


def source_name(code: str) -> str:
    if code == "zh":
        return "Chinese"
    return SOURCES.get(code, target_name(code))


def ocr_language(code: str) -> str:
    return OCR.get("zh-hans" if code == "zh" else code, "japan")


def cased(code: str) -> bool:
    """Whether the language is written with capital letters."""
    lang = get(code)
    return lang.cased if lang else True


def spaced(code: str) -> bool:
    """Whether words are separated by spaces (where lines may break)."""
    lang = get(code)
    return lang.spaced if lang else True


def rtl(code: str) -> bool:
    lang = get(code)
    return bool(lang and lang.rtl)


def latin(code: str) -> bool:
    """Whether the language is written in the Latin alphabet (its own name is), where names
    keep their spelling; a language not in the table, by the letters of its code."""
    lang = get(code)
    name = lang.native if lang else code
    return all(unicodedata.name(c, "").startswith("LATIN") for c in name if c.isalpha())


def you(code: str) -> str:
    """What stands for the player's name in a translation when no name is known."""
    lang = get(code)
    return lang.you if lang else "you"
