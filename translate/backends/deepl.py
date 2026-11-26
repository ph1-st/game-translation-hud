"""DeepL through its API, for live translation, with the player's own key (DeepL API Free
or Pro: a Free key ends in ``:fx`` and goes to the Free server).

DeepL is no language model: it gets the lines themselves, not the prompt. What the prompt
does for the models is done with its XML tag handling instead: the placeholders
({player}, {n}, {s}) and the game's names and terms from its glossary are sent inside
``<x>`` tags DeepL leaves as they are (the names already in their translation), and the
recent dialogue goes in ``context``, which DeepL reads but neither translates nor bills."""

from __future__ import annotations

import html
import logging
import re
import threading
from typing import Any

from i18n import N_, _
from translate import languages
from translate.backends.http import WebService, json_body
from translate.base import Context, Item, StopError, TranslationError

log = logging.getLogger("translate")

FREE_API = "https://api-free.deepl.com"
PRO_API = "https://api.deepl.com"
SOURCE_CODES = {"ja": "JA", "ko": "KO", "zh-hans": "ZH", "zh-hant": "ZH"}
# the DeepL code(s) for one of ours, the best first (the plain language when DeepL has no
# variant of it; never Simplified Chinese for Traditional)
TARGET_CODES = {
    "en": ("EN-US", "EN"), "en-gb": ("EN-GB", "EN"), "es-419": ("ES-419", "ES"),
    "pt-br": ("PT-BR", "PT"), "pt-pt": ("PT-PT", "PT"), "zh-hans": ("ZH-HANS", "ZH"),
    "zh-hant": ("ZH-HANT",),
}  # fmt: skip
CONTEXT_LINES = 6  # recent dialogue lines sent as context
MIN_TERM = 2  # glossary terms shorter than this (one kanji) are left to DeepL
_KEEP = "x"  # the tag DeepL is told to leave alone
_PLACEHOLDER_RE = re.compile(r"\{(?:player|n|s)\}")
_KEPT_RE = re.compile(rf"<{_KEEP}>(.*?)</{_KEEP}>", re.DOTALL)


def api_for(key: str) -> str:
    return FREE_API if key.strip().endswith(":fx") else PRO_API


def target_candidates(code: str) -> tuple[str, ...]:
    code = languages.normalize(code)
    return TARGET_CODES.get(code) or (code.upper(),)


class DeepLBackend(WebService):
    """DeepL API (v2 translate). Characters sent are summed in ``characters`` (what DeepL
    bills; the free plan has 500,000 a month)."""

    service = "DeepL"
    no_key = N_(
        "No DeepL API key is set. Create one at deepl.com (Account > API keys; the "
        "free plan translates 500,000 characters a month) and paste it into the "
        "settings."
    )
    key_refused = N_(
        "Your DeepL API key was refused. Check that it was pasted completely "
        "and is still active at deepl.com, or paste a new one into the "
        "settings."
    )
    busy = N_("DeepL is busy right now (too many requests). Wait a minute and try again.")
    offline = N_("Could not reach DeepL (check your internet connection).")

    def __init__(
        self, api_key: str, session: Any = None, timeout: float = 20.0, url: str | None = None
    ) -> None:
        super().__init__(api_key, session, timeout)
        self.url = (url or api_for(self.api_key)).rstrip("/")
        self.name = "DeepL"
        self.characters = 0
        self._targets: set[str] | None = None  # the target codes DeepL offers, once asked
        self._targets_lock = threading.Lock()

    def translate(self, items: list[Item], context: Context) -> list[str]:
        if not items:
            return []
        self._need_key()
        target = self.target_code(context.target)
        terms = _terms(context.glossary)
        body: dict[str, Any] = {
            "text": [_marked(it.text, terms) for it in items],
            "target_lang": target,
            "tag_handling": "xml",
            "ignore_tags": [_KEEP],
            "preserve_formatting": True,
        }
        source = SOURCE_CODES.get(languages.normalize(context.source))
        if source:
            body["source_lang"] = source
        recent = [src for _speaker, src, _tr in context.recent[-CONTEXT_LINES:]]
        if recent:
            body["context"] = "\n".join(recent)
        resp = self._call("post", f"{self.url}/v2/translate", json=body, headers=self._headers())
        data = json_body(resp)
        if resp.status_code >= 400:
            raise self._refusal(resp, data, context.target)
        rows = data.get("translations") if isinstance(data, dict) else None
        if not isinstance(rows, list) or len(rows) != len(items):
            raise TranslationError(_("Unexpected answer from DeepL."))
        with self._lock:
            self.characters += sum(len(t) for t in body["text"])
        return [_unmarked(str(r.get("text") or "")) if isinstance(r, dict) else "" for r in rows]

    def check(self) -> str:
        """One tiny real request (one word); raises TranslationError if it fails."""
        self.translate([Item(text="設定", kind="ui")], Context())
        plan = _("free plan") if self.url == FREE_API else _("Pro")
        return _("Connected ({name})").format(name=f"DeepL, {plan}")

    def target_code(self, target: str) -> str:
        """DeepL's code for the language *target* (ours: "es-419"); a StopError when DeepL
        does not offer it."""
        candidates = target_candidates(target)
        offered = self._offered()
        if offered is None:  # (the list could not be had: the first guess, DeepL decides)
            return candidates[0]
        for c in candidates:
            if c in offered:
                return c
        raise StopError(
            _("DeepL does not translate into {language}. Choose another engine for it.").format(
                language=languages.target_name(target)
            )
        )

    # ---- plumbing ----------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"DeepL-Auth-Key {self.api_key}"}

    def _offered(self) -> set[str] | None:
        with self._targets_lock:
            if self._targets is not None:
                return self._targets
            try:
                resp = self._call("get", f"{self.url}/v3/languages",
                                  params={"resource": "translate_text"},
                                  headers=self._headers())  # fmt: skip
            except TranslationError:
                return None
            data = json_body(resp)
            if resp.status_code in (401, 403):
                raise StopError(_(self.key_refused))
            if resp.status_code >= 400 or not isinstance(data, list):
                log.info("DeepL's language list: HTTP %s", resp.status_code)
                return None
            self._targets = {
                str(d.get("lang") or "").upper()
                for d in data
                if isinstance(d, dict) and d.get("usable_as_target")
            }
            return self._targets

    def _is_busy(self, resp: Any) -> bool:
        return resp.status_code in (429, 529) or resp.status_code >= 500

    def _refusal(self, resp: Any, data: Any, target: str) -> TranslationError:
        status = resp.status_code
        message = ""
        if isinstance(data, dict):  # {"message": ...}, or {"error": {"message": ...}}
            err = data.get("error")
            message = str(
                data.get("message") or (err.get("message") if isinstance(err, dict) else "") or ""
            )
        message = message or f"HTTP {status}"
        if status in (401, 403):
            return StopError(_(self.key_refused))
        if status == 456:
            return StopError(
                _(
                    "Your DeepL quota is used up for this month (or a cost limit "
                    "you set at deepl.com was reached)."
                )
            )
        if status == 400 and "target_lang" in message:
            return StopError(
                _("DeepL does not translate into {language}. Choose another engine for it.").format(
                    language=languages.target_name(target)
                )
            )
        return TranslationError(_("DeepL refused the request: {message}").format(message=message))


def _terms(glossary: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The glossary's terms to keep, the longest first (a name inside a longer one)."""
    pairs = {src: tr for src, tr in glossary if len(src) >= MIN_TERM and tr.strip()}
    return sorted(pairs.items(), key=lambda p: -len(p[0]))


def _marked(text: str, terms: list[tuple[str, str]]) -> str:
    """*text* as DeepL's XML: escaped, its placeholders and glossary terms in <x> tags
    (the terms already translated)."""
    parts: list[tuple[str, bool]] = [(text, False)]  # (text, kept as it is)
    for src, tr in terms:
        out: list[tuple[str, bool]] = []
        for piece, kept in parts:
            if kept or src not in piece:
                out.append((piece, kept))
                continue
            for i, bit in enumerate(piece.split(src)):
                if i:
                    out.append((tr, True))
                if bit:
                    out.append((bit, False))
        parts = out
    xml = []
    for piece, kept in parts:
        if kept:
            xml.append(f"<{_KEEP}>{html.escape(piece, quote=False)}</{_KEEP}>")
            continue
        pos = 0
        for m in _PLACEHOLDER_RE.finditer(piece):
            xml.append(html.escape(piece[pos : m.start()], quote=False))
            xml.append(f"<{_KEEP}>{m.group(0)}</{_KEEP}>")
            pos = m.end()
        xml.append(html.escape(piece[pos:], quote=False))
    return "".join(xml)


def _unmarked(text: str) -> str:
    """DeepL's answer as plain text: the kept parts out of their tags, unescaped."""
    return html.unescape(_KEPT_RE.sub(lambda m: m.group(1), text)).strip()
