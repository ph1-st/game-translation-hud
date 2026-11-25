"""PLaMo 3 Translate on this computer (:mod:`translate.local`), for live translation.

PLaMo takes no instructions but a prompt format of its own (PFML3, from Preferred Networks'
model card): a ``context`` (what the text is), the lines just before as pairs of source
and translation, the ``glossary`` entries that occur in the line, then the line, then the
open output, which it completes. The pairs are the last lines of the conversation: the
database's lines the HUD showed and PLaMo's own earlier translations alike
(:class:`~translate.memory.GameMemory`); the glossary is the databases' names and the game
profile's, so a name comes out as the databases have it.

One line is one short request (PLaMo's long-context accuracy is not yet all it should be
in llama.cpp), decoded greedily and stopped at PLaMo's next tag; the lines of a batch go
to the server's slots side by side. The text keeps its placeholders (``{player}``, ``{n}``,
``{s}``), which PLaMo copies like any other symbol.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from i18n import N_, _
from translate import languages
from translate.backends.http import WebService, json_body, service_message
from translate.base import BadAnswerError, Context, Item, StopError, TranslationError
from translate.local import LocalServer
from translate.prompts import fix_placeholders, placeholders_kept

log = logging.getLogger("translate")

STOPS = ["<|plamo:tag|>", "<|plamo:bos|>"]
# the languages PLaMo 3 Translate writes, by its codes (Preferred Networks' model card)
CODES = frozenset({
    "ar", "az", "be", "bg", "bn", "ca", "ceb", "de", "en", "es", "et", "eu", "fa", "fr",
    "gu", "he", "hi", "hr", "hu", "hy", "id", "it", "ja", "jv", "ka", "kk", "km", "ko",
    "lt", "lv", "ml", "mr", "ms", "my", "ne", "nl", "pl", "ru", "si", "sk", "sl", "sr",
    "sw", "ta", "te", "th", "tl", "ur", "uz", "vi", "zh-hans", "zh-hant",
})  # fmt: skip
RECENT = 2  # lines of the conversation before a line of dialogue
NAMES = 8  # glossary names at most before a line
WARM_UP = ("この先は危険だ。気をつけて進もう。原初の海はもうすぐだよ。",
           "装備を強化すると、戦闘で受けるダメージを減らすことができます。",
           "あの日のことは、今でもはっきりと覚えているんだ。",
           "報酬を受け取るには、もう一度クエストを完了してください。")  # (as many as slots)


def plamo_code(code: str) -> str:
    """PLaMo's code for one of the app's languages ("" when PLaMo does not write it)."""
    code = languages.normalize(code)
    if code in CODES:
        return code
    primary = "tl" if code == "fil" else languages.primary(code)
    return primary if primary in CODES else ""


def prompt(text: str, source: str, target: str, about: str = "",
           pairs: list[tuple[str, str]] | None = None,
           glossary: list[tuple[str, str]] | None = None, bos: bool = False) -> str:
    """PLaMo's prompt for *text*: what it is (*about*), the *pairs* (source, translation)
    before it, the *glossary* entries for it, then the line and the open output. *bos*:
    begin with PLaMo's start mark (llama.cpp's server adds it by itself)."""
    tag = "<|plamo:tag|>"
    into = f"{tag}input<|plamo:key|>lang<|plamo:val|>{source}<|plamo:msg|>"
    out = f"{tag}output<|plamo:key|>lang<|plamo:val|>{target}<|plamo:msg|>"
    parts = ["<|plamo:bos|>" if bos else "", f"{tag}dataset<|plamo:msg|>translation"]
    if about:
        parts.append(f"{tag}context<|plamo:msg|>{_plain(about)}")
    parts += [into + _plain(s) + out + _plain(t) for s, t in pairs or []]
    if glossary:
        parts.append(f"{tag}glossary" + "".join(
            f"<|plamo:index|>{_plain(s)}<|plamo:item|>{_plain(t)}" for s, t in glossary))
    return "".join(parts) + into + _plain(text) + out


def _plain(text: str) -> str:
    """Text as PLaMo reads it: its marks are made of < and >, so those become full width."""
    return " ".join(text.split("\n")).replace("<", "＜").replace(">", "＞")


class PlamoBackend(WebService):
    """PLaMo 3 Translate through the local server (started on the first request)."""

    service = "PLaMo"
    busy = N_("The local model is busy: wait a moment.")
    offline = N_("The local model stopped answering.")

    def __init__(self, server: LocalServer, name: str = "PLaMo 3 Translate",
                 timeout: float = 30.0) -> None:
        super().__init__("", None, timeout)
        self.server = server
        self.name = name
        self.retry_delays = (0.5,)
        self._bos: bool | None = None  # does the prompt need PLaMo's start mark?

    def translate(self, items: list[Item], context: Context) -> list[str]:
        if not items:
            return []
        source, target = plamo_code(context.source), plamo_code(context.target)
        if not target or not source:
            missing = context.target if not target else context.source
            raise StopError(_("PLaMo does not translate {language}: choose Claude or ChatGPT "
                              "under Live translation.").format(
                language=languages.native_name(languages.normalize(missing))))
        url = self.server.start()
        if self._bos is None:
            self._bos = not self._adds_bos(url)
        recent = [(s, t) for _sp, s, t in context.recent[-RECENT:]]
        work = [prompt(it.text, source, target, _about(it, context.game),
                       recent if it.kind == "dialogue" else None,
                       _names(it.text, context.glossary), self._bos) for it in items]
        if len(work) == 1:
            answers = [self._complete(url, work[0], items[0].text)]
        else:
            with ThreadPoolExecutor(max_workers=min(len(work), self.server.slots)) as pool:
                answers = list(pool.map(self._complete, [url] * len(work), work,
                                        [it.text for it in items]))
        out = []
        for it, answer in zip(items, answers, strict=True):
            fixed = fix_placeholders(it.text, answer)
            if not placeholders_kept(it.text, fixed):
                log.warning("PLaMo lost a placeholder: %r -> %r", it.text, fixed)
            out.append(fixed)
        return out

    def check(self) -> str:
        """Start the model and translate one word; raises TranslationError if it fails."""
        self.translate([Item(text="設定", kind="ui")], Context())
        where = _("graphics card") if self.server.gpu else _("processor")
        return _("Ready ({name}, on the {where})").format(name=self.name, where=where)

    def warm_up(self) -> None:
        """Load the model and get the graphics card ready before the first line needs it:
        the first long prompt, and the first lines side by side, take seconds each (the
        card's programs for them are built then)."""
        try:
            self.check()
            context = Context(glossary=[("原初の海", "Primordial Sea")],
                              recent=[(None, "ねえ、来て！", "Hey, come here!")] * RECENT)
            lines = [Item(text, "dialogue") for text in WARM_UP]
            self.translate(lines[:1], context)  # one long prompt, then several side by side
            self.translate(lines[: self.server.slots], context)
        except TranslationError as e:
            log.warning("the local model did not warm up: %s", e)

    def _complete(self, url: str, text: str, source: str) -> str:
        body: dict[str, Any] = {
            "prompt": text, "n_predict": min(384, 3 * len(source) + 16), "temperature": 0,
            "top_k": 1, "repeat_penalty": 1.0, "stop": STOPS, "cache_prompt": True,
        }
        resp = self._call("post", f"{url}/completion", json=body)
        data = json_body(resp)
        if resp.status_code >= 400 or not isinstance(data, dict):
            raise TranslationError(_("The local model refused the request: {message}").format(
                message=service_message(data, resp)))
        answer = str(data.get("content") or "").strip()
        if not answer:
            raise BadAnswerError(_("The local model gave no translation."))
        return answer

    def _adds_bos(self, url: str) -> bool:
        """Whether the server begins every prompt with the model's start mark itself."""
        resp = self._call("post", f"{url}/tokenize", json={"content": "", "add_special": True})
        tokens = (json_body(resp) or {}).get("tokens") if resp.status_code < 400 else None
        return bool(tokens)


def _about(item: Item, game: str) -> str:
    """What the text is, for PLaMo's context."""
    what = "dialogue" if item.kind == "dialogue" else "menus and on-screen text"
    speaker = f" Speaker: {item.speaker}." if item.kind == "dialogue" and item.speaker else ""
    return f"{game + ': ' if game else ''}a video game's {what}.{speaker}"


def _names(text: str, glossary: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The glossary's names that occur in *text*, longest first (at most NAMES)."""
    found = sorted({(s, t) for s, t in glossary if len(s) >= 2 and t and s in text},
                   key=lambda p: -len(p[0]))
    return found[:NAMES]
