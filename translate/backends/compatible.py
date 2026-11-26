"""Any model behind an OpenAI-compatible server, for live translation: a model the player
runs themselves (Ollama, LM Studio, llama.cpp's server, KoboldCpp, vLLM) or a service with
such an API (OpenRouter, Google Gemini, DeepSeek, Groq, Mistral...). Nothing is installed
for it: the player gives the server's address, the model's name and a key if the server
needs one.

The request is Chat Completions, which all of them speak, with the same prompt as Claude
and ChatGPT. The answer is asked for as JSON in the strictest form the server takes (a
JSON schema, else any JSON object, else the prompt's word alone); what a server refused is
remembered. A reasoning model's thinking (``<think>…</think>``) is left out of the answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from i18n import N_, _
from translate.backends.claude import ANSWER_SCHEMA
from translate.backends.http import WebService, json_body, service_message
from translate.backends.llm import MAX_TOKENS, answer_budget, ask_twice, time_budget
from translate.base import BadAnswerError, Context, Item, StopError, TranslationError
from translate.prompts import build_system_prompt, build_user_message

log = logging.getLogger("translate")

# Addresses to start from (the panel's list; any other works the same way)
SERVERS: dict[str, str] = {
    "Ollama": "http://localhost:11434/v1",
    "LM Studio": "http://localhost:1234/v1",
    "llama.cpp": "http://localhost:8080/v1",
    "KoboldCpp": "http://localhost:5001/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Google Gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Groq": "https://api.groq.com/openai/v1",
    "Mistral": "https://api.mistral.ai/v1",
}
FORMATS = ("json_schema", "json_object", None)  # the answer's format, the strictest first
FIRST_ANSWER_S = 120.0  # a server on this computer may load the model first
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)


class CompatibleBackend(WebService):
    """A model behind an OpenAI-compatible server at *url* (".../v1"). *api_key* may be
    empty (most servers on this computer take none). Token use is summed in
    ``input_tokens`` and ``output_tokens`` when the server says."""

    service = "the model's server"
    no_key = N_("The server asks for an API key: paste it into the settings.")
    key_refused = N_(
        "The server refused the API key. Check that it was pasted completely, "
        "or paste a new one into the settings."
    )
    busy = N_(
        "The model's server is busy right now (too many requests). Wait a minute and try again."
    )
    offline = N_("Could not reach the model's server. Check its address, and that it is running.")

    def __init__(
        self, url: str, model: str, api_key: str = "", session: Any = None, timeout: float = 60.0
    ) -> None:
        super().__init__(api_key, session, timeout)
        self.url = normalize_url(url)
        self.model = model.strip()
        self.name = self.model or _("the model")
        self.format_level = 0  # index into FORMATS: what the server takes
        self.input_tokens = 0
        self.output_tokens = 0
        self._answered = False  # the first answer may wait for the model to load

    def translate(self, items: list[Item], context: Context) -> list[str]:
        if not items:
            return []
        self._ready()
        system = build_system_prompt(context)
        budget = answer_budget(items)
        timeout = max(self.timeout, time_budget(items))
        if not self._answered:
            timeout = max(timeout, FIRST_ANSWER_S)

        def ask(note: str | None) -> str:
            nonlocal budget
            data = self._complete({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": build_user_message(items, context, note=note)},
                ],
                "max_tokens": budget,
                "temperature": 0.2,
                "stream": False,
            }, timeout)  # fmt: skip
            choice = (data.get("choices") or [{}])[0]
            if not isinstance(choice, dict):
                raise BadAnswerError(_("Unexpected answer from the model's server."))
            if choice.get("finish_reason") == "length":
                budget = MAX_TOKENS  # cut off: the retry gets all the room there is
            message = choice.get("message") or {}
            if message.get("refusal"):
                raise BadAnswerError(_("The model declined to translate this text."))
            return answer_text(message.get("content"))

        out = ask_twice(items, ask)
        self._answered = True
        return out

    def check(self) -> str:
        """One tiny real request (one word); raises TranslationError if it fails."""
        self.translate([Item(text="設定", kind="ui")], Context())
        return _("Connected ({name})").format(name=self.name)

    def models(self) -> list[str]:
        """The models the server offers (GET /models), by name."""
        self._ready(model=False)
        resp = self._call("get", f"{self.url}/models", headers=self._headers())
        data = json_body(resp)
        if resp.status_code >= 400:
            raise self._refusal(resp, data)
        rows = data.get("data") if isinstance(data, dict) else None
        names = [
            r["id"] for r in rows or [] if isinstance(r, dict) and isinstance(r.get("id"), str)
        ]
        return sorted(dict.fromkeys(n.removeprefix("models/") for n in names), key=str.lower)

    # ---- plumbing ----------------------------------------------------------------------
    def _ready(self, model: bool = True) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise StopError(
                _(
                    "Type the address of the model's server (for example "
                    "http://localhost:11434/v1 for Ollama)."
                )
            )
        if model and not self.model:
            raise StopError(
                _("Type the name of the model to use (Test lists what the server has).")
            )
        if self.api_key and (not self.api_key.isascii() or any(c.isspace() for c in self.api_key)):
            raise StopError(_(self.key_refused))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _complete(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        """POST /chat/completions in the strictest answer format the server takes."""
        while True:
            fmt = FORMATS[self.format_level]
            if fmt == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "translations",
                        "strict": True,
                        "schema": ANSWER_SCHEMA,
                    },
                }
            elif fmt == "json_object":
                body["response_format"] = {"type": "json_object"}
            else:
                body.pop("response_format", None)
            resp = self._call("post", f"{self.url}/chat/completions", timeout=timeout,
                              json=body, headers=self._headers())  # fmt: skip
            data = json_body(resp)
            if (
                resp.status_code in (400, 422)
                and fmt is not None
                and self.format_level < len(FORMATS) - 1
            ):
                log.info(
                    "%s takes no %s answer format (%s): asking without it",
                    self.url,
                    fmt,
                    service_message(data, resp)[:200],
                )
                self.format_level += 1
                continue
            if resp.status_code >= 400 or not isinstance(data, dict):
                raise self._refusal(resp, data)
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            with self._lock:
                self.input_tokens += int(usage.get("prompt_tokens") or 0)
                self.output_tokens += int(usage.get("completion_tokens") or 0)
            return data

    def _refusal(self, resp: Any, data: Any) -> TranslationError:
        status, message = resp.status_code, service_message(data, resp)
        if status in (401, 403):
            return StopError(_(self.no_key) if not self.api_key else _(self.key_refused))
        if status == 404:
            return StopError(
                _(
                    "The server has no model {model}, or {url} is not its address "
                    "(it usually ends in /v1)."
                ).format(model=self.model, url=self.url)
            )
        if status == 402:  # no credit left
            return StopError(
                _("The model's server refused the request: {message}").format(message=message)
            )
        return TranslationError(
            _("The model's server refused the request: {message}").format(message=message)
        )


def normalize_url(url: str) -> str:
    """The server's address as the requests need it: no trailing slash, and without
    "/chat/completions" when the whole endpoint was pasted."""
    url = (url or "").strip().rstrip("/")
    for tail in ("/chat/completions", "/completions", "/models"):
        if url.endswith(tail):
            url = url[: -len(tail)]
    return url


def answer_text(content: Any) -> str:
    """The text of an answer's content (a string, or a list of parts), thinking left out."""
    if isinstance(content, list):
        content = "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    if not isinstance(content, str):
        return ""
    return _THINK_RE.sub("", content).strip()
