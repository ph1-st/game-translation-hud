"""ChatGPT through the OpenAI API (the Responses API), for live translation.

The player brings their own API key and picks a model from those the key can use
(:meth:`OpenAIBackend.models`): OpenAI renames its models often, so none is fixed here
but the default. The rules and the glossary are the ``instructions`` (OpenAI caches a
repeated prompt start by itself); the answer is constrained to the JSON schema of
:data:`~translate.backends.claude.ANSWER_SCHEMA`; nothing is stored at OpenAI
(``store: false``)."""

from __future__ import annotations

import logging
from typing import Any

from i18n import N_, _
from translate.backends.claude import ANSWER_SCHEMA
from translate.backends.http import WebService, json_body, service_message
from translate.backends.llm import MAX_TOKENS, answer_budget, ask_twice, time_budget
from translate.base import BadAnswerError, Context, Item, StopError, TranslationError
from translate.prompts import build_system_prompt, build_user_message

log = logging.getLogger("translate")

API = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
# model ids that are no text model (the list of a key holds every kind)
_NOT_TEXT = ("audio", "realtime", "transcribe", "tts", "image", "search", "embedding",
             "moderation", "instruct", "computer-use", "dall-e", "whisper", "sora")


class OpenAIBackend(WebService):
    """ChatGPT through the OpenAI API. ``effort`` is the reasoning effort ("low" keeps the
    HUD's latency down); a model that does not reason is asked again without it. Token use
    is summed in ``input_tokens``, ``output_tokens`` and ``cache_read_tokens``."""

    service = "OpenAI"
    no_key = N_("No OpenAI API key is set. Create one at platform.openai.com (API keys) and "
                "paste it into the settings.")
    key_refused = N_("Your OpenAI API key was refused. Check that it was pasted completely "
                     "and is still active at platform.openai.com, or paste a new one into "
                     "the settings.")
    busy = N_("The OpenAI API is busy right now (too many requests, or overloaded). Wait a "
              "minute and try again.")
    offline = N_("Could not reach the OpenAI API (check your internet connection).")

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL, session: Any = None,
                 timeout: float = 30.0, effort: str | None = "low", url: str = API) -> None:
        super().__init__(api_key, session, timeout)
        self.model = model or DEFAULT_OPENAI_MODEL
        self.effort = effort
        self.url = url.rstrip("/")
        self.name = f"ChatGPT ({self.model})"
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    def translate(self, items: list[Item], context: Context) -> list[str]:
        if not items:
            return []
        self._need_key()
        instructions = build_system_prompt(context)
        budget = answer_budget(items)
        timeout = max(self.timeout, time_budget(items))

        def ask(note: str | None) -> str:
            nonlocal budget
            data = self._respond({
                "model": self.model,
                "instructions": instructions,
                "input": build_user_message(items, context, note=note),
                "text": {"format": {"type": "json_schema", "name": "translations",
                                    "strict": True, "schema": ANSWER_SCHEMA}},
                "max_output_tokens": budget,
                "store": False,
            }, timeout)  # fmt: skip
            if (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
                budget = MAX_TOKENS  # cut off: the retry gets all the room there is
            return output_text(data)

        return ask_twice(items, ask)

    def check(self) -> str:
        """One tiny real request (one word); raises TranslationError if it fails."""
        self.translate([Item(text="設定", kind="ui")], Context())
        return _("Connected ({name})").format(name=self.name)

    def models(self) -> list[str]:
        """The text models the key can use, newest first."""
        self._need_key()
        resp = self._call("get", f"{self.url}/models", headers=self._headers())
        data = json_body(resp)
        if resp.status_code >= 400:
            raise self._refusal(resp, data)
        rows = data.get("data") if isinstance(data, dict) else None
        found = [(r.get("created") or 0, r["id"]) for r in rows or []
                 if isinstance(r, dict) and isinstance(r.get("id"), str)]
        return [mid for _, mid in sorted(found, reverse=True)
                if mid.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
                and not any(word in mid for word in _NOT_TEXT)]

    # ---- plumbing ----------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _respond(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        """POST /responses; the answer's JSON, or a TranslationError."""
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        resp = self._call("post", f"{self.url}/responses", timeout=timeout, json=body,
                          headers=self._headers())
        data = json_body(resp)
        if resp.status_code == 400 and self.effort and "reasoning" in service_message(
                data, resp).lower():
            log.info("%s takes no reasoning effort: asking without it", self.model)
            self.effort = None  # (a model that does not reason: never again)
            body.pop("reasoning", None)
            resp = self._call("post", f"{self.url}/responses", timeout=timeout, json=body,
                              headers=self._headers())
            data = json_body(resp)
        if resp.status_code >= 400 or not isinstance(data, dict):
            raise self._refusal(resp, data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        details = usage.get("input_tokens_details") or {}
        with self._lock:
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
            self.cache_read_tokens += int(details.get("cached_tokens") or 0)
        return data

    def _is_busy(self, resp: Any) -> bool:
        if resp.status_code == 429 and "insufficient_quota" in (resp.text or ""):
            return False  # no credit left: asking again changes nothing
        return super()._is_busy(resp)

    def _refusal(self, resp: Any, data: Any) -> TranslationError:
        status, message = resp.status_code, service_message(data, resp)
        if status == 401:
            return StopError(_(self.key_refused))
        if status == 404:
            return StopError(_("The OpenAI model {model} is not available to your API key. "
                               "Choose another model in the settings.").format(model=self.model))
        if status in (403, 429):  # no credit, no access to the model, a country not served
            return StopError(_("The OpenAI API refused the request: {message}").format(
                message=message))
        return TranslationError(_("The OpenAI API refused the request: {message}").format(
            message=message))


def output_text(data: dict[str, Any]) -> str:
    """The text of a Responses API answer (its message parts, reasoning left out)."""
    parts: list[str] = []
    for out in data.get("output") or []:
        if not isinstance(out, dict) or out.get("type") != "message":
            continue
        for part in out.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise BadAnswerError(_("ChatGPT declined to translate this text."))
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)
