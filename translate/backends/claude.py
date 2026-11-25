"""Claude through the Anthropic API, for live translation."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from i18n import N_, _
from translate.backends.llm import MAX_TOKENS, answer_budget, ask_twice, time_budget
from translate.base import BadAnswerError, Context, Item, StopError, TranslationError
from translate.prompts import build_system_prompt, build_user_message

log = logging.getLogger("translate")

CLAUDE_MODELS: dict[str, str] = {
    "claude-haiku-5-5": N_("Claude Haiku 5.5 — fastest, cheapest"),
    "claude-sonnet-5-5": N_("Claude Sonnet 5.5 — better quality, costs more"),
    "claude-opus-5-5": N_("Claude Opus 5.5 — best quality, most expensive"),
}
DEFAULT_LIVE_MODEL = "claude-haiku-5-5"
DEFAULT_BATCH_MODEL = "claude-sonnet-5-5"

BUSY_WAIT_MAX_S = 30.0  # the longest wait before asking a busy API again
# Claude's structured output: the answer is guaranteed to be this JSON object.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
    "required": ["translations"],
    "additionalProperties": False,
}

NO_KEY = N_(
    "No Claude API key is set. Create one at console.anthropic.com (Settings > API keys) and "
    "paste it into the settings."
)
KEY_REFUSED = N_(
    "Your Claude API key was refused. Check that it was pasted completely and is still active "
    "at console.anthropic.com, or paste a new one into the settings."
)
BUSY = N_(
    "The Claude API is busy right now (too many requests, or overloaded). Wait a minute and "
    "try again."
)
OFFLINE = N_("Could not reach the Claude API (check your internet connection).")


def claude_model_name(model: str) -> str:
    """The name to show for a model id ("Claude Haiku 5.5"); unknown ids as they are."""
    label = CLAUDE_MODELS.get(model)
    return label.split(" — ")[0] if label else model


class ClaudeBackend:
    """Claude through the Anthropic API.

    The rules and the glossary go in a cached system block; the answer is constrained to the
    JSON schema of :data:`ANSWER_SCHEMA`. ``effort`` sets how much the model may think
    ("low" keeps the HUD's latency down; None leaves the model's default). ``timeout`` is
    the wait for one answer; a big batch gets longer (``llm.time_budget``), since nothing
    arrives before the whole answer is written. The client's own retries (``max_retries=2``)
    are followed by up to three tries here when the API is busy (429, 529, 5xx),
    ``retry_delays`` apart. Token use is summed in ``input_tokens``, ``output_tokens``,
    ``cache_read_tokens`` and ``cache_write_tokens``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LIVE_MODEL,
        client: Any = None,
        timeout: float = 30.0,
        effort: str | None = "low",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.effort = effort
        self.name = claude_model_name(model)
        self.retry_delays: tuple[float, ...] = (2.0, 6.0)
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self._client = client
        self._lock = threading.Lock()
        self._sleep: Callable[[float], None] = time.sleep

    def translate(self, items: list[Item], context: Context) -> list[str]:
        if not items:
            return []
        system = [{"type": "text", "text": build_system_prompt(context),
                   "cache_control": {"type": "ephemeral"}}]
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort
        max_tokens = answer_budget(items)
        timeout = max(self.timeout, time_budget(items))

        def ask(note: str | None) -> str:
            nonlocal max_tokens
            resp = self._create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": build_user_message(items, context, note=note)}
                ],
                output_config=output_config,
                timeout=timeout,
            )
            self._count(resp)
            stop = getattr(resp, "stop_reason", None)
            if stop == "refusal":
                raise BadAnswerError(_("Claude declined to translate this text."))
            if stop == "max_tokens":  # cut off: the retry gets all the room there is
                max_tokens = MAX_TOKENS
            blocks = getattr(resp, "content", None) or []
            return "".join(b.text for b in blocks if getattr(b, "type", None) == "text")

        return ask_twice(items, ask)

    def check(self) -> str:
        """One tiny real request (one word); raises TranslationError if it fails."""
        self.translate([Item(text="設定", kind="ui")], Context())
        return _("Connected ({name})").format(name=self.name)

    # ---- plumbing ----------------------------------------------------------------------
    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                import anthropic

                # An empty key lets the SDK look for ANTHROPIC_API_KEY or a login profile.
                self._client = anthropic.Anthropic(
                    api_key=self.api_key or None, max_retries=2, timeout=self.timeout
                )
            return self._client

    def _create(self, **request: Any) -> Any:
        """``messages.create`` with the SDK's errors turned into TranslationErrors."""
        import anthropic

        client = self._get_client()
        busy = (anthropic.RateLimitError, anthropic.OverloadedError, anthropic.InternalServerError)
        tries = len(self.retry_delays) + 1
        for attempt in range(tries):
            try:
                return client.messages.create(**request)
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                raise StopError(_(KEY_REFUSED)) from exc
            except anthropic.NotFoundError as exc:
                raise StopError(_(
                    "The Claude model {model} is not available to your API key. Choose "
                    "another model in the settings."
                ).format(model=self.model)) from exc
            except anthropic.APIStatusError as exc:
                if not (isinstance(exc, busy) or exc.status_code >= 500):
                    message = _api_message(exc)
                    # no credit left: every request would be refused the same way
                    error = StopError if "credit balance" in message.lower() else TranslationError
                    raise error(_("The Claude API refused the request: {message}").format(
                        message=message)) from exc
                if attempt == tries - 1:
                    raise TranslationError(_(BUSY)) from exc
                wait = self._busy_wait(exc, attempt)
                log.info("Claude API busy (%s): trying again in %.1f s", exc.status_code, wait)
                self._sleep(wait)
            except anthropic.APITimeoutError as exc:
                seconds = request.get("timeout", self.timeout)
                raise TranslationError(_("{offline} No answer came within {seconds} s.").format(
                    offline=_(OFFLINE), seconds=f"{seconds:.0f}")) from exc
            except anthropic.APIConnectionError as exc:
                raise TranslationError(_(OFFLINE)) from exc
            except anthropic.APIError as exc:
                raise TranslationError(_("Unexpected answer from the Claude API: {error}").format(
                    error=exc)) from exc
            except TypeError as exc:  # the SDK found no key at all
                if "authentication" not in str(exc).lower():
                    raise
                raise StopError(_(NO_KEY)) from exc
        raise AssertionError("unreachable")

    def _busy_wait(self, exc: Any, attempt: int) -> float:
        delay = self.retry_delays[attempt]
        try:
            delay = max(delay, float(exc.response.headers.get("retry-after", 0)))
        except (AttributeError, TypeError, ValueError):
            pass
        return min(delay, BUSY_WAIT_MAX_S) * random.uniform(1.0, 1.25)

    def _count(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        with self._lock:
            self.input_tokens += _tokens(usage, "input_tokens")
            self.output_tokens += _tokens(usage, "output_tokens")
            self.cache_read_tokens += _tokens(usage, "cache_read_input_tokens")
            self.cache_write_tokens += _tokens(usage, "cache_creation_input_tokens")


def _tokens(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0)
    return value if isinstance(value, int) else 0


def _api_message(exc: Any) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return str(getattr(exc, "message", "") or exc)
