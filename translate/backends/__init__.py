"""Translation backends for live translation: PLaMo 3 Translate on this computer, the
player's own key for Claude (Anthropic API), ChatGPT (OpenAI API) or DeepL, and any model
behind an OpenAI-compatible server the player names (:mod:`translate.backends.compatible`).

Claude and ChatGPT get the requests built by :mod:`translate.prompts` and their answers
are read with :func:`~translate.prompts.parse_translations`; an unusable answer (not JSON,
the wrong number of lines, a lost placeholder) is asked for once more with a note saying
what was wrong (:func:`~translate.backends.llm.ask_twice`). PLaMo takes its own prompt
format, one line at a time (:mod:`translate.backends.plamo`). Every failure comes out as a
:class:`TranslationError` whose message tells the player what to do;
:class:`BadAnswerError` marks the ones that depend on the text that was sent.
"""

from translate.backends.claude import (
    ANSWER_SCHEMA,
    CLAUDE_MODELS,
    DEFAULT_BATCH_MODEL,
    DEFAULT_LIVE_MODEL,
    ClaudeBackend,
    claude_model_name,
)
from translate.backends.compatible import SERVERS, CompatibleBackend
from translate.backends.deepl import DeepLBackend
from translate.backends.llm import ask_twice
from translate.backends.openai import DEFAULT_OPENAI_MODEL, OpenAIBackend
from translate.backends.plamo import PlamoBackend
from translate.base import Backend, BadAnswerError, Context, Item, StopError, TranslationError

__all__ = [
    "ANSWER_SCHEMA",
    "CLAUDE_MODELS",
    "DEFAULT_BATCH_MODEL",
    "DEFAULT_LIVE_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "SERVERS",
    "BadAnswerError",
    "Backend",
    "ClaudeBackend",
    "CompatibleBackend",
    "Context",
    "DeepLBackend",
    "Item",
    "OpenAIBackend",
    "PlamoBackend",
    "StopError",
    "TranslationError",
    "ask_twice",
    "claude_model_name",
]
