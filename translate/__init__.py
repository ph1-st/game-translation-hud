"""Live translation of text the database does not know.

* :mod:`translate.languages`: every language the app offers, and how its text is laid out
* :mod:`translate.backends`: PLaMo 3 Translate on this computer, Claude (Anthropic API) and
  ChatGPT (OpenAI API)
* :mod:`translate.local`: installing and running the local model (llama.cpp's server)
* :mod:`translate.prompts`: what the models are told, and how their answers are read
* :mod:`translate.memory`: per-game context (glossary, recent dialogue), persisted as JSON
* :mod:`translate.live`: background translation of single lines for the HUD
"""

from translate.backends import (
    CLAUDE_MODELS,
    DEFAULT_BATCH_MODEL,
    DEFAULT_LIVE_MODEL,
    DEFAULT_OPENAI_MODEL,
    Backend,
    BadAnswerError,
    ClaudeBackend,
    Context,
    Item,
    OpenAIBackend,
    PlamoBackend,
    StopError,
    TranslationError,
    claude_model_name,
)
from translate.live import LiveTranslator
from translate.memory import GameMemory
from translate.prompts import build_system_prompt, build_user_message, parse_translations

__all__ = [
    "CLAUDE_MODELS",
    "DEFAULT_BATCH_MODEL",
    "DEFAULT_LIVE_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "BadAnswerError",
    "Backend",
    "ClaudeBackend",
    "Context",
    "GameMemory",
    "Item",
    "LiveTranslator",
    "OpenAIBackend",
    "PlamoBackend",
    "StopError",
    "TranslationError",
    "build_system_prompt",
    "build_user_message",
    "claude_model_name",
    "parse_translations",
]
