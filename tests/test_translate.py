"""translate/: the prompts and how answers are read, the Claude backend (a fake client, and
the real SDK against a local stand-in for the API), ChatGPT and PLaMo (local stand-ins for
the OpenAI API and llama.cpp's server), the per-game memory and the live translator. Nothing
here leaves the machine."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from db.templates import templated_text, templatize
from translate import (
    BadAnswerError,
    ClaudeBackend,
    Context,
    GameMemory,
    Item,
    LiveTranslator,
    StopError,
    TranslationError,
    build_system_prompt,
    build_user_message,
    claude_model_name,
    parse_translations,
)

GLOSSARY = [("テララ", "Terara"), ("ルミィ", "Lumie"), ("原初の海", "Primordial Sea")]


def wait_until(condition, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@contextmanager
def local_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[ThreadingHTTPServer]:
    """An HTTP server on 127.0.0.1, in a thread; its requests are collected in ``requests``."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.requests = []
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class JsonHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def read_json(self) -> object:
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def send_json(self, status: int, body: object, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)


# ---- prompts ---------------------------------------------------------------------------------
def test_system_prompt_holds_the_rules_and_the_glossary():
    ctx = Context(
        game="azur-promilia",
        glossary=[*GLOSSARY, ("テララ", "Duplicate")],
        recent=[("テララ", "行こう！", "Let's go!")],
    )
    text = build_system_prompt(ctx)
    assert "Azur Promilia" in text
    assert "{player}" in text and "{n}" in text
    assert "テララ = Terara" in text and "原初の海 = Primordial Sea" in text
    assert "Duplicate" not in text
    assert "Let's go!" not in text  # the conversation changes: it stays out of the cached part
    assert '{"translations"' in text
    assert build_system_prompt(ctx) == text  # the same bytes every time, so it caches
    only = build_system_prompt(
        ctx, only_for=[Item("ルミィの帽子"), Item("はい", "dialogue", "テララ")]
    )
    assert "ルミィ = Lumie" in only and "テララ = Terara" in only and "原初の海" not in only
    assert "Glossary" not in build_system_prompt(Context())


def test_user_message_numbers_the_items_after_the_conversation():
    ctx = Context(
        glossary=GLOSSARY, recent=[("テララ", "行こう！", "Let's go!"), (None, "…", "...")]
    )
    items = [
        Item("設定"),
        Item("いまなら\n何でも", "dialogue", "テララ"),
        Item("誰？", "dialogue", "謎の声"),
    ]
    text = build_user_message(items, ctx)
    assert "Terara: 行こう！ => Let's go!" in text
    assert "\n… => ...\n" in text
    assert "1. [ui] 設定" in text
    assert "2. [dialogue, Terara] いまなら 何でも" in text  # glossary name, one line
    assert "3. [dialogue, 謎の声] 誰？" in text
    assert text.index("Let's go!") < text.index("1. [ui]")
    assert "Let's go!" not in build_user_message(items, ctx, max_recent=0)
    retry = build_user_message([Item("設定")], ctx, note="It was not JSON.")
    assert "It was not JSON." in retry and "exactly 1 string," in retry


@pytest.mark.parametrize(
    "answer",
    [
        '{"translations": ["Settings", "Back"]}',
        '```json\n{"translations": ["Settings", "Back"]}\n```',
        'Here you go:\n{"translations": ["Settings", "Back"]}\nHope this helps! {player}',
        'Note: {player} kept. {"result": {"translations": ["Settings", "Back"]}}',
        '["Settings", "Back"]',
        '{"translations": {"2": "Back", "1": "Settings"}}',
        '{"1": "Settings", "2": "Back"}',
        '{"translations": [{"id": 1, "en": " Settings "}, "Back"]}',
    ],
)
def test_parse_translations_finds_the_list(answer):
    assert parse_translations(answer, 2) == ["Settings", "Back"]


def test_parse_translations_rejects_unusable_answers():
    assert parse_translations('{"translations": "Settings"}', 1) == ["Settings"]
    with pytest.raises(TranslationError, match="not JSON"):
        parse_translations("Settings, Back", 2)
    with pytest.raises(BadAnswerError, match="3 translations for 2 lines"):
        parse_translations('{"translations": ["a", "b", "c"]}', 2)
    with pytest.raises(BadAnswerError, match='no "translations" list'):
        parse_translations('{"answer": "a", "other": "b"}', 2)
    with pytest.raises(BadAnswerError, match="not text"):
        parse_translations('{"translations": [["a"]]}', 1)


# ---- Claude ----------------------------------------------------------------------------------
class FakeClaude:
    """Stands in for anthropic.Anthropic: ``messages.create`` records the request and
    returns (or raises) the next scripted answer."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **request):
        self.calls.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def reply(
    text: str, stop_reason: str = "end_turn", input_tokens: int = 100, output_tokens: int = 20
):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="x"),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=7,
            cache_creation_input_tokens=None,
        ),
    )


def api_error(cls, status: int, message: str = "error", headers: dict | None = None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls(message, response=response, body={"error": {"type": "x", "message": message}})


def claude(*answers: object) -> tuple[ClaudeBackend, FakeClaude]:
    client = FakeClaude(*answers)
    backend = ClaudeBackend("sk-ant-test", client=client)
    backend.retry_delays = (0.0, 0.0)
    backend._sleep = lambda seconds: None
    return backend, client


def test_claude_request_answer_and_usage():
    backend, client = claude(
        reply('{"translations": ["Settings", "Right now, {player}..."]}', input_tokens=120)
    )
    ctx = Context(
        game="azur-promilia", glossary=GLOSSARY, recent=[("テララ", "行こう！", "Let's go!")]
    )
    items = [Item("設定"), Item("いまなら…{player}", "dialogue", "テララ")]
    assert backend.translate(items, ctx) == ["Settings", "Right now, {player}..."]
    (call,) = client.calls
    assert call["model"] == "claude-haiku-5-5"
    assert call["max_tokens"] >= 2048
    (block,) = call["system"]
    assert block["type"] == "text" and block["cache_control"] == {"type": "ephemeral"}
    assert "テララ = Terara" in block["text"] and "Let's go!" not in block["text"]
    (message,) = call["messages"]
    assert message["role"] == "user"
    assert "2. [dialogue, Terara] いまなら…{player}" in message["content"]
    assert "Terara: 行こう！ => Let's go!" in message["content"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"
    assert (backend.input_tokens, backend.output_tokens) == (120, 20)
    assert (backend.cache_read_tokens, backend.cache_write_tokens) == (7, 0)
    assert backend.name == "Claude Haiku 5.5" and claude_model_name("my-model") == "my-model"
    assert backend.translate([], ctx) == [] and len(client.calls) == 1
    assert 30 <= call["timeout"] < 35  # a short batch waits about the client's timeout

    batch = [Item("あ" * 20, "dialogue") for _ in range(30)]
    backend, client = claude(reply(json.dumps({"translations": ["Ah"] * 30})))
    assert backend.translate(batch, Context()) == ["Ah"] * 30
    assert client.calls[0]["timeout"] > 60  # nothing arrives before the whole answer


def test_claude_asks_once_more_after_an_unusable_answer():
    backend, client = claude(
        reply('{"translations": ["A"]}'),
        reply('```json\n{"translations": ["A", "B"]}\n```'),
    )
    assert backend.translate([Item("あ"), Item("い")], Context()) == ["A", "B"]
    assert len(client.calls) == 2
    retry = client.calls[1]["messages"][0]["content"]
    assert "1 translations for 2 lines" in retry and "exactly 2 strings" in retry
    assert backend.input_tokens == 200  # both requests are counted

    backend, client = claude(reply("Sorry, I can't."), reply('{"translations": []}'))
    with pytest.raises(BadAnswerError):
        backend.translate([Item("あ")], Context())
    assert len(client.calls) == 2


def test_claude_lost_placeholders():
    backend, client = claude(
        reply('{"translations": ["30 seconds left"]}'),
        reply('{"translations": ["{N} seconds left"]}'),  # repaired to {n}
    )
    assert backend.translate([Item("残り{n}秒")], Context()) == ["{n} seconds left"]
    assert "placeholders" in client.calls[1]["messages"][0]["content"]

    # still lost on the retry: the answer is kept (a missing number beats no English)
    backend, client = claude(reply('["30 seconds left"]'), reply("garbage"))
    assert backend.translate([Item("残り{n}秒")], Context()) == ["30 seconds left"]


def test_claude_cut_off_answer_gets_more_room():
    backend, client = claude(
        reply('{"translations": ["Sett', stop_reason="max_tokens"),
        reply('{"translations": ["Settings"]}'),
    )
    assert backend.translate([Item("設定")], Context()) == ["Settings"]
    assert client.calls[1]["max_tokens"] == 16000 > client.calls[0]["max_tokens"]


def test_claude_errors_become_plain_messages():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cases = [
        (api_error(anthropic.AuthenticationError, 401, "invalid x-api-key"), "API key was refused"),
        (api_error(anthropic.PermissionDeniedError, 403), "API key was refused"),
        (anthropic.APIConnectionError(request=request), "Could not reach the Claude API"),
        (anthropic.APITimeoutError(request=request), "Could not reach the Claude API"),
        (api_error(anthropic.NotFoundError, 404, "model: x"), "not available to your API key"),
        (api_error(anthropic.BadRequestError, 400, "Your credit balance is too low"), "credit"),
        (
            TypeError("Could not resolve authentication method. Expected one of api_key"),
            "No Claude",
        ),
    ]
    for error, message in cases:
        backend, client = claude(error)
        with pytest.raises(TranslationError, match=message) as info:
            backend.translate([Item("あ")], Context())
        assert not isinstance(info.value, BadAnswerError)
        assert len(client.calls) == 1

    backend, client = claude(reply("", stop_reason="refusal"))
    with pytest.raises(BadAnswerError, match="declined"):
        backend.translate([Item("あ")], Context())
    assert len(client.calls) == 1  # a refusal is not asked again


def test_claude_busy_api_is_tried_three_times():
    sleeps: list[float] = []
    backend, client = claude(
        api_error(anthropic.RateLimitError, 429, headers={"retry-after": "1.5"}),
        api_error(anthropic.OverloadedError, 529),
        reply('{"translations": ["Yes"]}'),
    )
    backend._sleep = sleeps.append
    assert backend.translate([Item("はい")], Context()) == ["Yes"]
    assert len(client.calls) == 3 and len(sleeps) == 2
    assert sleeps[0] >= 1.5  # the server's retry-after is honoured

    backend, client = claude(*(api_error(anthropic.InternalServerError, 500) for _ in range(3)))
    with pytest.raises(TranslationError, match="busy"):
        backend.translate([Item("はい")], Context())
    assert len(client.calls) == 3


def test_claude_check_and_client():
    backend, client = claude(reply('{"translations": ["Settings"]}'))
    assert backend.check() == "Connected (Claude Haiku 5.5)"
    assert "Translate this item:\n1. [ui] 設定" in client.calls[0]["messages"][0]["content"]
    backend, _ = claude(api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(TranslationError, match="API key was refused"):
        backend.check()

    lazy = ClaudeBackend("sk-ant-test", model="claude-sonnet-5-5", timeout=12.0)
    assert lazy._client is None and lazy.name == "Claude Sonnet 5.5"
    real = lazy._get_client()  # built, nothing sent
    assert isinstance(real, anthropic.Anthropic)
    assert real.max_retries == 2 and real.timeout == 12.0 and lazy._get_client() is real


class MessagesHandler(JsonHandler):
    """A stand-in for POST /v1/messages, answering from ``server.replies``, a list of
    (status, body, headers)."""

    def do_POST(self) -> None:
        self.server.requests.append((self.path, self.headers, self.read_json()))
        self.send_json(*self.server.replies.pop(0))


def message_json(text: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 9,
            "cache_read_input_tokens": 600,
            "cache_creation_input_tokens": 0,
        },
    }


def error_json(kind: str, message: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}


def test_claude_through_the_sdk():
    """The real SDK against a local stand-in for the Messages API: the request it sends,
    and how its answers and errors are read."""
    with local_server(MessagesHandler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        client = anthropic.Anthropic(api_key="sk-ant-test", base_url=url, max_retries=0)
        backend = ClaudeBackend("sk-ant-test", client=client)
        backend.retry_delays = (0.0, 0.0)
        server.replies = [(200, message_json('{"translations": ["Settings", "Terara"]}'), {})]
        items = [Item("設定"), Item("テララ", "dialogue")]
        assert backend.translate(items, Context(glossary=GLOSSARY)) == ["Settings", "Terara"]
        path, headers, body = server.requests[0]
        assert path == "/v1/messages" and headers.get("x-api-key") == "sk-ant-test"
        assert body["model"] == "claude-haiku-5-5" and body["max_tokens"] >= 2048
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "テララ = Terara" in body["system"][0]["text"]
        assert body["messages"][0]["role"] == "user"
        assert body["output_config"]["format"]["schema"]["required"] == ["translations"]
        assert body["output_config"]["effort"] == "low"
        assert (backend.input_tokens, backend.output_tokens, backend.cache_read_tokens) == (
            50,
            9,
            600,
        )

        server.replies = [
            (529, error_json("overloaded_error", "Overloaded"), {}),
            (429, error_json("rate_limit_error", "Slow down"), {"retry-after": "0"}),
            (200, message_json('{"translations": ["Yes"]}'), {}),
        ]
        assert backend.translate([Item("はい")], Context()) == ["Yes"]

        for status, kind, expected in [
            (401, "authentication_error", "API key was refused"),
            (403, "permission_error", "API key was refused"),
            (404, "not_found_error", "not available to your API key"),
            (400, "invalid_request_error", "refused the request: Your credit balance is too low"),
        ]:
            server.replies = [(status, error_json(kind, "Your credit balance is too low"), {})]
            with pytest.raises(TranslationError, match=expected):
                backend.translate([Item("はい")], Context())


# ---- ChatGPT (the OpenAI API) ----------------------------------------------------------------
class OpenAIHandler(JsonHandler):
    """The parts of the OpenAI API the backend uses: /v1/responses and /v1/models."""

    def do_GET(self) -> None:
        self.server.requests.append((self.path, dict(self.headers), None))
        self.send_json(200, {"data": [
            {"id": "gpt-old", "created": 1}, {"id": "gpt-new-mini", "created": 3},
            {"id": "gpt-new-audio", "created": 4}, {"id": "text-embedding-3", "created": 5},
            {"id": "o4-mini", "created": 2}]})

    def do_POST(self) -> None:
        self.server.requests.append((self.path, dict(self.headers), self.read_json()))
        status, body = self.server.replies.pop(0)
        self.send_json(status, body)


def response_json(text: str, reason: str | None = None) -> dict:
    return {"status": "incomplete" if reason else "completed",
            "incomplete_details": {"reason": reason} if reason else None,
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 120, "output_tokens": 30,
                      "input_tokens_details": {"cached_tokens": 100}}}


@pytest.fixture
def openai_api():
    with local_server(OpenAIHandler) as server:
        server.replies = []
        server.url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        yield server


def test_chatgpt_request_answer_and_usage(openai_api):
    from translate import OpenAIBackend

    backend = OpenAIBackend("sk-test", model="gpt-test", url=openai_api.url)
    backend.retry_delays = (0.0, 0.0)
    openai_api.replies = [(200, response_json('{"translations": ["Settings", "Terara"]}'))]
    items = [Item("設定"), Item("テララ", "dialogue", "テララ")]
    assert backend.translate(items, Context(glossary=GLOSSARY)) == ["Settings", "Terara"]
    path, headers, body = openai_api.requests[0]
    assert path == "/v1/responses" and headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-test" and body["store"] is False
    assert "テララ = Terara" in body["instructions"] and "1. [ui] 設定" in body["input"]
    assert body["text"]["format"]["strict"] and body["reasoning"] == {"effort": "low"}
    assert (backend.input_tokens, backend.output_tokens, backend.cache_read_tokens) == (
        120, 30, 100)
    assert backend.name == "ChatGPT (gpt-test)"


def test_chatgpt_asks_a_model_that_does_not_reason_without_the_effort(openai_api):
    from translate import OpenAIBackend

    backend = OpenAIBackend("sk-test", url=openai_api.url)
    openai_api.replies = [
        (400, {"error": {"message": "Unsupported parameter: 'reasoning.effort'."}}),
        (200, response_json('{"translations": ["Yes"]}')),
        (200, response_json('{"translations": ["No"]}')),
    ]
    assert backend.translate([Item("はい")], Context()) == ["Yes"]
    assert "reasoning" not in openai_api.requests[1][2] and backend.effort is None
    assert backend.translate([Item("いいえ")], Context()) == ["No"]
    assert "reasoning" not in openai_api.requests[2][2]  # (never again)


def test_chatgpt_errors_become_plain_messages(openai_api):
    from translate import OpenAIBackend

    backend = OpenAIBackend("sk-test", url=openai_api.url)
    backend.retry_delays = (0.0, 0.0)
    for status, error, expected, stop in [
        (401, {"message": "Incorrect API key"}, "API key was refused", True),
        (404, {"message": "no such model"}, "not available to your API key", True),
        (429, {"message": "You exceeded your quota", "code": "insufficient_quota"},
         "exceeded your quota", True),
        (400, {"message": "Bad input"}, "refused the request: Bad input", False),
    ]:
        openai_api.replies = [(status, {"error": error})]
        with pytest.raises(TranslationError, match=expected) as info:
            backend.translate([Item("はい")], Context())
        assert isinstance(info.value, StopError) == stop
        assert not openai_api.replies  # (asked once: no credit is never "busy")
    openai_api.replies = [(503, {"error": "busy"})] * 3
    with pytest.raises(TranslationError, match="busy"):
        backend.translate([Item("はい")], Context())
    openai_api.replies = [(200, {"output": [{"type": "message", "content": [
        {"type": "refusal", "refusal": "no"}]}]})]
    with pytest.raises(BadAnswerError, match="declined"):
        backend.translate([Item("はい")], Context())
    with pytest.raises(StopError, match="No OpenAI API key"):
        OpenAIBackend("", url=openai_api.url).translate([Item("はい")], Context())


def test_chatgpt_lists_the_text_models_newest_first(openai_api):
    from translate import OpenAIBackend

    assert OpenAIBackend("sk-test", url=openai_api.url).models() == [
        "gpt-new-mini", "o4-mini", "gpt-old"]


# ---- PLaMo (the local model) -----------------------------------------------------------------
class LlamaHandler(JsonHandler):
    """The parts of llama.cpp's server the PLaMo backend uses: /tokenize, /completion."""

    def do_POST(self) -> None:
        body = self.read_json()
        self.server.requests.append((self.path, body))
        if self.path == "/tokenize":
            self.send_json(200, {"tokens": [1]})  # (the server adds the start mark itself)
        else:
            line = body["prompt"].rsplit("<|plamo:msg|>", 2)[-2].split("<|plamo:tag|>")[0]
            self.send_json(200, {"content": " " + self.server.answers.get(line, "?")})


@pytest.fixture
def plamo():
    from translate import PlamoBackend

    with local_server(LlamaHandler) as server:
        server.answers = {}
        url = f"http://127.0.0.1:{server.server_address[1]}"
        local = SimpleNamespace(start=lambda: url, slots=2, gpu=True)
        yield server, PlamoBackend(local, "PLaMo 3 Translate 2B")


def test_plamo_prompt_is_its_own_format():
    from translate.backends.plamo import prompt

    assert prompt("原初の海が<見える>\nよ", "ja", "en", "A game.", [("ねえ", "Hey")],
                  [("原初の海", "Primordial Sea")]) == (
        "<|plamo:tag|>dataset<|plamo:msg|>translation"
        "<|plamo:tag|>context<|plamo:msg|>A game."
        "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>ねえ"
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hey"
        "<|plamo:tag|>glossary<|plamo:index|>原初の海<|plamo:item|>Primordial Sea"
        "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>原初の海が＜見える＞ よ"
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>")


def test_plamo_translates_each_line_with_the_conversation_and_the_names(plamo):
    server, backend = plamo
    server.answers = {"設定": "Settings", "テララ、待って！": "Terara, wait!"}
    recent = [("ルミィ", "ねえ", "Hey"), ("テララ", "なに？", "What?"), ("ルミィ", "来て", "Come")]
    items = [Item("設定"), Item("テララ、待って！", "dialogue", "ルミィ")]
    out = backend.translate(items, Context(game="Star", glossary=GLOSSARY, recent=recent))
    assert out == ["Settings", "Terara, wait!"]
    prompts = {b["prompt"].rsplit("<|plamo:msg|>", 2)[-2].split("<|plamo:tag|>")[0]: b
               for path, b in server.requests if path == "/completion"}
    ui, line = prompts["設定"]["prompt"], prompts["テララ、待って！"]["prompt"]
    assert not ui.startswith("<|plamo:bos|>")  # (the server adds it)
    assert "なに？" not in ui and "テララ<|plamo:item|>" not in ui
    assert "Speaker: ルミィ" in line and "ねえ" not in line  # the last two lines only
    assert "なに？<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>What?" in line
    assert "<|plamo:tag|>glossary<|plamo:index|>テララ<|plamo:item|>Terara" in line
    body = prompts["設定"]
    assert body["temperature"] == 0 and body["stop"] == ["<|plamo:tag|>", "<|plamo:bos|>"]
    assert body["n_predict"] == 3 * 2 + 16


def test_plamo_keeps_placeholders_and_says_what_it_cannot(plamo):
    from translate.backends.plamo import plamo_code

    server, backend = plamo
    server.answers = {"{player}さん、{n}個": "{player}, {n} pieces"}
    assert backend.translate([Item("{player}さん、{n}個")], Context()) == ["{player}, {n} pieces"]
    server.answers = {}
    server.answers["空"] = ""
    with pytest.raises(BadAnswerError, match="no translation"):
        backend.translate([Item("空")], Context())
    with pytest.raises(StopError, match="PLaMo does not translate Português"):
        backend.translate([Item("設定")], Context(target="pt-br"))
    assert [plamo_code(c) for c in ("es-419", "fil", "zh-hant", "en-gb", "pt-br")] == [
        "es", "tl", "zh-hant", "en", ""]


# ---- memory ----------------------------------------------------------------------------------
def test_memory_round_trip(tmp_path):
    path = tmp_path / "cache" / "memory.json"
    memory = GameMemory(path, game="azur-promilia", max_recent=3)
    assert memory.recent == []  # no file yet
    memory.set_glossary([*GLOSSARY, ("", "x"), ("y", " ")])
    memory.remember("行こう！", "Let's go!", "テララ")
    memory.remember("行こう！", "Let's go!", "テララ")  # an exact repeat of the last line
    memory.remember("設定", "Settings", kind="ui")  # not dialogue
    for i in range(4):
        memory.remember(f"台詞{i}", f"Line {i}")
    assert memory.recent == [(None, f"台詞{i}", f"Line {i}") for i in (1, 2, 3)]
    assert memory.save()
    assert [p.name for p in path.parent.iterdir()] == ["memory.json"]  # no temporary file left
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["version"], data["game"], data["target"]) == (2, "azur-promilia", "en")
    assert data["recent"][0] == {"speaker": None, "source": "台詞1", "translation": "Line 1"}
    assert data["glossary"][0] == ["テララ", "Terara"]

    again = GameMemory(path, game="azur-promilia", max_recent=3)
    assert again.recent == memory.recent and again.glossary == GLOSSARY
    ctx = again.context()
    assert (ctx.game, ctx.glossary, ctx.recent) == ("azur-promilia", GLOSSARY, memory.recent)
    ctx.recent.clear()  # a snapshot: the memory is untouched
    assert again.recent
    again.clear()
    assert again.recent == [] and again.glossary == GLOSSARY
    assert GameMemory(None).save() is False
    # a memory kept in another language is not this one's context
    assert GameMemory(path, game="azur-promilia", target="es-419").recent == []


def test_memory_survives_bad_files(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json", encoding="utf-8")
    memory = GameMemory(path)
    assert memory.recent == [] and memory.glossary == []
    path.write_bytes(b"\xff\xfe\x00garbage")
    memory.load()
    path.write_text(json.dumps({"version": 99, "recent": [{"jp": "あ", "en": "A"}]}), "utf-8")
    memory.load()
    assert memory.recent == []
    other_game = {
        "version": 1,
        "game": "other",
        "recent": [{"speaker": None, "jp": "あ", "en": "A"}],
    }
    path.write_text(json.dumps(other_game), "utf-8")
    assert GameMemory(path, game="azur-promilia").recent == []
    assert GameMemory(path).recent == [(None, "あ", "A")]  # no game given: any file
    damaged = {
        "version": 1,
        "recent": [{"jp": "あ"}, "x", {"speaker": "テララ", "jp": "い", "en": "I"}],
        "glossary": [["テララ", "Terara"], ["", "x"], 5, {"jp": "ルミィ", "en": "Lumie"}],
    }
    path.write_text(json.dumps(damaged), "utf-8")
    memory = GameMemory(path)
    assert memory.recent == [("テララ", "い", "I")]
    assert memory.glossary == [("テララ", "Terara"), ("ルミィ", "Lumie")]


# ---- live translation ------------------------------------------------------------------------
class FakeBackend:
    """Translates "x" as "EN:x" and records each batch with its context. ``fail`` makes it
    raise; texts in ``bad`` make the answer unusable; ``gate`` holds a request until set."""

    name = "fake"

    def __init__(self) -> None:
        self.batches: list[list[Item]] = []
        self.contexts: list[Context] = []
        self.fail: Exception | None = None
        self.bad: set[str] = set()
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def translate(self, items: list[Item], context: Context) -> list[str]:
        self.entered.set()
        assert self.gate.wait(5)
        self.batches.append(list(items))
        self.contexts.append(context)
        if self.fail is not None:
            raise self.fail
        if any(it.text in self.bad for it in items):
            raise BadAnswerError(f"The translator returned 1 translations for {len(items)} lines.")
        return [f"EN:{it.text}" for it in items]

    def check(self) -> str:
        return "Connected (fake)"


def test_live_batches_dedupes_and_remembers():
    backend = FakeBackend()
    memory = GameMemory(None)
    results: list[tuple[str, str]] = []
    live = LiveTranslator(backend, memory, lambda item, en: results.append((item.text, en)))
    assert live.request(Item("設定"))  # queued before start: sent once the worker runs
    assert not live.request(Item("設 定"))  # the same text (by key) is queued already
    assert live.request(Item("行こう！", "dialogue", "テララ"))
    assert live.request(Item("戻る"))
    assert not live.request(Item("  "))
    live.start()
    try:
        assert wait_until(lambda: len(results) == 3)
        assert [[it.text for it in b] for b in backend.batches] == [["設定", "行こう！", "戻る"]]
        assert ("行こう！", "EN:行こう！") in results
        assert memory.recent == [("テララ", "行こう！", "EN:行こう！")]  # dialogue only
        assert not live.request(Item("設定"))  # translated already
        stats = live.stats()
        assert stats["backend"] == "fake" and stats["requests"] == 3
        assert (stats["translated"], stats["failed"], stats["pending"]) == (3, 0, 0)
        assert stats["last_error"] is None and isinstance(stats["last_latency_ms"], int)
        assert live.request(Item("うん", "dialogue", "ルミィ"))
        assert wait_until(lambda: len(results) == 4)
        assert backend.contexts[1].recent == [("テララ", "行こう！", "EN:行こう！")]
    finally:
        live.stop()


def test_live_lines_arriving_together_share_a_request():
    backend = FakeBackend()
    live = LiveTranslator(backend, GameMemory(None), lambda item, en: None, max_batch=2)
    live.start()
    try:
        for jp in ("一つ目", "二つ目", "三つ目"):
            assert live.request(Item(jp))
        assert wait_until(lambda: len(backend.batches) == 2)
        # more than one request takes: the newest first (the screen as it is now)
        texts = [[it.text for it in b] for b in backend.batches]
        assert texts == [["二つ目", "三つ目"], ["一つ目"]]
    finally:
        live.stop()


def test_live_failure_is_recorded_and_retried():
    backend = FakeBackend()
    backend.fail = TranslationError("Could not reach the Claude API (check your internet).")
    results: list[str] = []
    live = LiveTranslator(
        backend,
        GameMemory(None),
        lambda item, en: results.append(en),
        retry_after=0.3,
        cooldown=(0.5, 1.0),
    )
    live.start()
    try:
        assert live.request(Item("設定")) and live.request(Item("戻る"))
        assert wait_until(lambda: live.stats()["failed"] == 2)
        stats = live.stats()
        assert stats["last_error"].startswith("Could not reach") and stats["pending"] == 0
        assert stats["waiting"] > 0  # the service failed: nothing is sent for a while
        assert not live.request(Item("設定"))  # failed just now
        backend.fail = None
        time.sleep(0.35)
        assert live.request(Item("設定")) and live.request(Item("戻る"))
        assert wait_until(lambda: sorted(results) == ["EN:戻る", "EN:設定"])
        # the service failed, not the lines: they go together again (and an outage never
        # counts towards giving a line up)
        assert [len(b) for b in backend.batches[-1:]] == [2]
        stats = live.stats()
        assert stats["last_error"] is None and stats["translated"] == 2
    finally:
        live.stop()


def test_live_survives_a_failing_handler():
    backend = FakeBackend()
    seen: list[str] = []

    def handler(item: Item, en: str) -> None:
        seen.append(en)
        raise RuntimeError("a bug in the HUD")

    live = LiveTranslator(backend, GameMemory(None), handler, batch_wait=0.01)
    live.start()
    try:
        assert live.request(Item("一つ目"))
        assert wait_until(lambda: seen == ["EN:一つ目"])
        assert live.request(Item("二つ目"))
        assert wait_until(lambda: seen == ["EN:一つ目", "EN:二つ目"])
    finally:
        live.stop()


def test_live_stop_drops_the_answer_in_flight():
    backend = FakeBackend()
    backend.gate.clear()
    results: list[str] = []
    live = LiveTranslator(backend, GameMemory(None), lambda item, en: results.append(en))
    live.start()
    assert live.request(Item("一つ目"))
    assert backend.entered.wait(5)
    assert live.stats()["pending"] == 1  # in flight
    stopper = threading.Thread(target=live.stop)
    stopper.start()
    time.sleep(0.05)
    backend.gate.set()
    stopper.join(5)
    assert not stopper.is_alive() and results == []
    assert not live.request(Item("二つ目"))  # stopped
    live.start()  # and started again: the dropped line can be asked for again
    try:
        assert live.request(Item("一つ目"))
        assert wait_until(lambda: results == ["EN:一つ目"])
    finally:
        live.stop()


def test_templated_text_keeps_the_template():
    assert templated_text("残り 30秒", templatize("残り 30秒").key) == "残り{n}秒"
    t = templatize("Wrenさん、2回目だね", "Wren")
    assert templated_text("Wrenさん、2回目だね", t.key) == "{player}さん、{n}回目だね"
    t = templatize("Wren、Wren！", "Wren")
    assert templated_text("Wren、Wren！", t.key) == "{player}、{player}!"
    # readable, not folded like the key (ちよつと待つて)
    assert templated_text("ちょっと待って", templatize("ちょっと待って").key) == "ちょっと待って"
    # a reading that cannot be lined up with its key: the key itself
    assert templated_text("全然違う", "{player}の{n}") == "{player}の{n}"


def test_live_stops_for_good_on_a_refused_key():
    from translate import StopError

    backend = FakeBackend()
    backend.fail = StopError("Your Claude API key was refused.")
    live = LiveTranslator(backend, GameMemory(None), lambda item, en: None, batch_wait=0.01)
    live.start()
    try:
        assert live.request(Item("設定"))
        assert wait_until(lambda: live.stats()["halted"] is not None)
        assert not live.request(Item("戻る"))  # nothing more is sent
        assert len(backend.batches) == 1
    finally:
        live.stop()


def test_live_bad_answer_sends_the_lines_one_by_one_and_gives_up_after_three():
    backend = FakeBackend()
    backend.bad = {"壊れた"}
    results: list[str] = []
    live = LiveTranslator(
        backend, GameMemory(None), lambda item, en: results.append(en), retry_after=0.0
    )
    for jp in ("一つ目", "壊れた", "三つ目"):
        assert live.request(Item(jp))
    live.start()
    try:
        assert wait_until(lambda: sorted(results) == ["EN:一つ目", "EN:三つ目"])
        assert [len(b) for b in backend.batches] == [3, 1, 1, 1]  # then one by one, at once
        assert wait_until(lambda: live.stats()["failed"] == 1)
        for _ in range(2):  # read again later: asked for again, up to three times
            assert live.request(Item("壊れた"))
            n = live.stats()["failed"]
            assert wait_until(lambda n=n: live.stats()["failed"] == n + 1)
        assert not live.request(Item("壊れた"))  # failed three times: given up
    finally:
        live.stop()


def test_live_drops_lines_that_waited_too_long():
    backend = FakeBackend()
    backend.gate.clear()
    live = LiveTranslator(backend, GameMemory(None), lambda item, en: None, stale_after=0.1)
    live.start()
    try:
        assert live.request(Item("一つ目"))
        assert backend.entered.wait(5)  # in flight, held
        assert live.request(Item("二つ目"))
        time.sleep(0.2)
        backend.gate.set()
        assert wait_until(lambda: live.stats()["pending"] == 0)
        assert [[it.text for it in b] for b in backend.batches] == [["一つ目"]]
        assert live.request(Item("二つ目"))  # on screen again: asked for again
        assert wait_until(lambda: len(backend.batches) == 2)
    finally:
        live.stop()


def test_live_an_outage_never_gives_a_line_up():
    backend = FakeBackend()
    backend.fail = TranslationError("Could not reach the Claude API.")
    results: list[str] = []
    live = LiveTranslator(
        backend, GameMemory(None), lambda item, en: results.append(en), retry_after=0.0,
        cooldown=(0.01, 0.02), batch_wait=0.01,
    )  # fmt: skip
    live.start()
    try:
        for n in range(1, 5):  # four outages in a row
            assert wait_until(lambda: live.request(Item("設定")))
            assert wait_until(lambda n=n: live.stats()["failed"] == n)
        backend.fail = None
        assert wait_until(lambda: live.request(Item("設定")))
        assert wait_until(lambda: results == ["EN:設定"])
    finally:
        live.stop()
