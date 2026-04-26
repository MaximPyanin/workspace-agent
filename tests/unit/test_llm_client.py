from typing import Any, cast

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel

from workplace_agent.llm.agent_response import AgentFinalResponse
from workplace_agent.llm.client import DEFAULT_BETA_HEADER, AnthropicClient, LLMError
from workplace_agent.llm.types import ToolDefinition, ToolUseBlock, user_text


class _SDKUsage:
    def __init__(self, input_tokens: int = 5, output_tokens: int = 7) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _SDKTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _SDKToolUseBlock:
    type = "tool_use"

    def __init__(self, *, id: str, name: str, input: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.input = input


class _SDKMessage:
    def __init__(
        self,
        *,
        content: list[Any],
        stop_reason: str = "end_turn",
        usage: _SDKUsage | None = None,
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or _SDKUsage()


class _SDKParsedMessage(_SDKMessage):
    def __init__(self, *, parsed_output: BaseModel | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parsed_output = parsed_output


class _RecordingMessages:
    def __init__(
        self,
        *,
        create_responses: list[Any] | None = None,
        parse_responses: list[Any] | None = None,
    ) -> None:
        self._create_responses = list(create_responses or [])
        self._parse_responses = list(parse_responses or [])
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        result = self._create_responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        result = self._parse_responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeSDKClient:
    def __init__(self, messages: _RecordingMessages) -> None:
        self.messages = messages


def _make_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limit", response=response, body=None)


def _make_connection_error() -> APIConnectionError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APIConnectionError(request=request)


def _make_5xx_error() -> APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(503, request=request)
    return APIStatusError("server", response=response, body=None)


def _make_4xx_error() -> APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    return APIStatusError("bad", response=response, body=None)


def _make_auth_error() -> AuthenticationError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request)
    return AuthenticationError("auth", response=response, body=None)


def _build_client(
    *,
    create_responses: list[Any] | None = None,
    parse_responses: list[Any] | None = None,
    max_retries: int = 3,
    sleep_calls: list[float] | None = None,
) -> tuple[AnthropicClient, _RecordingMessages]:
    msgs = _RecordingMessages(
        create_responses=create_responses,
        parse_responses=parse_responses,
    )
    client = _FakeSDKClient(msgs)
    sleep_log = sleep_calls if sleep_calls is not None else []

    async def _sleep(seconds: float) -> None:
        sleep_log.append(seconds)

    wrapper = AnthropicClient(
        api_key="x",
        model="claude-sonnet-4-5",
        max_retries=max_retries,
        client=cast(AsyncAnthropic, client),
        backoff_base=0.0,
        backoff_factor=1.0,
        backoff_jitter=0.0,
        sleep=_sleep,
    )
    return wrapper, msgs


async def test_create_converts_sdk_response_to_llm_response() -> None:
    sdk_response = _SDKMessage(
        content=[
            _SDKTextBlock(text="thinking out loud"),
            _SDKToolUseBlock(id="tu_1", name="slack_send_message", input={"channel": "#x"}),
        ],
        stop_reason="tool_use",
        usage=_SDKUsage(input_tokens=10, output_tokens=20),
    )
    client, msgs = _build_client(create_responses=[sdk_response])

    resp = await client.create(
        system="sys",
        messages=[user_text("hi")],
        tools=[
            ToolDefinition(
                name="slack_send_message",
                description="send",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        ],
    )

    assert resp.stop_reason == "tool_use"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 20
    assert len(resp.text_blocks()) == 1
    assert resp.text_blocks()[0].text == "thinking out loud"
    tool_uses = resp.tool_uses()
    assert tool_uses[0].name == "slack_send_message"
    assert tool_uses[0].input == {"channel": "#x"}

    call = msgs.create_calls[0]
    assert call["model"] == "claude-sonnet-4-5"
    assert call["extra_headers"] == {"anthropic-beta": DEFAULT_BETA_HEADER}
    assert call["tools"][0]["strict"] is True
    assert call["tools"][0]["name"] == "slack_send_message"
    assert call["system"] == "sys"
    assert call["messages"][0]["role"] == "user"


async def test_create_retries_on_rate_limit_then_succeeds() -> None:
    sdk_response = _SDKMessage(content=[_SDKTextBlock(text="ok")])
    sleep_log: list[float] = []
    client, msgs = _build_client(
        create_responses=[_make_rate_limit_error(), sdk_response],
        max_retries=3,
        sleep_calls=sleep_log,
    )

    resp = await client.create(system="s", messages=[user_text("hi")], tools=[])

    assert resp.stop_reason == "end_turn"
    assert len(msgs.create_calls) == 2
    assert len(sleep_log) == 1


async def test_create_gives_up_after_max_retries() -> None:
    client, msgs = _build_client(
        create_responses=[
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_rate_limit_error(),
        ],
        max_retries=2,
    )
    with pytest.raises(LLMError) as excinfo:
        await client.create(system="s", messages=[user_text("hi")], tools=[])
    assert "giving up" in str(excinfo.value)
    assert len(msgs.create_calls) == 3


async def test_create_does_not_retry_on_4xx() -> None:
    client, msgs = _build_client(
        create_responses=[_make_4xx_error()],
        max_retries=5,
    )
    with pytest.raises(LLMError) as excinfo:
        await client.create(system="s", messages=[user_text("hi")], tools=[])
    assert "non-retryable" in str(excinfo.value)
    assert len(msgs.create_calls) == 1


async def test_create_serialises_tool_use_messages_back_to_sdk() -> None:
    sdk_response = _SDKMessage(content=[_SDKTextBlock(text="ok")])
    client, msgs = _build_client(create_responses=[sdk_response])

    from workplace_agent.llm.types import (
        ToolResultBlock,
        assistant_blocks,
        user_tool_results,
    )

    history = [
        user_text("please do X"),
        assistant_blocks([ToolUseBlock(id="tu_1", name="t", input={"a": 1})]),
        user_tool_results([ToolResultBlock(tool_use_id="tu_1", content="done")]),
    ]
    await client.create(system="s", messages=history, tools=[])
    sent = msgs.create_calls[0]["messages"]
    assert sent[0]["content"][0]["type"] == "text"
    assert sent[1]["content"][0]["type"] == "tool_use"
    assert sent[1]["content"][0]["input"] == {"a": 1}
    assert sent[2]["content"][0]["type"] == "tool_result"
    assert sent[2]["content"][0]["tool_use_id"] == "tu_1"


async def test_create_omits_tools_kwarg_when_tools_list_is_empty() -> None:
    sdk_response = _SDKMessage(content=[_SDKTextBlock(text="ok")])
    client, msgs = _build_client(create_responses=[sdk_response])

    await client.create(system="s", messages=[user_text("hi")], tools=[])

    assert "tools" not in msgs.create_calls[0]


async def test_parse_populates_parsed_field_from_parsed_output() -> None:
    final = AgentFinalResponse(kind="done", summary="all set")
    sdk_parsed = _SDKParsedMessage(
        content=[_SDKTextBlock(text="all set")],
        parsed_output=final,
    )
    client, msgs = _build_client(parse_responses=[sdk_parsed])

    resp = await client.parse(
        system="s",
        messages=[user_text("hi")],
        tools=[],
        output_format=AgentFinalResponse,
    )

    assert isinstance(resp.parsed, AgentFinalResponse)
    assert resp.parsed.kind == "done"
    assert resp.stop_reason == "end_turn"

    parse_call = msgs.parse_calls[0]
    assert parse_call["output_format"] is AgentFinalResponse
    assert parse_call["extra_headers"] == {"anthropic-beta": DEFAULT_BETA_HEADER}


async def test_parse_returns_none_parsed_when_sdk_has_no_parsed_output() -> None:
    sdk_parsed = _SDKParsedMessage(
        content=[_SDKTextBlock(text="no struct")],
        parsed_output=None,
    )
    client, _ = _build_client(parse_responses=[sdk_parsed])
    resp = await client.parse(
        system="s",
        messages=[user_text("hi")],
        tools=[],
        output_format=AgentFinalResponse,
    )
    assert resp.parsed is None


async def test_parse_retries_on_rate_limit_then_succeeds() -> None:
    final = AgentFinalResponse(kind="done", summary="ok")
    sdk_parsed = _SDKParsedMessage(content=[_SDKTextBlock(text="ok")], parsed_output=final)
    client, msgs = _build_client(
        parse_responses=[_make_rate_limit_error(), sdk_parsed],
        max_retries=2,
    )
    resp = await client.parse(
        system="s",
        messages=[user_text("p")],
        tools=[],
        output_format=AgentFinalResponse,
    )
    assert isinstance(resp.parsed, AgentFinalResponse)
    assert len(msgs.parse_calls) == 2
