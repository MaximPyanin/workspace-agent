import itertools
from collections import deque
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from workplace_agent import cli
from workplace_agent.agent.loop import Agent
from workplace_agent.config import Settings
from workplace_agent.llm.agent_response import AgentFinalResponse
from workplace_agent.llm.client import AnthropicClient
from workplace_agent.llm.fake import FakeLLM
from workplace_agent.llm.types import LLMResponse, TextBlock, Usage
from workplace_agent.tools.registry import ToolRegistry, ToolSpec


class _NoopParams(BaseModel):
    pass


class _NoopResult(BaseModel):
    ok: bool = True


async def _noop_handler(_params: _NoopParams) -> _NoopResult:
    return _NoopResult()


def _registry_with_one_tool() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="noop",
            description="Does nothing.",
            params_model=_NoopParams,
            result_model=_NoopResult,
            handler=_noop_handler,
        )
    )
    return registry


def _final_response(summary: str = "ok") -> LLMResponse:
    return LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text=summary)],
        usage=Usage(),
        parsed=AgentFinalResponse(kind="done", summary=summary),
    )


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "anthropic_api_key": "test-key",
        "agent_use_fake_llm": False,
        "log_level": "INFO",
        "agent_max_steps": 4,
    }
    base.update(overrides)
    return Settings(**base)


def test_parse_args_defaults() -> None:
    args = cli.parse_args([])
    assert args.mock_only is False
    assert args.host is None
    assert args.port is None


def test_parse_args_mock_only_flag() -> None:
    args = cli.parse_args(["--mock-only"])
    assert args.mock_only is True


def test_parse_args_host_and_port() -> None:
    args = cli.parse_args(["--host", "0.0.0.0", "--port", "9001"])
    assert args.host == "0.0.0.0"
    assert args.port == 9001


def test_wait_for_health_returns_immediately_on_200() -> None:
    calls: list[str] = []

    def get_fn(url: str) -> httpx.Response:
        calls.append(url)
        return httpx.Response(200, json={"ok": True})

    cli.wait_for_health(
        "http://example",
        timeout=1.0,
        interval=0.0,
        get_fn=get_fn,
        sleep_fn=lambda _s: None,
        monotonic_fn=lambda: 0.0,
    )
    assert calls == ["http://example/health"]


def test_wait_for_health_retries_until_healthy() -> None:
    responses = deque(
        [
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    sleep_calls: list[float] = []

    def get_fn(_url: str) -> httpx.Response:
        return responses.popleft()

    times = iter([0.0, 0.1, 0.2, 0.3])
    cli.wait_for_health(
        "http://example",
        timeout=5.0,
        interval=0.01,
        get_fn=get_fn,
        sleep_fn=sleep_calls.append,
        monotonic_fn=lambda: next(times),
    )
    assert len(sleep_calls) == 2
    assert not responses


def test_wait_for_health_raises_timeout_when_never_healthy() -> None:
    def get_fn(_url: str) -> httpx.Response:
        return httpx.Response(503)

    times = itertools.chain([0.0, 0.5, 1.5], itertools.repeat(99.0))
    with pytest.raises(TimeoutError) as exc:
        cli.wait_for_health(
            "http://example",
            timeout=1.0,
            interval=0.01,
            get_fn=get_fn,
            sleep_fn=lambda _s: None,
            monotonic_fn=lambda: next(times),
        )
    assert "did not become healthy" in str(exc.value)


def test_build_llm_client_returns_fake_when_flag_set() -> None:
    settings = _settings(agent_use_fake_llm=True, anthropic_api_key="")
    client = cli.build_llm_client(settings)
    assert isinstance(client, FakeLLM)


def test_build_llm_client_returns_anthropic_when_real() -> None:
    settings = _settings()
    client = cli.build_llm_client(settings)
    assert isinstance(client, AnthropicClient)
    assert client.model == settings.anthropic_model
    assert client.beta_header == settings.anthropic_beta_header
    assert client.timeout_seconds == settings.llm_timeout_seconds
    assert client.max_retries == settings.llm_max_retries


async def test_repl_async_exits_on_eof() -> None:
    fake = FakeLLM()
    agent = Agent(llm=fake, registry=_registry_with_one_tool(), max_steps=2)

    async def reader() -> str:
        raise EOFError

    from rich.console import Console

    console = Console(record=True, width=80)
    await cli.repl_async(agent, console, read_prompt=reader)

    assert fake.create_calls == []


async def test_repl_async_runs_agent_then_exits_on_quit() -> None:
    fake = FakeLLM()
    fake.queue_create(
        LLMResponse(
            stop_reason="end_turn",
            content=[TextBlock(text="hi")],
            usage=Usage(),
        )
    )
    fake.queue_parse(_final_response("hello back"))
    agent = Agent(llm=fake, registry=_registry_with_one_tool(), max_steps=2)

    prompts = iter(["  hello  ", "exit"])

    async def reader() -> str:
        return next(prompts)

    from rich.console import Console

    console = Console(record=True, width=120)
    await cli.repl_async(agent, console, read_prompt=reader)
    output = console.export_text()
    assert "hello back" in output
    assert len(fake.create_calls) == 1
    assert len(fake.parse_calls) == 1


def test_run_agent_full_lifecycle_with_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(agent_use_fake_llm=True, anthropic_api_key="", mock_api_port=0)
    captured: dict[str, Any] = {}

    async def fake_repl_async(agent: Agent, _console: Any, **_kwargs: Any) -> None:
        captured["agent"] = agent
        captured["base_url"] = str(agent._registry.list_definitions()[0].name)  # type: ignore[attr-defined]

    monkeypatch.setattr(cli, "repl_async", fake_repl_async)
    cli.run_agent(settings)
    assert isinstance(captured["agent"], Agent)
    assert isinstance(captured["agent"]._llm, FakeLLM)  # type: ignore[attr-defined]
