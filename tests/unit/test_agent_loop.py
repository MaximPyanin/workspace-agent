from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from workplace_agent.agent.loop import Agent
from workplace_agent.agent.prompts import build_system_prompt
from workplace_agent.llm.agent_response import ActionLogEntry, AgentFinalResponse
from workplace_agent.llm.fake import FakeLLM
from workplace_agent.llm.types import (
    LLMResponse,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
    Usage,
)
from workplace_agent.tools.registry import ToolRegistry, ToolSpec


class EchoParams(BaseModel):
    msg: str


class EchoResult(BaseModel):
    echoed: str


async def _echo_handler(params: EchoParams) -> EchoResult:
    return EchoResult(echoed=params.msg)


async def _failing_handler(_params: EchoParams) -> EchoResult:
    raise RuntimeError("boom")


def _registry_with(name: str, handler: Any = _echo_handler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=name,
            description="Echo input back to the caller.",
            params_model=EchoParams,
            result_model=EchoResult,
            handler=handler,
        )
    )
    return registry


def _create_with_tool_use(
    name: str, args: dict[str, Any], *, tool_use_id: str = "tu_1"
) -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        content=[ToolUseBlock(id=tool_use_id, name=name, input=args)],
        usage=Usage(),
    )


def _create_end_turn(text: str = "all done") -> LLMResponse:
    return LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text=text)],
        usage=Usage(),
    )


def _parse_done(summary: str = "ok") -> LLMResponse:
    parsed = AgentFinalResponse(kind="done", summary=summary)
    return LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text=summary)],
        usage=Usage(),
        parsed=parsed,
    )


def _parse_needs_clarification(question: str, summary: str = "ambiguous") -> LLMResponse:
    parsed = AgentFinalResponse(
        kind="needs_clarification",
        summary=summary,
        clarification_question=question,
    )
    return LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text="?")],
        usage=Usage(),
        parsed=parsed,
    )


def test_build_system_prompt_injects_today_date() -> None:
    fixed = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
    prompt = build_system_prompt(now=fixed)
    assert prompt.startswith("Today's date is 2026-04-26 (Sunday) UTC.")

    assert "needs_clarification" in prompt


async def test_invalid_max_steps_rejected() -> None:
    fake = FakeLLM()
    with pytest.raises(ValueError):
        Agent(llm=fake, registry=ToolRegistry(), max_steps=0)


async def test_single_tool_flow_returns_done() -> None:
    fake = FakeLLM(
        create_responses=[
            _create_with_tool_use("echo", {"msg": "hi"}, tool_use_id="t1"),
            _create_end_turn("finished"),
        ],
        parse_responses=[_parse_done("posted message")],
    )
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=5)

    result = await agent.run("Echo hi")

    assert result.kind == "done"
    assert result.summary == "posted message"
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.tool == "echo"
    assert action.arguments == {"msg": "hi"}
    assert action.is_error is False
    assert "echoed" in action.result_summary
    assert len(fake.create_calls) == 2
    assert len(fake.parse_calls) == 1


async def test_multi_tool_flow_records_ordered_actions() -> None:
    fake = FakeLLM(
        create_responses=[
            _create_with_tool_use("echo", {"msg": "first"}, tool_use_id="t1"),
            _create_with_tool_use("echo", {"msg": "second"}, tool_use_id="t2"),
            _create_with_tool_use("echo", {"msg": "third"}, tool_use_id="t3"),
            _create_end_turn("done"),
        ],
        parse_responses=[_parse_done("ran three")],
    )
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=10)

    result = await agent.run("Echo three things")

    assert result.kind == "done"
    assert [a.arguments for a in result.actions] == [
        {"msg": "first"},
        {"msg": "second"},
        {"msg": "third"},
    ]
    assert all(not a.is_error for a in result.actions)


async def test_tool_error_is_logged_and_reported_in_summary() -> None:
    fake = FakeLLM(
        create_responses=[
            _create_with_tool_use("fail", {"msg": "x"}, tool_use_id="t1"),
            _create_end_turn("recovered after failure"),
        ],
        parse_responses=[_parse_done("the tool failed but I told the user")],
    )
    agent = Agent(llm=fake, registry=_registry_with("fail", _failing_handler), max_steps=5)

    result = await agent.run("Try the failing tool")

    assert result.kind == "done"
    assert len(result.actions) == 1
    assert result.actions[0].is_error is True
    assert "tool failed" in result.summary


async def test_clarification_flow_terminates_without_tool_calls() -> None:
    fake = FakeLLM(
        create_responses=[_create_end_turn("?")],
        parse_responses=[_parse_needs_clarification("Which channel?")],
    )
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=5)

    result = await agent.run("Send an update")

    assert result.kind == "needs_clarification"
    assert result.clarification_question == "Which channel?"
    assert result.actions == []
    assert len(fake.create_calls) == 1
    assert len(fake.parse_calls) == 1


async def test_step_limit_synthesizes_truncated_response() -> None:
    fake = FakeLLM(
        create_responses=[
            _create_with_tool_use("echo", {"msg": f"loop-{i}"}, tool_use_id=f"t{i}")
            for i in range(10)
        ],
    )
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=3)

    result = await agent.run("loop forever")

    assert result.kind == "partial"
    assert "step limit" in result.summary.lower()
    assert len(result.actions) == 3
    assert len(fake.parse_calls) == 0


async def test_finalize_falls_back_to_text_when_parsed_missing() -> None:
    fallback = LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text="fallback summary")],
        usage=Usage(),
        parsed=None,
    )
    fake = FakeLLM(
        create_responses=[_create_end_turn("done")],
        parse_responses=[fallback],
    )
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=2)

    result = await agent.run("trivial")

    assert result.kind == "done"
    assert result.summary == "fallback summary"
    assert result.actions == []


async def test_finalize_overrides_llm_actions_with_harness_recorded_actions() -> None:

    parsed_with_paraphrased_actions = AgentFinalResponse(
        kind="done",
        summary="custom",
        actions=[
            ActionLogEntry(tool="echo", arguments={}, result_summary="paraphrased", is_error=False)
        ],
    )
    parse_response = LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text="custom")],
        usage=Usage(),
        parsed=parsed_with_paraphrased_actions,
    )
    fake = FakeLLM(
        create_responses=[
            _create_with_tool_use("echo", {"msg": "real"}, tool_use_id="t1"),
            _create_end_turn("done"),
        ],
        parse_responses=[parse_response],
    )
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=3)

    result = await agent.run("go")

    assert len(result.actions) == 1
    assert result.actions[0].arguments == {"msg": "real"}
    assert result.actions[0].tool == "echo"


async def test_refusal_stop_reason_returns_needs_clarification() -> None:
    refusal_response = LLMResponse(
        stop_reason="refusal",
        content=[TextBlock(text="I cannot help with that.")],
        usage=Usage(),
    )
    fake = FakeLLM(create_responses=[refusal_response])
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=5)

    result = await agent.run("do something bad")
    assert result.kind == "needs_clarification"
    assert (
        "declined" in result.summary.lower()
        or "refused" in result.summary.lower()
        or "decline" in result.summary.lower()
    )
    assert result.clarification_question is not None


async def test_max_tokens_stop_reason_returns_partial() -> None:
    truncated_response = LLMResponse(
        stop_reason="max_tokens",
        content=[TextBlock(text="truncated")],
        usage=Usage(),
    )
    fake = FakeLLM(create_responses=[truncated_response])
    agent = Agent(llm=fake, registry=_registry_with("echo"), max_steps=5)

    result = await agent.run("hello")
    assert result.kind == "partial"
    assert "token limit" in result.summary


async def test_run_passes_system_prompt_and_tool_definitions() -> None:
    fake = FakeLLM(
        create_responses=[_create_end_turn("ok")],
        parse_responses=[_parse_done("ok")],
    )
    custom_prompt = "custom prompt"
    agent = Agent(
        llm=fake,
        registry=_registry_with("echo"),
        max_steps=2,
        system_prompt=custom_prompt,
    )

    await agent.run("hi")

    assert fake.create_calls[0]["system"] == custom_prompt
    raw_tools = fake.create_calls[0]["tools"]
    assert isinstance(raw_tools, list)
    tools = [t for t in raw_tools if isinstance(t, ToolDefinition)]
    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].strict is False
