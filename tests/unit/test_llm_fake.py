import pytest

from workplace_agent.llm.agent_response import AgentFinalResponse
from workplace_agent.llm.fake import FakeLLM, FakeLLMOutputMismatch, FakeLLMScriptExhausted
from workplace_agent.llm.types import (
    LLMResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
    user_text,
)


def _create_response_with_tool_use() -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        content=[ToolUseBlock(id="tu_1", name="t", input={"x": 1})],
        usage=Usage(),
    )


def _parse_response_done() -> LLMResponse:
    parsed = AgentFinalResponse(kind="done", summary="ok")
    return LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text="ok")],
        usage=Usage(),
        parsed=parsed,
    )


async def test_fake_llm_create_yields_scripted_responses_in_order() -> None:
    r1 = _create_response_with_tool_use()
    r2 = LLMResponse(stop_reason="end_turn", content=[TextBlock(text="bye")])
    fake = FakeLLM(create_responses=[r1, r2])

    out1 = await fake.create(system="s", messages=[user_text("p")], tools=[])
    out2 = await fake.create(system="s", messages=[user_text("p")], tools=[])
    assert out1 is r1
    assert out2 is r2
    assert fake.create_remaining == 0
    assert len(fake.create_calls) == 2


async def test_fake_llm_create_raises_when_exhausted() -> None:
    fake = FakeLLM()
    with pytest.raises(FakeLLMScriptExhausted):
        await fake.create(system="", messages=[], tools=[])


async def test_fake_llm_parse_returns_response_with_parsed_field() -> None:
    fake = FakeLLM(parse_responses=[_parse_response_done()])
    out = await fake.parse(
        system="s",
        messages=[user_text("p")],
        tools=[],
        output_format=AgentFinalResponse,
    )
    assert isinstance(out.parsed, AgentFinalResponse)
    assert out.parsed.kind == "done"
    assert fake.parse_remaining == 0


async def test_fake_llm_parse_rejects_mismatched_output_format() -> None:
    class OtherModel(AgentFinalResponse):
        pass

    parsed = AgentFinalResponse(kind="done", summary="ok")
    fake = FakeLLM(
        parse_responses=[
            LLMResponse(
                stop_reason="end_turn",
                content=[TextBlock(text="ok")],
                parsed=parsed,
            )
        ]
    )

    with pytest.raises(FakeLLMOutputMismatch):
        await fake.parse(system="", messages=[], tools=[], output_format=OtherModel)


async def test_fake_llm_queue_methods_append_responses() -> None:
    fake = FakeLLM()
    fake.queue_create(_create_response_with_tool_use())
    fake.queue_parse(_parse_response_done())
    assert fake.create_remaining == 1
    assert fake.parse_remaining == 1


async def test_fake_llm_records_call_arguments() -> None:
    fake = FakeLLM(create_responses=[_create_response_with_tool_use()])
    await fake.create(
        system="sys",
        messages=[user_text("hi")],
        tools=[],
    )
    call = fake.create_calls[0]
    assert call["system"] == "sys"
    assert isinstance(call["messages"], list)
