import pytest

from workplace_agent.llm.agent_response import ActionLogEntry, AgentFinalResponse
from workplace_agent.llm.types import (
    LLMResponse,
    Message,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
    Usage,
    assistant_blocks,
    user_text,
)


def test_user_text_builds_message_with_text_block() -> None:
    msg = user_text("hello")
    assert msg.role == "user"
    assert msg.content == [TextBlock(text="hello")]


def test_assistant_blocks_builds_assistant_message() -> None:
    blocks = [TextBlock(text="hi"), ToolUseBlock(id="tu_1", name="t", input={"a": 1})]
    msg = assistant_blocks(blocks)
    assert msg.role == "assistant"
    assert len(msg.content) == 2


def test_message_discriminator_round_trips_through_pydantic() -> None:
    msg = Message(
        role="assistant",
        content=[TextBlock(text="ok"), ToolUseBlock(id="tu_1", name="t", input={"x": 1})],
    )
    serialized = msg.model_dump()
    restored = Message.model_validate(serialized)
    assert restored == msg


def test_llm_response_helpers_split_content_blocks() -> None:
    resp = LLMResponse(
        stop_reason="tool_use",
        content=[
            TextBlock(text="thinking"),
            ToolUseBlock(id="tu_1", name="slack_send_message", input={"channel": "#x"}),
        ],
        usage=Usage(input_tokens=12, output_tokens=8),
    )
    assert [b.text for b in resp.text_blocks()] == ["thinking"]
    assert [b.name for b in resp.tool_uses()] == ["slack_send_message"]
    assert resp.joined_text() == "thinking"


def test_tool_definition_serialises_with_strict_true_by_default() -> None:
    td = ToolDefinition(
        name="t",
        description="d",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    dumped = td.model_dump()
    assert dumped["strict"] is True
    assert dumped["name"] == "t"


def test_agent_final_response_done_is_valid_without_question() -> None:
    resp = AgentFinalResponse(kind="done", summary="all set")
    assert resp.kind == "done"
    assert resp.clarification_question is None
    assert resp.actions == []


def test_agent_final_response_clarification_requires_question() -> None:
    with pytest.raises(ValueError):
        AgentFinalResponse(kind="needs_clarification", summary="?")


def test_agent_final_response_with_actions_round_trips() -> None:
    resp = AgentFinalResponse(
        kind="done",
        summary="sent",
        actions=[
            ActionLogEntry(
                tool="slack_send_message",
                arguments={"channel": "#x", "text": "hi"},
                result_summary="ok",
            )
        ],
    )
    again = AgentFinalResponse.model_validate(resp.model_dump())
    assert again == resp
