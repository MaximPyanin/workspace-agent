from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"]


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


def user_text(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


def user_tool_results(results: Sequence[ToolResultBlock]) -> Message:
    return Message(role="user", content=list(results))


def assistant_blocks(blocks: Sequence[ContentBlock]) -> Message:
    return Message(role="assistant", content=list(blocks))


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ToolDefinition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True


class LLMResponse(BaseModel):
    stop_reason: StopReason
    content: list[ContentBlock]
    usage: Usage = Field(default_factory=Usage)
    parsed: BaseModel | None = None

    def text_blocks(self) -> list[TextBlock]:
        return [b for b in self.content if isinstance(b, TextBlock)]

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def joined_text(self) -> str:
        return "\n".join(b.text for b in self.text_blocks())
