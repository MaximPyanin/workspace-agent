from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


class ActionLogEntry(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    is_error: bool = False


class _AgentLLMResponse(BaseModel):
    kind: Literal["done", "needs_clarification"]
    summary: str
    clarification_question: str | None = None
    actions: list[ActionLogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_clarification_consistency(self) -> Self:
        if self.kind == "needs_clarification" and not self.clarification_question:
            raise ValueError("clarification_question is required when kind='needs_clarification'")
        return self


class AgentFinalResponse(_AgentLLMResponse):
    kind: Literal["done", "needs_clarification", "partial"]  # type: ignore[assignment]
