from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

from workplace_agent.llm.types import LLMResponse, Message, ToolDefinition

T = TypeVar("T", bound=BaseModel)


class FakeLLMScriptExhausted(RuntimeError):
    pass


class FakeLLMOutputMismatch(RuntimeError):
    pass


class FakeLLM:
    def __init__(
        self,
        *,
        create_responses: Iterable[LLMResponse] | None = None,
        parse_responses: Iterable[LLMResponse] | None = None,
    ) -> None:
        self._create_queue: list[LLMResponse] = list(create_responses or [])
        self._parse_queue: list[LLMResponse] = list(parse_responses or [])
        self.create_calls: list[dict[str, object]] = []
        self.parse_calls: list[dict[str, object]] = []

    def queue_create(self, response: LLMResponse) -> None:
        self._create_queue.append(response)

    def queue_parse(self, response: LLMResponse) -> None:
        self._parse_queue.append(response)

    @property
    def create_remaining(self) -> int:
        return len(self._create_queue)

    @property
    def parse_remaining(self) -> int:
        return len(self._parse_queue)

    async def create(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        self.create_calls.append(
            {"system": system, "messages": list(messages), "tools": list(tools)}
        )
        if not self._create_queue:
            raise FakeLLMScriptExhausted("no scripted create() responses left")
        return self._create_queue.pop(0)

    async def parse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        output_format: type[T],
    ) -> LLMResponse:
        self.parse_calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": list(tools),
                "output_format": output_format,
            }
        )
        if not self._parse_queue:
            raise FakeLLMScriptExhausted("no scripted parse() responses left")
        response = self._parse_queue.pop(0)
        if response.parsed is not None and not isinstance(response.parsed, output_format):
            raise FakeLLMOutputMismatch(
                f"scripted parse response.parsed is {type(response.parsed).__name__}, "
                f"expected {output_format.__name__}"
            )
        return response
