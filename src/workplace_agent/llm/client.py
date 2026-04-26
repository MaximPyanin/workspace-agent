import asyncio
import random
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, get_args

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)
from anthropic.types import Message as SDKMessage
from pydantic import BaseModel

from workplace_agent.llm.types import (
    ContentBlock,
    LLMResponse,
    Message,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

_KNOWN_STOP_REASONS: frozenset[str] = frozenset(get_args(StopReason))

if TYPE_CHECKING:
    from anthropic.types import MessageParam, ToolUnionParam

T = TypeVar("T", bound=BaseModel)


DEFAULT_MAX_TOKENS = 4096
DEFAULT_BETA_HEADER = "structured-outputs-2025-11-13"


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    async def create(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> LLMResponse: ...

    async def parse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        output_format: type[T],
    ) -> LLMResponse: ...


def _message_to_param(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [b.model_dump() for b in message.content]}


def _tool_to_param(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "strict": tool.strict,
    }


def _convert_sdk_block(block: Any) -> ContentBlock | None:
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return TextBlock(text=block.text)
    if block_type == "tool_use":
        raw_input = block.input
        input_dict = cast(dict[str, Any], raw_input) if isinstance(raw_input, dict) else {}
        return ToolUseBlock(id=block.id, name=block.name, input=input_dict)
    if block_type == "tool_result":
        return ToolResultBlock(
            tool_use_id=block.tool_use_id,
            content=block.content if isinstance(block.content, str) else str(block.content),
            is_error=bool(getattr(block, "is_error", False)),
        )
    return None


def _convert_sdk_response(
    sdk_message: SDKMessage,
    *,
    parsed: BaseModel | None = None,
) -> LLMResponse:
    blocks: list[ContentBlock] = []
    for raw in sdk_message.content:
        converted = _convert_sdk_block(raw)
        if converted is not None:
            blocks.append(converted)
    raw_stop = sdk_message.stop_reason or "end_turn"
    stop_reason: StopReason = cast(
        StopReason, raw_stop if raw_stop in _KNOWN_STOP_REASONS else "end_turn"
    )
    usage = Usage(
        input_tokens=sdk_message.usage.input_tokens,
        output_tokens=sdk_message.usage.output_tokens,
    )
    return LLMResponse(stop_reason=stop_reason, content=blocks, usage=usage, parsed=parsed)


class AnthropicClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        beta_header: str = DEFAULT_BETA_HEADER,
        client: AsyncAnthropic | None = None,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
        backoff_jitter: float = 0.25,
        sleep: Any = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = min(max(0, max_retries), 5)
        self.beta_header = beta_header
        self._client = client or AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)
        self._backoff_base = backoff_base
        self._backoff_factor = backoff_factor
        self._backoff_jitter = backoff_jitter
        self._sleep = sleep or asyncio.sleep

    def _extra_headers(self) -> dict[str, str]:
        return {"anthropic-beta": self.beta_header}

    async def _with_retries(self, func: Any) -> Any:
        attempt = 0
        last_exc: Exception | None = None
        while True:
            try:
                return await func()
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                else:
                    raise LLMError(f"non-retryable API error {exc.status_code}: {exc}") from exc
            attempt += 1
            if attempt > self.max_retries:
                raise LLMError(f"giving up after {attempt} attempts: {last_exc}") from last_exc
            delay = self._backoff_base * (self._backoff_factor ** (attempt - 1))
            jitter = random.uniform(0, self._backoff_jitter)
            await self._sleep(delay + jitter)

    async def create(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        message_params = cast("list[MessageParam]", [_message_to_param(m) for m in messages])
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": message_params,
            "extra_headers": self._extra_headers(),
        }
        if tools:
            kwargs["tools"] = cast("list[ToolUnionParam]", [_tool_to_param(t) for t in tools])

        async def call() -> SDKMessage:
            return await self._client.messages.create(**kwargs)

        sdk_message = await self._with_retries(call)
        return _convert_sdk_response(sdk_message)

    async def parse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        output_format: type[T],
    ) -> LLMResponse:
        message_params = cast("list[MessageParam]", [_message_to_param(m) for m in messages])
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": message_params,
            "output_format": output_format,
            "extra_headers": self._extra_headers(),
        }
        if tools:
            kwargs["tools"] = cast("list[ToolUnionParam]", [_tool_to_param(t) for t in tools])

        async def call() -> Any:
            return await self._client.messages.parse(**kwargs)

        parsed_message = await self._with_retries(call)
        parsed = getattr(parsed_message, "parsed_output", None)
        return _convert_sdk_response(parsed_message, parsed=parsed)
