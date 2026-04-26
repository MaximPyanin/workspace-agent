from workplace_agent.agent.prompts import DEFAULT_SYSTEM_PROMPT
from workplace_agent.llm.agent_response import ActionLogEntry, AgentFinalResponse, _AgentLLMResponse
from workplace_agent.llm.client import LLMClient
from workplace_agent.llm.types import (
    Message,
    ToolResultBlock,
    ToolUseBlock,
    assistant_blocks,
    user_text,
    user_tool_results,
)
from workplace_agent.tools.registry import ToolRegistry

TRUNCATED_RESULT_CHARS = 240
DEFAULT_MAX_STEPS = 12

STEP_LIMIT_SUMMARY_TEMPLATE = (
    "Reached the configured step limit of {limit} before finishing. "
    "The action log shows the partial progress made."
)
MAX_TOKENS_SUMMARY = (
    "Model hit the token limit before finishing. The action log shows the partial progress made."
)
REFUSAL_SUMMARY = "The model declined to process this request."
REFUSAL_QUESTION = "Could you rephrase your request?"


class AgentError(RuntimeError):
    pass


def _summarize_result(result: ToolResultBlock) -> str:
    text = result.content or ""
    if len(text) <= TRUNCATED_RESULT_CHARS:
        return text
    return text[:TRUNCATED_RESULT_CHARS] + "..."


def _action_log_entry(tool_use: ToolUseBlock, result: ToolResultBlock) -> ActionLogEntry:
    return ActionLogEntry(
        tool=tool_use.name,
        arguments=dict(tool_use.input),
        result_summary=_summarize_result(result),
        is_error=result.is_error,
    )


class Agent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._llm = llm
        self._registry = registry
        self._system_prompt = system_prompt
        self._max_steps = max_steps

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    async def run(self, user_prompt: str) -> AgentFinalResponse:
        messages: list[Message] = [user_text(user_prompt)]
        actions: list[ActionLogEntry] = []
        tools = self._registry.list_definitions()

        for _ in range(self._max_steps):
            response = await self._llm.create(
                system=self._system_prompt,
                messages=messages,
                tools=tools,
            )
            tool_uses = response.tool_uses()
            if response.stop_reason == "max_tokens":
                return AgentFinalResponse(
                    kind="partial",
                    summary=MAX_TOKENS_SUMMARY,
                    actions=actions,
                )
            if response.stop_reason == "refusal":
                return AgentFinalResponse(
                    kind="needs_clarification",
                    summary=REFUSAL_SUMMARY,
                    clarification_question=REFUSAL_QUESTION,
                    actions=actions,
                )
            if response.stop_reason not in ("end_turn", "tool_use", "stop_sequence"):
                raise AgentError(f"unexpected stop_reason {response.stop_reason!r}")
            if not tool_uses:
                if response.stop_reason == "tool_use":
                    raise AgentError("stop_reason=tool_use but no tool_use blocks in response")

                return await self._finalize(messages, actions)
            messages.append(assistant_blocks(response.content))

            tool_results: list[ToolResultBlock] = []
            for tool_use in tool_uses:
                result = await self._registry.dispatch(tool_use)
                tool_results.append(result)
                actions.append(_action_log_entry(tool_use, result))
            messages.append(user_tool_results(tool_results))

        return AgentFinalResponse(
            kind="partial",
            summary=STEP_LIMIT_SUMMARY_TEMPLATE.format(limit=self._max_steps),
            actions=actions,
        )

    async def _finalize(
        self,
        messages: list[Message],
        actions: list[ActionLogEntry],
    ) -> AgentFinalResponse:

        parse_response = await self._llm.parse(
            system=self._system_prompt,
            messages=messages,
            tools=[],
            output_format=_AgentLLMResponse,
        )
        parsed = parse_response.parsed
        if isinstance(parsed, _AgentLLMResponse):
            return AgentFinalResponse(
                kind=parsed.kind,
                summary=parsed.summary,
                clarification_question=parsed.clarification_question,
                actions=list(actions),
            )
        return AgentFinalResponse(
            kind="done",
            summary=parse_response.joined_text() or "(no summary returned)",
            actions=list(actions),
        )
