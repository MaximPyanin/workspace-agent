import json

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from workplace_agent.llm.agent_response import ActionLogEntry, AgentFinalResponse


def _format_arguments(arguments: dict[str, object]) -> str:
    if not arguments:
        return ""
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(arguments)


def _format_result(action: ActionLogEntry) -> str:
    text = action.result_summary or ""
    if action.is_error:
        return f"ERROR: {text}" if text else "ERROR"
    return text


def _action_table(actions: list[ActionLogEntry]) -> Table:
    table = Table(title="Action log", expand=True)
    table.add_column("#", justify="right", no_wrap=True, style="cyan")
    table.add_column("tool", no_wrap=True)
    table.add_column("arguments", overflow="fold")
    table.add_column("result", overflow="fold")
    table.add_column("status", justify="center", no_wrap=True)
    for index, action in enumerate(actions, start=1):
        table.add_row(
            str(index),
            action.tool,
            _format_arguments(action.arguments),
            _format_result(action),
            Text("error", style="red") if action.is_error else Text("ok", style="green"),
        )
    return table


def render_response(response: AgentFinalResponse) -> RenderableType:
    parts: list[RenderableType] = []
    if response.kind == "needs_clarification":
        question = response.clarification_question or "(missing question)"
        body = Text(question, style="yellow")
        parts.append(Panel(body, title="Clarification needed", border_style="yellow", expand=True))
        if response.summary:
            parts.append(Text(response.summary, style="dim"))
    elif response.kind == "partial":
        parts.append(
            Panel(
                Text(response.summary or "(no summary)", style="yellow"),
                title="Partial (step limit reached)",
                border_style="yellow",
                expand=True,
            )
        )
    else:
        parts.append(
            Panel(
                Text(response.summary or "(no summary)", style="green"),
                title="Done",
                border_style="green",
                expand=True,
            )
        )
    if response.actions:
        parts.append(_action_table(response.actions))
    return Group(*parts)


def print_response(response: AgentFinalResponse, *, console: Console | None = None) -> None:
    target = console or Console()
    target.print(render_response(response))
