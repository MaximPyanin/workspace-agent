from rich.console import Console

from workplace_agent.agent.summary import print_response, render_response
from workplace_agent.llm.agent_response import ActionLogEntry, AgentFinalResponse


def _render_to_text(response: AgentFinalResponse) -> str:
    console = Console(record=True, width=120, color_system=None)
    print_response(response, console=console)
    return console.export_text()


def test_render_done_response_includes_summary_and_actions() -> None:
    response = AgentFinalResponse(
        kind="done",
        summary="message sent to engineering",
        actions=[
            ActionLogEntry(
                tool="slack_send_message",
                arguments={"channel": "#engineering", "text": "hi"},
                result_summary='{"ok":true,"ts":"123"}',
                is_error=False,
            )
        ],
    )

    text = _render_to_text(response)

    assert "message sent to engineering" in text
    assert "Done" in text
    assert "slack_send_message" in text
    assert "#engineering" in text
    assert "ok" in text


def test_render_clarification_response_shows_question() -> None:
    response = AgentFinalResponse(
        kind="needs_clarification",
        summary="ambiguous request",
        clarification_question="Which channel should I post to?",
    )

    text = _render_to_text(response)

    assert "Which channel should I post to?" in text
    assert "Clarification" in text


def test_render_done_without_actions_omits_table() -> None:
    response = AgentFinalResponse(kind="done", summary="all set")

    text = _render_to_text(response)

    assert "all set" in text
    assert "Action log" not in text


def test_render_action_marks_error_with_status() -> None:
    response = AgentFinalResponse(
        kind="done",
        summary="partial",
        actions=[
            ActionLogEntry(
                tool="jira_get_issue",
                arguments={"key": "ENG-1"},
                result_summary='{"error":"not_found"}',
                is_error=True,
            )
        ],
    )

    text = _render_to_text(response)

    assert "error" in text.lower()
    assert "ENG-1" in text


def test_render_response_returns_renderable() -> None:
    response = AgentFinalResponse(kind="done", summary="ok")

    renderable = render_response(response)

    console = Console(record=True, width=120, color_system=None)
    console.print(renderable)
    assert "ok" in console.export_text()
