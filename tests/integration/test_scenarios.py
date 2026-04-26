from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest_asyncio

from workplace_agent.agent.loop import Agent
from workplace_agent.config import ErrorInjectionRule
from workplace_agent.llm.agent_response import AgentFinalResponse
from workplace_agent.llm.fake import FakeLLM
from workplace_agent.llm.types import (
    LLMResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from workplace_agent.mock_api.server import create_app
from workplace_agent.mock_api.state import AppState
from workplace_agent.tools import default_registry


def _fixed_clock():
    fixed = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    return lambda: fixed


def _create_tool_use(name: str, args: dict[str, Any], *, tool_use_id: str) -> LLMResponse:
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


def _parse_done(summary: str) -> LLMResponse:
    parsed = AgentFinalResponse(kind="done", summary=summary)
    return LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text=summary)],
        usage=Usage(),
        parsed=parsed,
    )


def _parse_clarification(question: str, summary: str = "needs more info") -> LLMResponse:
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


@pytest_asyncio.fixture
async def integration_env() -> AsyncIterator[tuple[AppState, httpx.AsyncClient]]:
    state = AppState.empty(current_time=_fixed_clock())
    app = create_app(state, seed_dir=None)
    state.load_seeds()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-api") as client:
        yield state, client


async def test_scenario_a_simple_slack_message(
    integration_env: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, http_client = integration_env
    fake = FakeLLM(
        create_responses=[
            _create_tool_use(
                "slack_send_message",
                {"channel": "#engineering", "text": "Build is green"},
                tool_use_id="tu_1",
            ),
            _create_end_turn("posted"),
        ],
        parse_responses=[
            _parse_done("Posted 'Build is green' to #engineering."),
        ],
    )
    registry = default_registry(http_client)
    agent = Agent(llm=fake, registry=registry, max_steps=5)

    result = await agent.run("Send a message to #engineering saying 'Build is green'")

    assert result.kind == "done"
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.tool == "slack_send_message"
    assert action.is_error is False
    assert action.arguments == {"channel": "#engineering", "text": "Build is green"}

    eng = state.slack.get_channel_by_name("engineering")
    assert eng is not None
    posted_messages = state.slack.list_messages(channel="engineering")
    assert any(m.text == "Build is green" for m in posted_messages)

    search_resp = await http_client.get("/slack/search", params={"query": "Build is green"})
    assert search_resp.status_code == 200
    body = search_resp.json()
    assert body["ok"] is True
    assert any(m["text"] == "Build is green" for m in body["messages"])


async def test_scenario_b_jira_create_transition_then_slack_notify(
    integration_env: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, http_client = integration_env
    fake = FakeLLM(
        create_responses=[
            _create_tool_use(
                "jira_create_issue",
                {
                    "project": "ENG",
                    "summary": "Login button regression",
                    "description": "Reported by QA on 2026-04-24. Submit form does nothing.",
                },
                tool_use_id="tu_1",
            ),
            _create_tool_use(
                "jira_transition_issue",
                {"key": "ENG-3", "status": "In Progress", "assignee": "alice"},
                tool_use_id="tu_2",
            ),
            _create_tool_use(
                "slack_send_message",
                {
                    "channel": "#engineering",
                    "text": "Filed ENG-3 (Login button regression), assigned to alice and moved to In Progress.",
                },
                tool_use_id="tu_3",
            ),
            _create_end_turn("multi-step done"),
        ],
        parse_responses=[
            _parse_done(
                "Filed ENG-3 'Login button regression', moved it to In Progress, "
                "assigned alice, and posted a heads-up in #engineering."
            ),
        ],
    )
    registry = default_registry(http_client)
    agent = Agent(llm=fake, registry=registry, max_steps=8)

    result = await agent.run(
        "QA found a regression in the login button - file a bug, assign alice, "
        "mark it in progress, and tell #engineering."
    )

    assert result.kind == "done"
    assert [a.tool for a in result.actions] == [
        "jira_create_issue",
        "jira_transition_issue",
        "slack_send_message",
    ]
    assert all(not a.is_error for a in result.actions)

    issue = state.jira.get_issue("ENG-3")
    assert issue is not None
    assert issue.summary == "Login button regression"
    assert issue.status == "In Progress"
    assert issue.assignee == "alice"

    eng_msgs = state.slack.list_messages(channel="engineering")
    assert any("ENG-3" in m.text for m in eng_msgs)


async def test_scenario_c_calendar_search_then_find_free_slot(
    integration_env: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, http_client = integration_env
    range_start = "2026-04-27T09:00:00+00:00"
    range_end = "2026-04-27T17:00:00+00:00"
    fake = FakeLLM(
        create_responses=[
            _create_tool_use(
                "calendar_list_events",
                {"range_start": range_start, "range_end": range_end},
                tool_use_id="tu_1",
            ),
            _create_tool_use(
                "calendar_find_free_slot",
                {
                    "duration_minutes": 30,
                    "range_start": range_start,
                    "range_end": range_end,
                },
                tool_use_id="tu_2",
            ),
            _create_end_turn("found a slot"),
        ],
        parse_responses=[
            _parse_done("Looked at Monday's calendar and identified a 30-minute free slot."),
        ],
    )
    registry = default_registry(http_client)
    agent = Agent(llm=fake, registry=registry, max_steps=6)

    result = await agent.run("Find me a 30-minute slot on Monday during business hours.")

    assert result.kind == "done"
    assert [a.tool for a in result.actions] == [
        "calendar_list_events",
        "calendar_find_free_slot",
    ]
    free_slot_action = result.actions[1]
    assert free_slot_action.is_error is False

    standup = next(e for e in state.calendar.list_events() if e.title == "Engineering standup")
    standup_start = standup.start
    standup_end = standup.end

    free_slot_resp = await http_client.post(
        "/calendar/find_free_slot",
        json={
            "duration_minutes": 30,
            "range_start": range_start,
            "range_end": range_end,
        },
    )
    assert free_slot_resp.status_code == 200
    body = free_slot_resp.json()
    assert body["ok"] is True
    assert body["found"] is True
    slot_start = datetime.fromisoformat(body["start"])
    slot_end = datetime.fromisoformat(body["end"])
    assert (slot_end - slot_start).total_seconds() == 30 * 60
    assert slot_end <= standup_start or slot_start >= standup_end


async def test_error_injection_surfaces_recoverable_error_in_summary() -> None:
    state = AppState.empty(current_time=_fixed_clock())
    rules = [
        ErrorInjectionRule(service="slack", action="send_message", error_type="rate_limit"),
    ]
    app = create_app(state, seed_dir=None, error_rules=rules)
    state.load_seeds()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-api") as http_client:
        fake = FakeLLM(
            create_responses=[
                _create_tool_use(
                    "slack_send_message",
                    {"channel": "#engineering", "text": "Build is green"},
                    tool_use_id="tu_1",
                ),
                _create_end_turn("hit a rate limit"),
            ],
            parse_responses=[
                _parse_done(
                    "Tried to post to #engineering but Slack returned rate_limited; retry needed."
                ),
            ],
        )
        registry = default_registry(http_client)
        agent = Agent(llm=fake, registry=registry, max_steps=4)

        result = await agent.run("Send 'Build is green' to #engineering")

        assert result.kind == "done"
        assert len(result.actions) == 1
        action = result.actions[0]
        assert action.tool == "slack_send_message"
        assert action.is_error is True
        assert "rate_limited" in action.result_summary
        assert "rate_limited" in result.summary

        posted = state.slack.list_messages(channel="engineering")
        assert all(m.text != "Build is green" for m in posted)


async def test_scenario_d_ambiguous_prompt_triggers_clarification(
    integration_env: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, http_client = integration_env
    fake = FakeLLM(
        create_responses=[
            _create_end_turn("?"),
        ],
        parse_responses=[
            _parse_clarification(
                "Which channel should I post the update to, and what should it say?"
            ),
        ],
    )
    registry = default_registry(http_client)
    agent = Agent(llm=fake, registry=registry, max_steps=4)

    before_messages = list(state.slack.list_messages())

    result = await agent.run("Send an update")

    assert result.kind == "needs_clarification"
    assert result.clarification_question is not None
    assert "channel" in result.clarification_question.lower()
    assert result.actions == []

    assert len(fake.create_calls) == 1
    assert len(fake.parse_calls) == 1

    after_messages = list(state.slack.list_messages())
    assert before_messages == after_messages
