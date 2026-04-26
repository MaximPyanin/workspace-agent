from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio

from workplace_agent.config import ErrorInjectionRule, ErrorType
from workplace_agent.mock_api.server import create_app
from workplace_agent.mock_api.state import AppState


def _fixed_clock():
    fixed = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    return lambda: fixed


def _make_state() -> AppState:
    state = AppState.empty(current_time=_fixed_clock())
    state.jira.create_issue(
        project="ENG",
        summary="Existing engineering ticket",
        description="prior",
        assignee="alice",
    )
    state.jira.create_issue(
        project="OPS",
        summary="Existing ops ticket",
        assignee="bob",
        status="In Progress",
    )
    return state


@pytest_asyncio.fixture
async def state_and_client() -> AsyncIterator[tuple[AppState, httpx.AsyncClient]]:
    state = _make_state()
    app = create_app(state, seed_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield state, client


async def test_create_issue_assigns_ascending_keys_per_project(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    first = await client.post(
        "/jira/issues",
        json={"project": "ENG", "summary": "Add login button"},
    )
    second = await client.post(
        "/jira/issues",
        json={"project": "ENG", "summary": "Polish login button"},
    )
    third = await client.post(
        "/jira/issues",
        json={"project": "OPS", "summary": "Rotate keys"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert first.json()["issue"]["key"] == "ENG-2"
    assert second.json()["issue"]["key"] == "ENG-3"
    assert third.json()["issue"]["key"] == "OPS-2"


async def test_create_issue_returns_full_payload_with_defaults(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/jira/issues",
        json={
            "project": "ENG",
            "summary": "Investigate flaky test",
            "description": "TestX failed 3x in a row",
            "assignee": "carol",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    issue = body["issue"]
    assert issue["project"] == "ENG"
    assert issue["summary"] == "Investigate flaky test"
    assert issue["description"] == "TestX failed 3x in a row"
    assert issue["assignee"] == "carol"
    assert issue["status"] == "Open"
    assert issue["created_at"] == issue["updated_at"]


async def test_create_issue_rejects_empty_summary(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/jira/issues",
        json={"project": "ENG", "summary": "   "},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "empty_summary"


async def test_get_issue_returns_full_payload(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get("/jira/issues/ENG-1")
    assert resp.status_code == 200
    issue = resp.json()["issue"]
    assert issue["key"] == "ENG-1"
    assert issue["summary"] == "Existing engineering ticket"
    assert issue["assignee"] == "alice"
    assert issue["status"] == "Open"


async def test_get_issue_unknown_returns_404(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get("/jira/issues/ENG-9999")
    assert resp.status_code == 404
    body = resp.json()["detail"]
    assert body["ok"] is False
    assert body["error"] == "issue_not_found"
    assert body["key"] == "ENG-9999"


async def test_transition_issue_updates_status(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, client = state_and_client
    resp = await client.post(
        "/jira/issues/ENG-1/transition",
        json={"status": "In Progress"},
    )
    assert resp.status_code == 200
    issue = resp.json()["issue"]
    assert issue["status"] == "In Progress"
    stored = state.jira.get_issue("ENG-1")
    assert stored is not None
    assert stored.status == "In Progress"


async def test_transition_issue_unknown_returns_404(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/jira/issues/NOPE-1/transition",
        json={"status": "Done"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "issue_not_found"


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_error"),
    [
        ("rate_limit", 429, "rate_limited"),
        ("not_found", 404, "not_found"),
        ("timeout", 504, "timeout"),
    ],
)
async def test_jira_error_injection_short_circuits(
    error_type: ErrorType, expected_status: int, expected_error: str
) -> None:
    state = _make_state()
    rules = [
        ErrorInjectionRule(service="jira", action="get_issue", error_type=error_type),
    ]
    app = create_app(state, seed_dir=None, error_rules=rules, timeout_hang_seconds=0.01)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        injected = await client.get("/jira/issues/ENG-1")
        unaffected = await client.post(
            "/jira/issues",
            json={"project": "ENG", "summary": "should still work"},
        )
    assert injected.status_code == expected_status
    assert injected.json() == {"ok": False, "error": expected_error}
    assert unaffected.status_code == 200
