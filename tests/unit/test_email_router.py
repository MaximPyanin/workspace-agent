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
    state.email.send(
        sender="ceo@example.com",
        recipients=["all@example.com"],
        subject="Q2 priorities",
        body="payments v2 and stability are top priorities for the team.",
        thread_id="thr_001",
    )
    state.email.send(
        sender="alice@example.com",
        recipients=["ceo@example.com"],
        subject="Re: Q2 priorities",
        body="quick clarification on payments v2 scope before we estimate.",
        thread_id="thr_001",
    )
    state.email.send(
        sender="bob@example.com",
        recipients=["alice@example.com"],
        subject="lunch?",
        body="want to grab a sandwich tomorrow?",
    )
    return state


@pytest_asyncio.fixture
async def state_and_client() -> AsyncIterator[tuple[AppState, httpx.AsyncClient]]:
    state = _make_state()
    app = create_app(state, seed_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield state, client


async def test_send_email_happy_path(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, client = state_and_client
    resp = await client.post(
        "/email/send",
        json={
            "sender": "alice@example.com",
            "recipients": ["bob@example.com", "carol@example.com"],
            "subject": "Status update",
            "body": "shipped the new auth flow today.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    email = body["email"]
    assert email["id"].startswith("em_")
    assert email["sender"] == "alice@example.com"
    assert email["recipients"] == ["bob@example.com", "carol@example.com"]
    assert email["subject"] == "Status update"
    assert email["thread_id"] is None
    stored = state.email.get(email["id"])
    assert stored is not None
    assert stored.sender == "alice@example.com"


async def test_send_email_rejects_empty_recipients(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/email/send",
        json={
            "sender": "alice@example.com",
            "recipients": [],
            "subject": "x",
            "body": "y",
        },
    )
    assert resp.status_code == 422


async def test_search_returns_relevance_ordering_subject_first(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get("/email/search", params={"query": "payments"})
    assert resp.status_code == 200
    results = resp.json()["results"]

    subjects = [r["subject"] for r in results]
    assert subjects == ["Q2 priorities", "Re: Q2 priorities"]


async def test_search_subject_match_outranks_body(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, client = state_and_client
    state.email.send(
        sender="dave@example.com",
        recipients=["alice@example.com"],
        subject="Sandwich logistics",
        body="planning lunch options",
    )
    resp = await client.get("/email/search", params={"query": "sandwich"})
    assert resp.status_code == 200
    results = resp.json()["results"]

    assert [r["subject"] for r in results] == ["Sandwich logistics", "lunch?"]


async def test_search_no_matches_returns_empty(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get("/email/search", params={"query": "kubernetes"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


async def test_search_rejects_empty_query(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get("/email/search", params={"query": ""})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_error"),
    [
        ("rate_limit", 429, "rate_limited"),
        ("not_found", 404, "not_found"),
        ("timeout", 504, "timeout"),
    ],
)
async def test_email_error_injection_short_circuits(
    error_type: ErrorType, expected_status: int, expected_error: str
) -> None:
    state = _make_state()
    rules = [
        ErrorInjectionRule(service="email", action="send_email", error_type=error_type),
    ]
    app = create_app(state, seed_dir=None, error_rules=rules, timeout_hang_seconds=0.01)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        injected = await client.post(
            "/email/send",
            json={
                "sender": "alice@example.com",
                "recipients": ["bob@example.com"],
                "subject": "x",
                "body": "y",
            },
        )
        unaffected = await client.get("/email/search", params={"query": "payments"})
    assert injected.status_code == expected_status
    assert injected.json() == {"ok": False, "error": expected_error}
    assert unaffected.status_code == 200
