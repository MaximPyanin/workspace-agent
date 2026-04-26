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
    state.slack.add_channel("engineering")
    state.slack.add_channel("product")
    state.slack.send_message(channel="engineering", text="deploy in 5 minutes", user="U001")
    state.slack.send_message(channel="product", text="roadmap review tomorrow", user="U002")
    return state


@pytest_asyncio.fixture
async def state_and_client() -> AsyncIterator[tuple[AppState, httpx.AsyncClient]]:
    state = _make_state()
    app = create_app(state, seed_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield state, client


async def test_list_channels_returns_seeded_channels(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get("/slack/channels")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    names = sorted(c["name"] for c in body["channels"])
    assert names == ["engineering", "product"]
    for c in body["channels"]:
        assert c["id"].startswith("C")


async def test_send_message_to_existing_channel_appends_to_history(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, client = state_and_client
    before = len(state.slack.list_messages(channel="engineering"))
    resp = await client.post(
        "/slack/send_message",
        json={"channel": "#engineering", "text": "PR-42 merged", "user": "U001"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["ts"]
    eng = state.slack.get_channel_by_name("engineering")
    assert eng is not None
    assert body["channel"] == eng.id
    after = state.slack.list_messages(channel="engineering")
    assert len(after) == before + 1
    assert after[-1].text == "PR-42 merged"
    assert after[-1].ts == body["ts"]


async def test_send_message_unknown_channel_without_hash_returns_404(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/slack/send_message",
        json={"channel": "no-such-channel", "text": "hi"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["detail"]["ok"] is False
    assert body["detail"]["error"] == "channel_not_found"


async def test_send_message_empty_text_returns_400(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/slack/send_message",
        json={"channel": "#engineering", "text": "   "},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["ok"] is False
    assert body["detail"]["error"] == "empty_text"


async def test_search_messages_substring_match_includes_channel_metadata(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, client = state_and_client
    resp = await client.get("/slack/search", params={"query": "deploy"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert len(body["messages"]) == 1
    match = body["messages"][0]
    assert match["text"] == "deploy in 5 minutes"
    eng = state.slack.get_channel_by_name("engineering")
    assert eng is not None
    assert match["channel"] == eng.id
    assert match["channel_name"] == "engineering"


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_error"),
    [
        ("rate_limit", 429, "rate_limited"),
        ("not_found", 404, "not_found"),
        ("timeout", 504, "timeout"),
    ],
)
async def test_send_message_error_injection_short_circuits(
    error_type: ErrorType, expected_status: int, expected_error: str
) -> None:
    state = _make_state()
    rules = [
        ErrorInjectionRule(service="slack", action="send_message", error_type=error_type),
    ]
    app = create_app(state, seed_dir=None, error_rules=rules, timeout_hang_seconds=0.01)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/slack/send_message",
            json={"channel": "#engineering", "text": "hi"},
        )
    assert resp.status_code == expected_status
    assert resp.json() == {"ok": False, "error": expected_error}
    assert all(m.text != "hi" for m in state.slack.list_messages(channel="engineering"))


async def test_search_error_injection_only_targets_search_action() -> None:
    state = _make_state()
    rules = [
        ErrorInjectionRule(service="slack", action="search_messages", error_type="rate_limit"),
    ]
    app = create_app(state, seed_dir=None, error_rules=rules)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        search_resp = await client.get("/slack/search", params={"query": "deploy"})
        channels_resp = await client.get("/slack/channels")
    assert search_resp.status_code == 429
    assert search_resp.json() == {"ok": False, "error": "rate_limited"}
    assert channels_resp.status_code == 200
