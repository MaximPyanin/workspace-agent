from collections.abc import AsyncIterator
from datetime import UTC, datetime, time, timedelta

import httpx
import pytest
import pytest_asyncio

from workplace_agent.config import ErrorInjectionRule, ErrorType
from workplace_agent.mock_api.routers.calendar import _scan_for_slot
from workplace_agent.mock_api.schemas.calendar import WorkingHours
from workplace_agent.mock_api.server import create_app
from workplace_agent.mock_api.state import AppState, CalendarEvent


def _fixed_clock():
    fixed = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    return lambda: fixed


def _dt(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 4, day, hour, minute, tzinfo=UTC)


def _make_state() -> AppState:
    state = AppState.empty(current_time=_fixed_clock())

    state.calendar.create_event(
        title="Standup",
        start=_dt(27, 9, 0),
        end=_dt(27, 9, 30),
    )
    state.calendar.create_event(title="Lunch", start=_dt(27, 12, 0), end=_dt(27, 13, 0))
    return state


@pytest_asyncio.fixture
async def state_and_client() -> AsyncIterator[tuple[AppState, httpx.AsyncClient]]:
    state = _make_state()
    app = create_app(state, seed_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield state, client


async def test_create_event_happy_path(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    state, client = state_and_client
    resp = await client.post(
        "/calendar/events",
        json={
            "title": "Design review",
            "start": "2026-04-28T14:00:00+00:00",
            "end": "2026-04-28T15:00:00+00:00",
            "attendees": ["alice@example.com", "bob@example.com"],
            "description": "Review the new schema",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    event = body["event"]
    assert event["title"] == "Design review"
    assert event["attendees"] == ["alice@example.com", "bob@example.com"]
    assert event["description"] == "Review the new schema"
    assert event["id"].startswith("evt_")
    assert state.calendar.get_event(event["id"]) is not None


async def test_create_event_rejects_end_before_start(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/calendar/events",
        json={
            "title": "Bad",
            "start": "2026-04-28T15:00:00+00:00",
            "end": "2026-04-28T14:00:00+00:00",
        },
    )
    assert resp.status_code == 422


async def test_create_event_rejects_naive_datetime(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/calendar/events",
        json={
            "title": "No-tz event",
            "start": "2026-04-28T14:00:00",
            "end": "2026-04-28T15:00:00",
        },
    )
    assert resp.status_code == 422


async def test_list_events_returns_all_when_no_range(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get("/calendar/events")
    assert resp.status_code == 200
    titles = [e["title"] for e in resp.json()["events"]]
    assert titles == ["Standup", "Lunch"]


async def test_list_events_filters_by_range(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.get(
        "/calendar/events",
        params={
            "from": "2026-04-27T11:00:00+00:00",
            "to": "2026-04-27T18:00:00+00:00",
        },
    )
    assert resp.status_code == 200
    titles = [e["title"] for e in resp.json()["events"]]
    assert titles == ["Lunch"]


async def test_find_free_slot_returns_gap_between_two_events(
    state_and_client: tuple[AppState, httpx.AsyncClient],
) -> None:
    _state, client = state_and_client
    resp = await client.post(
        "/calendar/find_free_slot",
        json={
            "duration_minutes": 30,
            "range_start": "2026-04-27T09:00:00+00:00",
            "range_end": "2026-04-27T15:00:00+00:00",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True

    assert body["start"] == "2026-04-27T09:30:00Z"
    assert body["end"] == "2026-04-27T10:00:00Z"


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_error"),
    [
        ("rate_limit", 429, "rate_limited"),
        ("not_found", 404, "not_found"),
        ("timeout", 504, "timeout"),
    ],
)
async def test_calendar_error_injection_short_circuits(
    error_type: ErrorType, expected_status: int, expected_error: str
) -> None:
    state = _make_state()
    rules = [
        ErrorInjectionRule(service="calendar", action="find_free_slot", error_type=error_type),
    ]
    app = create_app(state, seed_dir=None, error_rules=rules, timeout_hang_seconds=0.01)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        injected = await client.post(
            "/calendar/find_free_slot",
            json={
                "duration_minutes": 30,
                "range_start": "2026-04-27T09:00:00+00:00",
                "range_end": "2026-04-27T15:00:00+00:00",
            },
        )
        unaffected = await client.get("/calendar/events")
    assert injected.status_code == expected_status
    assert injected.json() == {"ok": False, "error": expected_error}
    assert unaffected.status_code == 200


def test_scan_empty_calendar_returns_window_start() -> None:
    slot = _scan_for_slot(
        duration=timedelta(minutes=45),
        range_start=_dt(28, 9, 0),
        range_end=_dt(28, 17, 0),
        events=[],
        working_hours=None,
    )
    assert slot == (_dt(28, 9, 0), _dt(28, 9, 45))


def test_scan_fully_busy_returns_none() -> None:
    blocker = CalendarEvent(
        id="evt_blocker",
        title="all-day",
        start=_dt(28, 9, 0),
        end=_dt(28, 17, 0),
    )
    slot = _scan_for_slot(
        duration=timedelta(minutes=30),
        range_start=_dt(28, 9, 0),
        range_end=_dt(28, 17, 0),
        events=[blocker],
        working_hours=None,
    )
    assert slot is None


def test_scan_finds_gap_between_two_events() -> None:
    morning = CalendarEvent(id="evt_a", title="m", start=_dt(28, 9, 0), end=_dt(28, 10, 0))
    later = CalendarEvent(id="evt_b", title="l", start=_dt(28, 11, 0), end=_dt(28, 12, 0))
    slot = _scan_for_slot(
        duration=timedelta(minutes=30),
        range_start=_dt(28, 9, 0),
        range_end=_dt(28, 13, 0),
        events=[later, morning],
        working_hours=None,
    )
    assert slot == (_dt(28, 10, 0), _dt(28, 10, 30))


def test_scan_with_working_hours_skips_outside_window() -> None:

    early = CalendarEvent(id="e", title="e", start=_dt(28, 8, 0), end=_dt(28, 8, 45))
    slot = _scan_for_slot(
        duration=timedelta(minutes=60),
        range_start=_dt(28, 0, 0),
        range_end=_dt(28, 23, 59),
        events=[early],
        working_hours=WorkingHours(start=time(9, 0), end=time(17, 0)),
    )
    assert slot == (_dt(28, 9, 0), _dt(28, 10, 0))


def test_scan_window_crossing_midnight_uses_next_day() -> None:

    busy_first_day = CalendarEvent(id="x", title="x", start=_dt(27, 9, 0), end=_dt(27, 17, 0))
    slot = _scan_for_slot(
        duration=timedelta(minutes=60),
        range_start=_dt(27, 9, 0),
        range_end=_dt(28, 17, 0),
        events=[busy_first_day],
        working_hours=WorkingHours(start=time(9, 0), end=time(17, 0)),
    )
    assert slot == (_dt(28, 9, 0), _dt(28, 10, 0))
