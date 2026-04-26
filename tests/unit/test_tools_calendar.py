from datetime import UTC, datetime, time
from typing import Any

import httpx
import pytest
import respx

from workplace_agent.tools.calendar import (
    CalendarCreateEventParams,
    CalendarFindFreeSlotParams,
    CalendarWorkingHoursModel,
    make_calendar_tools,
)
from workplace_agent.tools.registry import ToolErrorResult

BASE_URL = "http://mock"

EVENT_PAYLOAD = {
    "id": "evt_000001",
    "title": "standup",
    "start": "2026-04-27T09:00:00+00:00",
    "end": "2026-04-27T09:30:00+00:00",
    "attendees": [],
    "description": "",
}


def _tools(client: httpx.AsyncClient) -> dict[str, Any]:
    return {spec.name: spec for spec in make_calendar_tools(client)}


@pytest.mark.asyncio
async def test_create_event_serialises_datetimes_iso() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            route = router.post("/calendar/events").mock(
                return_value=httpx.Response(200, json={"ok": True, "event": EVENT_PAYLOAD})
            )
            params = CalendarCreateEventParams(
                title="standup",
                start=datetime(2026, 4, 27, 9, 0, tzinfo=UTC),
                end=datetime(2026, 4, 27, 9, 30, tzinfo=UTC),
            )
            result = await _tools(client)["calendar_create_event"].handler(params)  # type: ignore[attr-defined]

    sent = route.calls.last.request.read()
    assert b"2026-04-27T09:00:00" in sent
    assert b"2026-04-27T09:30:00" in sent
    assert result.event.id == "evt_000001"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_find_free_slot_returns_typed_result() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            route = router.post("/calendar/find_free_slot").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "found": True,
                        "start": "2026-04-27T10:00:00+00:00",
                        "end": "2026-04-27T10:30:00+00:00",
                    },
                )
            )
            params = CalendarFindFreeSlotParams(
                duration_minutes=30,
                range_start=datetime(2026, 4, 27, 9, 0, tzinfo=UTC),
                range_end=datetime(2026, 4, 27, 17, 0, tzinfo=UTC),
                working_hours=CalendarWorkingHoursModel(start=time(9, 0), end=time(17, 0)),
            )
            result = await _tools(client)["calendar_find_free_slot"].handler(params)  # type: ignore[attr-defined]
    sent = route.calls.last.request.read()
    assert b"working_hours" in sent
    assert result.found is True  # type: ignore[union-attr]
    assert result.start is not None  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_find_free_slot_4xx_becomes_tool_error_result() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            router.post("/calendar/find_free_slot").mock(
                return_value=httpx.Response(429, json={"ok": False, "error": "rate_limited"})
            )
            params = CalendarFindFreeSlotParams(
                duration_minutes=30,
                range_start=datetime(2026, 4, 27, 9, 0, tzinfo=UTC),
                range_end=datetime(2026, 4, 27, 17, 0, tzinfo=UTC),
            )
            result = await _tools(client)["calendar_find_free_slot"].handler(params)  # type: ignore[attr-defined]
    assert isinstance(result, ToolErrorResult)
    assert result.error == "rate_limited"
