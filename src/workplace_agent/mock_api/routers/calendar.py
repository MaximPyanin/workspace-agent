from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request

from workplace_agent.mock_api.schemas.calendar import (
    CreateEventRequest,
    CreateEventResponse,
    Event,
    FreeSlotRequest,
    FreeSlotResponse,
    ListEventsResponse,
    WorkingHours,
)
from workplace_agent.mock_api.state import AppState, CalendarEvent


def _state(request: Request) -> AppState:
    state = request.app.state.app_state
    assert isinstance(state, AppState)
    return state


def _to_schema(event: CalendarEvent) -> Event:
    return Event(
        id=event.id,
        title=event.title,
        start=event.start,
        end=event.end,
        attendees=list(event.attendees),
        description=event.description,
    )


router = APIRouter(prefix="/calendar", tags=["calendar"])


@router.post("/events", name="create_event", response_model=CreateEventResponse)
async def create_event(payload: CreateEventRequest, request: Request) -> CreateEventResponse:
    state = _state(request)
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "empty_title"})
    try:
        event = state.calendar.create_event(
            title=title,
            start=payload.start,
            end=payload.end,
            attendees=payload.attendees,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)}) from None
    return CreateEventResponse(ok=True, event=_to_schema(event))


@router.get("/events", name="list_events", response_model=ListEventsResponse)
async def list_events(
    request: Request,
    from_: Annotated[datetime | None, Query(alias="from")] = None,
    to: Annotated[datetime | None, Query()] = None,
) -> ListEventsResponse:
    if from_ is not None and from_.tzinfo is None:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "naive_datetime"})
    if to is not None and to.tzinfo is None:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "naive_datetime"})
    if from_ is not None and to is not None and to <= from_:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "to_must_be_after_from"})
    state = _state(request)
    events = state.calendar.list_events(from_=from_, to=to)
    return ListEventsResponse(ok=True, events=[_to_schema(e) for e in events])


@router.post("/find_free_slot", name="find_free_slot", response_model=FreeSlotResponse)
async def find_free_slot(payload: FreeSlotRequest, request: Request) -> FreeSlotResponse:
    state = _state(request)
    duration = timedelta(minutes=payload.duration_minutes)
    events = state.calendar.list_events(from_=payload.range_start, to=payload.range_end)
    found = _scan_for_slot(
        duration=duration,
        range_start=payload.range_start,
        range_end=payload.range_end,
        events=events,
        working_hours=payload.working_hours,
    )
    if found is None:
        return FreeSlotResponse(ok=True, found=False, reason="no_slot_available")
    start, end = found
    return FreeSlotResponse(ok=True, found=True, start=start, end=end)


def _iter_windows(
    range_start: datetime,
    range_end: datetime,
    working_hours: WorkingHours | None,
) -> Iterator[tuple[datetime, datetime]]:
    if working_hours is None:
        yield range_start, range_end
        return
    tz = range_start.tzinfo
    cursor_day = range_start.date()
    end_day = range_end.date()
    while cursor_day <= end_day:
        wh_start = datetime.combine(cursor_day, working_hours.start, tzinfo=tz)
        wh_end = datetime.combine(cursor_day, working_hours.end, tzinfo=tz)
        ws = max(wh_start, range_start)
        we = min(wh_end, range_end)
        if ws < we:
            yield ws, we
        cursor_day = cursor_day + timedelta(days=1)


def _scan_for_slot(
    *,
    duration: timedelta,
    range_start: datetime,
    range_end: datetime,
    events: list[CalendarEvent],
    working_hours: WorkingHours | None,
) -> tuple[datetime, datetime] | None:
    sorted_events = sorted(events, key=lambda e: e.start)
    for w_start, w_end in _iter_windows(range_start, range_end, working_hours):
        cursor = w_start
        for ev in sorted_events:
            if ev.end <= w_start or ev.start >= w_end:
                continue
            ev_start = max(ev.start, w_start)
            ev_end = min(ev.end, w_end)
            if ev_start - cursor >= duration:
                return cursor, cursor + duration
            if ev_end > cursor:
                cursor = ev_end
        if w_end - cursor >= duration:
            return cursor, cursor + duration
    return None
