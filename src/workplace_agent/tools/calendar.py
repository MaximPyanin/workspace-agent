from datetime import datetime, time
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from workplace_agent.tools._http import http_error
from workplace_agent.tools.registry import ToolErrorResult, ToolRegistry, ToolSpec


class CalendarEventModel(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = Field(default_factory=list)
    description: str = ""


class CalendarCreateEventParams(BaseModel):
    title: str = Field(..., description="Event title.")
    start: datetime = Field(..., description="ISO-8601 start time (include timezone).")
    end: datetime = Field(..., description="ISO-8601 end time, must be after start.")
    attendees: list[str] = Field(
        default_factory=list, description="List of attendee emails or handles."
    )
    description: str = Field(default="", description="Optional event description.")


class CalendarCreateEventResult(BaseModel):
    ok: bool = True
    event: CalendarEventModel


class CalendarListEventsParams(BaseModel):
    range_start: datetime | None = Field(
        default=None, description="Optional ISO-8601 lower bound (inclusive, timezone-aware)."
    )
    range_end: datetime | None = Field(
        default=None, description="Optional ISO-8601 upper bound (exclusive, timezone-aware)."
    )

    @field_validator("range_start", "range_end", mode="after")
    @classmethod
    def _require_tz(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v


class CalendarListEventsResult(BaseModel):
    ok: bool = True
    events: list[CalendarEventModel]


class CalendarWorkingHoursModel(BaseModel):
    start: time = Field(..., description="Daily window start (e.g. '09:00').")
    end: time = Field(..., description="Daily window end (e.g. '17:00').")


class CalendarFindFreeSlotParams(BaseModel):
    duration_minutes: int = Field(..., gt=0, description="Slot length in minutes.")
    range_start: datetime = Field(..., description="Earliest acceptable slot start.")
    range_end: datetime = Field(..., description="Latest acceptable slot end.")
    working_hours: CalendarWorkingHoursModel | None = Field(
        default=None,
        description="Optional daily window restricting candidate slots.",
    )


class CalendarFindFreeSlotResult(BaseModel):
    ok: bool = True
    found: bool
    start: datetime | None = None
    end: datetime | None = None
    reason: str | None = None


def make_calendar_tools(client: httpx.AsyncClient) -> list[ToolSpec[Any, Any]]:
    async def create_event(
        params: CalendarCreateEventParams,
    ) -> CalendarCreateEventResult | ToolErrorResult:
        response = await client.post(
            "/calendar/events",
            json=params.model_dump(mode="json"),
        )
        if response.status_code != 200:
            return http_error(response)
        return CalendarCreateEventResult.model_validate(response.json())

    async def list_events(
        params: CalendarListEventsParams,
    ) -> CalendarListEventsResult | ToolErrorResult:
        query: dict[str, str] = {}
        if params.range_start is not None:
            query["from"] = params.range_start.isoformat()
        if params.range_end is not None:
            query["to"] = params.range_end.isoformat()
        response = await client.get("/calendar/events", params=query)
        if response.status_code != 200:
            return http_error(response)
        return CalendarListEventsResult.model_validate(response.json())

    async def find_free_slot(
        params: CalendarFindFreeSlotParams,
    ) -> CalendarFindFreeSlotResult | ToolErrorResult:
        response = await client.post(
            "/calendar/find_free_slot",
            json=params.model_dump(mode="json"),
        )
        if response.status_code != 200:
            return http_error(response)
        return CalendarFindFreeSlotResult.model_validate(response.json())

    return [
        ToolSpec(
            name="calendar_create_event",
            description=(
                "Create a calendar event with the given title, start, and end (ISO-8601 with "
                "timezone). Side effect: blocks the time on the calendar."
            ),
            params_model=CalendarCreateEventParams,
            result_model=CalendarCreateEventResult,
            handler=create_event,
        ),
        ToolSpec(
            name="calendar_list_events",
            description=(
                "List calendar events. Optional 'range_start' and 'range_end' bound the window; "
                "results are sorted by start time."
            ),
            params_model=CalendarListEventsParams,
            result_model=CalendarListEventsResult,
            handler=list_events,
        ),
        ToolSpec(
            name="calendar_find_free_slot",
            description=(
                "Find the earliest available slot of 'duration_minutes' within the given range, "
                "optionally restricted to daily 'working_hours'. Returns 'found=false' when the "
                "calendar offers no gap that fits."
            ),
            params_model=CalendarFindFreeSlotParams,
            result_model=CalendarFindFreeSlotResult,
            handler=find_free_slot,
        ),
    ]


def register_calendar_tools(registry: ToolRegistry, client: httpx.AsyncClient) -> None:
    for spec in make_calendar_tools(client):
        registry.register(spec)
