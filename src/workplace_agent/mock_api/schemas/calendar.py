from datetime import datetime, time
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Event(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = Field(default_factory=list)
    description: str = ""


class CreateEventRequest(BaseModel):
    title: str = Field(..., description="Event title")
    start: datetime = Field(..., description="ISO-8601 start time")
    end: datetime = Field(..., description="ISO-8601 end time, must be after start")
    attendees: list[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("start", "end", mode="after")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _validate_times(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class CreateEventResponse(BaseModel):
    ok: bool = True
    event: Event


class ListEventsResponse(BaseModel):
    ok: bool = True
    events: list[Event]


class WorkingHours(BaseModel):
    start: time = Field(..., description="Start of working window (local, naive)")
    end: time = Field(..., description="End of working window, must be after start")

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if self.end <= self.start:
            raise ValueError("working_hours.end must be after working_hours.start")
        return self


class FreeSlotRequest(BaseModel):
    duration_minutes: int = Field(..., gt=0, description="Length of slot to find, in minutes")
    range_start: datetime = Field(..., description="Earliest acceptable slot start")
    range_end: datetime = Field(..., description="Latest acceptable slot end")
    working_hours: WorkingHours | None = Field(
        default=None,
        description="Optional daily window; when omitted, the full range is considered",
    )

    @field_validator("range_start", "range_end", mode="after")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.range_end <= self.range_start:
            raise ValueError("range_end must be after range_start")
        return self


class FreeSlotResponse(BaseModel):
    ok: bool = True
    found: bool
    start: datetime | None = None
    end: datetime | None = None
    reason: str | None = None
