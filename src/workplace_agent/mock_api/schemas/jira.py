from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Issue(BaseModel):
    key: str
    project: str
    summary: str
    description: str = ""
    status: str = "Open"
    assignee: str | None = None
    created_at: datetime
    updated_at: datetime


class CreateIssueRequest(BaseModel):
    project: str = Field(..., description="Project key prefix, e.g. 'ENG'")
    summary: str = Field(..., description="Short issue title")
    description: str = ""
    assignee: str | None = None
    status: str | None = Field(
        default=None,
        description="Optional initial status; defaults to 'Open' when omitted",
    )


class CreateIssueResponse(BaseModel):
    ok: bool = True
    issue: Issue


class TransitionRequest(BaseModel):
    status: str | None = Field(default=None, description="New workflow status")
    assignee: str | None = Field(default=None, description="New assignee handle")

    @model_validator(mode="after")
    def _require_one(self) -> Self:
        if self.status is None and self.assignee is None:
            raise ValueError("transition requires status or assignee")
        return self


class TransitionResponse(BaseModel):
    ok: bool = True
    issue: Issue


class GetIssueResponse(BaseModel):
    ok: bool = True
    issue: Issue
