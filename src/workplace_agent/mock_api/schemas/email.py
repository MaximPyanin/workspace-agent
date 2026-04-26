import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _validate_address(value: str) -> str:
    candidate = value.strip()
    if not EMAIL_REGEX.match(candidate):
        raise ValueError(f"invalid email address: {value!r}")
    return candidate


class Email(BaseModel):
    id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    sent_at: datetime
    thread_id: str | None = None


class SearchResult(BaseModel):
    id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    sent_at: datetime
    thread_id: str | None = None


class SendEmailRequest(BaseModel):
    sender: str = Field(..., description="Sender email address")
    recipients: list[str] = Field(..., description="Recipient email addresses; at least one")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    thread_id: str | None = Field(default=None, description="Existing thread to append to")

    @field_validator("sender")
    @classmethod
    def _check_sender(cls, v: str) -> str:
        return _validate_address(v)

    @field_validator("recipients")
    @classmethod
    def _check_recipients(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one recipient is required")
        return [_validate_address(addr) for addr in v]


class SendEmailResponse(BaseModel):
    ok: bool = True
    email: Email


class SearchResponse(BaseModel):
    ok: bool = True
    results: list[SearchResult]
