from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from workplace_agent.tools._http import http_error
from workplace_agent.tools.registry import ToolErrorResult, ToolRegistry, ToolSpec


class EmailModel(BaseModel):
    id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    sent_at: datetime
    thread_id: str | None = None


class EmailSendParams(BaseModel):
    sender: str = Field(..., description="Sender email address.")
    recipients: list[str] = Field(
        ..., min_length=1, description="One or more recipient email addresses; must be non-empty."
    )
    subject: str = Field(..., description="Subject line; must be non-empty.")
    body: str = Field(..., description="Email body text.")
    thread_id: str | None = Field(
        default=None, description="Existing thread id to append to. Optional."
    )


class EmailSendResult(BaseModel):
    ok: bool = True
    email: EmailModel


class EmailSearchParams(BaseModel):
    query: str = Field(..., min_length=1, description="Substring matched against subject and body.")


class EmailSearchHit(BaseModel):
    id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    sent_at: datetime
    thread_id: str | None = None


class EmailSearchResult(BaseModel):
    ok: bool = True
    results: list[EmailSearchHit]


def make_email_tools(client: httpx.AsyncClient) -> list[ToolSpec[Any, Any]]:
    async def send_email(params: EmailSendParams) -> EmailSendResult | ToolErrorResult:
        response = await client.post(
            "/email/send",
            json=params.model_dump(exclude_none=True),
        )
        if response.status_code != 200:
            return http_error(response)
        return EmailSendResult.model_validate(response.json())

    async def search_email(params: EmailSearchParams) -> EmailSearchResult | ToolErrorResult:
        response = await client.get("/email/search", params={"query": params.query})
        if response.status_code != 200:
            return http_error(response)
        return EmailSearchResult.model_validate(response.json())

    return [
        ToolSpec(
            name="email_send",
            description=(
                "Send an email to one or more recipients. Subject and body must be non-empty. "
                "Side effect: persists in the outbox."
            ),
            params_model=EmailSendParams,
            result_model=EmailSendResult,
            handler=send_email,
        ),
        ToolSpec(
            name="email_search",
            description=(
                "Search emails by case-insensitive substring across subject and body. "
                "Subject hits are ranked above body-only hits."
            ),
            params_model=EmailSearchParams,
            result_model=EmailSearchResult,
            handler=search_email,
        ),
    ]


def register_email_tools(registry: ToolRegistry, client: httpx.AsyncClient) -> None:
    for spec in make_email_tools(client):
        registry.register(spec)
