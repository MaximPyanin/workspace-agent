from typing import Any

import httpx
import pytest
import respx

from workplace_agent.tools.email import (
    EmailSendParams,
    make_email_tools,
)
from workplace_agent.tools.registry import ToolErrorResult

BASE_URL = "http://mock"

EMAIL_PAYLOAD = {
    "id": "em_000001",
    "sender": "alice@example.com",
    "recipients": ["bob@example.com"],
    "subject": "hello",
    "body": "hi there",
    "sent_at": "2026-04-25T12:00:00+00:00",
    "thread_id": None,
}


def _tools(client: httpx.AsyncClient) -> dict[str, Any]:
    return {spec.name: spec for spec in make_email_tools(client)}


@pytest.mark.asyncio
async def test_send_email_returns_typed_result() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            route = router.post("/email/send").mock(
                return_value=httpx.Response(200, json={"ok": True, "email": EMAIL_PAYLOAD})
            )
            result = await _tools(client)["email_send"].handler(  # type: ignore[attr-defined]
                EmailSendParams(
                    sender="alice@example.com",
                    recipients=["bob@example.com"],
                    subject="hello",
                    body="hi there",
                ),
            )
    sent = route.calls.last.request.read()
    assert b"thread_id" not in sent
    assert result.email.id == "em_000001"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_send_email_400_becomes_tool_error_result() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            router.post("/email/send").mock(
                return_value=httpx.Response(
                    400, json={"detail": {"ok": False, "error": "empty_subject"}}
                )
            )
            result = await _tools(client)["email_send"].handler(  # type: ignore[attr-defined]
                EmailSendParams(
                    sender="alice@example.com",
                    recipients=["bob@example.com"],
                    subject="hi",
                    body="x",
                ),
            )
    assert isinstance(result, ToolErrorResult)
    assert result.error == "empty_subject"
