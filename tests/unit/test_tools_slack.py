from typing import Any

import httpx
import pytest
import respx

from workplace_agent.tools.registry import ToolErrorResult
from workplace_agent.tools.slack import (
    SlackSendMessageParams,
    make_slack_tools,
)

BASE_URL = "http://mock"


def _tools(client: httpx.AsyncClient) -> dict[str, Any]:
    return {spec.name: spec for spec in make_slack_tools(client)}


@pytest.mark.asyncio
async def test_send_message_returns_typed_result() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL, assert_all_called=True) as router:
            route = router.post("/slack/send_message").mock(
                return_value=httpx.Response(
                    200,
                    json={"ok": True, "ts": "1.000001", "channel": "C00001"},
                )
            )
            tools = _tools(client)
            send = tools["slack_send_message"]
            result = await send.handler(  # type: ignore[attr-defined]
                SlackSendMessageParams(channel="#engineering", text="hi", user="U001"),
            )

    assert route.called
    body = route.calls.last.request.read()
    assert b"#engineering" in body
    assert b"hi" in body
    assert b"U001" in body
    assert result.ok is True  # type: ignore[union-attr]
    assert result.ts == "1.000001"  # type: ignore[union-attr]
    assert result.channel == "C00001"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_send_message_omits_none_user() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL, assert_all_called=True) as router:
            route = router.post("/slack/send_message").mock(
                return_value=httpx.Response(
                    200, json={"ok": True, "ts": "1.0", "channel": "C00001"}
                )
            )
            tools = _tools(client)
            await tools["slack_send_message"].handler(  # type: ignore[attr-defined]
                SlackSendMessageParams(channel="#x", text="hi"),
            )
    sent = route.calls.last.request.read()
    assert b"user" not in sent


@pytest.mark.asyncio
async def test_send_message_translates_429_with_top_level_error() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            router.post("/slack/send_message").mock(
                return_value=httpx.Response(429, json={"ok": False, "error": "rate_limited"})
            )
            tools = _tools(client)
            result = await tools["slack_send_message"].handler(  # type: ignore[attr-defined]
                SlackSendMessageParams(channel="#x", text="hi"),
            )
    assert isinstance(result, ToolErrorResult)
    assert result.error == "rate_limited"
    assert result.detail["status_code"] == 429
