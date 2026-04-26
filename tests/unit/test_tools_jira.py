from typing import Any

import httpx
import pytest
import respx

from workplace_agent.tools.jira import (
    JiraCreateIssueParams,
    JiraGetIssueParams,
    make_jira_tools,
)
from workplace_agent.tools.registry import ToolErrorResult

BASE_URL = "http://mock"

ISSUE_PAYLOAD = {
    "key": "ENG-1",
    "project": "ENG",
    "summary": "ship it",
    "description": "",
    "status": "Open",
    "assignee": None,
    "created_at": "2026-04-25T12:00:00+00:00",
    "updated_at": "2026-04-25T12:00:00+00:00",
}


def _tools(client: httpx.AsyncClient) -> dict[str, Any]:
    return {spec.name: spec for spec in make_jira_tools(client)}


@pytest.mark.asyncio
async def test_create_issue_returns_typed_result() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            route = router.post("/jira/issues").mock(
                return_value=httpx.Response(200, json={"ok": True, "issue": ISSUE_PAYLOAD})
            )
            result = await _tools(client)["jira_create_issue"].handler(  # type: ignore[attr-defined]
                JiraCreateIssueParams(project="ENG", summary="ship it"),
            )
    assert route.called
    sent = route.calls.last.request.read()
    assert b'"project":"ENG"' in sent
    assert b"status" not in sent
    assert result.issue.key == "ENG-1"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_get_issue_404_becomes_tool_error_result() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        with respx.mock(base_url=BASE_URL) as router:
            router.get("/jira/issues/ENG-99").mock(
                return_value=httpx.Response(
                    404, json={"detail": {"ok": False, "error": "issue_not_found"}}
                )
            )
            result = await _tools(client)["jira_get_issue"].handler(  # type: ignore[attr-defined]
                JiraGetIssueParams(key="ENG-99"),
            )
    assert isinstance(result, ToolErrorResult)
    assert result.error == "issue_not_found"
