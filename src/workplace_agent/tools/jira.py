from datetime import datetime
from typing import Any, Self

import httpx
from pydantic import BaseModel, Field, model_validator

from workplace_agent.tools._http import http_error
from workplace_agent.tools.registry import ToolErrorResult, ToolRegistry, ToolSpec


class JiraIssueModel(BaseModel):
    key: str
    project: str
    summary: str
    description: str = ""
    status: str = "Open"
    assignee: str | None = None
    created_at: datetime
    updated_at: datetime


class JiraCreateIssueParams(BaseModel):
    project: str = Field(..., description="Project key prefix, e.g. 'ENG'.")
    summary: str = Field(..., description="Short issue title.")
    description: str = Field(default="", description="Optional longer description.")
    assignee: str | None = Field(
        default=None, description="Assignee handle, e.g. 'jdoe'. Optional at creation."
    )
    status: str | None = Field(
        default=None,
        description="Optional initial status; defaults to 'Open' when omitted.",
    )


class JiraCreateIssueResult(BaseModel):
    ok: bool = True
    issue: JiraIssueModel


class JiraGetIssueParams(BaseModel):
    key: str = Field(..., description="Full issue key, e.g. 'ENG-123'.")


class JiraGetIssueResult(BaseModel):
    ok: bool = True
    issue: JiraIssueModel


class JiraTransitionIssueParams(BaseModel):
    key: str = Field(..., description="Issue key to update, e.g. 'ENG-123'.")
    status: str | None = Field(
        default=None,
        description="New workflow status (e.g. 'In Progress', 'Done'). Optional if assignee is set.",
    )
    assignee: str | None = Field(
        default=None,
        description="New assignee handle. Optional if status is set.",
    )

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> Self:
        if self.status is None and self.assignee is None:
            raise ValueError("at least one of 'status' or 'assignee' must be provided")
        return self


class JiraTransitionIssueResult(BaseModel):
    ok: bool = True
    issue: JiraIssueModel


def make_jira_tools(client: httpx.AsyncClient) -> list[ToolSpec[Any, Any]]:
    async def create_issue(
        params: JiraCreateIssueParams,
    ) -> JiraCreateIssueResult | ToolErrorResult:
        response = await client.post(
            "/jira/issues",
            json=params.model_dump(exclude_none=True),
        )
        if response.status_code != 200:
            return http_error(response)
        return JiraCreateIssueResult.model_validate(response.json())

    async def get_issue(
        params: JiraGetIssueParams,
    ) -> JiraGetIssueResult | ToolErrorResult:
        response = await client.get(f"/jira/issues/{params.key}")
        if response.status_code != 200:
            return http_error(response)
        return JiraGetIssueResult.model_validate(response.json())

    async def transition_issue(
        params: JiraTransitionIssueParams,
    ) -> JiraTransitionIssueResult | ToolErrorResult:
        body = params.model_dump(exclude={"key"}, exclude_none=True)
        response = await client.post(
            f"/jira/issues/{params.key}/transition",
            json=body,
        )
        if response.status_code != 200:
            return http_error(response)
        return JiraTransitionIssueResult.model_validate(response.json())

    return [
        ToolSpec(
            name="jira_create_issue",
            description=(
                "Create a new Jira issue in the given project. Returns the assigned issue key "
                "(e.g. 'ENG-12'). Side effect: persists a new ticket."
            ),
            params_model=JiraCreateIssueParams,
            result_model=JiraCreateIssueResult,
            handler=create_issue,
        ),
        ToolSpec(
            name="jira_get_issue",
            description="Fetch a Jira issue by key. Returns 'issue_not_found' when the key is unknown.",
            params_model=JiraGetIssueParams,
            result_model=JiraGetIssueResult,
            handler=get_issue,
        ),
        ToolSpec(
            name="jira_transition_issue",
            description=(
                "Change a Jira issue's status and/or assignee. At least one of 'status' or "
                "'assignee' must be provided. Side effect: updates the ticket."
            ),
            params_model=JiraTransitionIssueParams,
            result_model=JiraTransitionIssueResult,
            handler=transition_issue,
        ),
    ]


def register_jira_tools(registry: ToolRegistry, client: httpx.AsyncClient) -> None:
    for spec in make_jira_tools(client):
        registry.register(spec)
