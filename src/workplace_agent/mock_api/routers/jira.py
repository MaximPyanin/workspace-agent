from fastapi import APIRouter, HTTPException, Request

from workplace_agent.mock_api.schemas.jira import (
    CreateIssueRequest,
    CreateIssueResponse,
    GetIssueResponse,
    Issue,
    TransitionRequest,
    TransitionResponse,
)
from workplace_agent.mock_api.state import AppState, JiraIssue


def _state(request: Request) -> AppState:
    state = request.app.state.app_state
    assert isinstance(state, AppState)
    return state


def _to_schema(issue: JiraIssue) -> Issue:
    return Issue(
        key=issue.key,
        project=issue.project,
        summary=issue.summary,
        description=issue.description,
        status=issue.status,
        assignee=issue.assignee,
        created_at=issue.created_at,
        updated_at=issue.updated_at,
    )


router = APIRouter(prefix="/jira", tags=["jira"])


@router.post("/issues", name="create_issue", response_model=CreateIssueResponse)
async def create_issue(payload: CreateIssueRequest, request: Request) -> CreateIssueResponse:
    state = _state(request)
    project = payload.project.strip()
    if not project:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "empty_project"})
    summary = payload.summary.strip()
    if not summary:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "empty_summary"})
    status = payload.status if payload.status is not None else "Open"
    if status not in state.jira.allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "error": "invalid_status",
                "allowed": list(state.jira.allowed_statuses),
            },
        )
    issue = state.jira.create_issue(
        project=project,
        summary=summary,
        description=payload.description,
        assignee=payload.assignee,
        status=status,
    )
    return CreateIssueResponse(ok=True, issue=_to_schema(issue))


@router.get("/issues/{key}", name="get_issue", response_model=GetIssueResponse)
async def get_issue(key: str, request: Request) -> GetIssueResponse:
    state = _state(request)
    issue = state.jira.get_issue(key)
    if issue is None:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "issue_not_found", "key": key},
        )
    return GetIssueResponse(ok=True, issue=_to_schema(issue))


@router.post(
    "/issues/{key}/transition",
    name="transition_issue",
    response_model=TransitionResponse,
)
async def transition_issue(
    key: str, payload: TransitionRequest, request: Request
) -> TransitionResponse:
    state = _state(request)
    if payload.status is not None and payload.status not in state.jira.allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "error": "invalid_status",
                "allowed": list(state.jira.allowed_statuses),
            },
        )
    try:
        issue = state.jira.transition_issue(
            key=key,
            status=payload.status,
            assignee=payload.assignee,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "issue_not_found", "key": key},
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": str(exc)},
        ) from None
    return TransitionResponse(ok=True, issue=_to_schema(issue))
