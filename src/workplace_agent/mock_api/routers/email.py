from fastapi import APIRouter, HTTPException, Query, Request

from workplace_agent.mock_api.schemas.email import (
    Email,
    SearchResponse,
    SearchResult,
    SendEmailRequest,
    SendEmailResponse,
)
from workplace_agent.mock_api.state import AppState
from workplace_agent.mock_api.state import Email as StoredEmail


def _state(request: Request) -> AppState:
    state = request.app.state.app_state
    assert isinstance(state, AppState)
    return state


def _to_email(em: StoredEmail) -> Email:
    return Email(
        id=em.id,
        sender=em.sender,
        recipients=list(em.recipients),
        subject=em.subject,
        body=em.body,
        sent_at=em.sent_at,
        thread_id=em.thread_id,
    )


def _to_search_result(em: StoredEmail) -> SearchResult:
    return SearchResult(
        id=em.id,
        sender=em.sender,
        recipients=list(em.recipients),
        subject=em.subject,
        body=em.body,
        sent_at=em.sent_at,
        thread_id=em.thread_id,
    )


router = APIRouter(prefix="/email", tags=["email"])


@router.post("/send", name="send_email", response_model=SendEmailResponse)
async def send_email(payload: SendEmailRequest, request: Request) -> SendEmailResponse:
    state = _state(request)
    subject = payload.subject.strip()
    if not subject:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "empty_subject"})
    if not payload.body or not payload.body.strip():
        raise HTTPException(status_code=400, detail={"ok": False, "error": "empty_body"})
    try:
        stored = state.email.send(
            sender=payload.sender,
            recipients=payload.recipients,
            subject=subject,
            body=payload.body,
            thread_id=payload.thread_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)}) from None
    return SendEmailResponse(ok=True, email=_to_email(stored))


@router.get("/search", name="search_email", response_model=SearchResponse)
async def search_email(
    request: Request,
    query: str = Query(..., min_length=1, description="Substring to match against subject+body"),
) -> SearchResponse:
    state = _state(request)
    matches = state.email.search(query)
    return SearchResponse(ok=True, results=[_to_search_result(em) for em in matches])
