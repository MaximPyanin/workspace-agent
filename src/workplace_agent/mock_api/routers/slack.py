from fastapi import APIRouter, HTTPException, Query, Request

from workplace_agent.mock_api.schemas.slack import (
    Channel,
    ListChannelsResponse,
    SearchMessage,
    SearchResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from workplace_agent.mock_api.state import AppState


def _state(request: Request) -> AppState:
    state = request.app.state.app_state
    assert isinstance(state, AppState)
    return state


router = APIRouter(prefix="/slack", tags=["slack"])


@router.post("/send_message", name="send_message", response_model=SendMessageResponse)
async def send_message(payload: SendMessageRequest, request: Request) -> SendMessageResponse:
    state = _state(request)
    text = payload.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail={"ok": False, "error": "empty_text"})

    raw = payload.channel.strip()
    if not raw:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "empty_channel"})

    if raw.startswith("#"):
        channel = state.slack.add_channel(raw)
    else:
        resolved = state.slack.resolve_channel(raw)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail={"ok": False, "error": "channel_not_found", "channel": raw},
            )
        channel = resolved

    msg = state.slack.send_message(channel=channel.id, text=text, user=payload.user)
    return SendMessageResponse(ok=True, ts=msg.ts, channel=channel.id)


@router.get("/channels", name="list_channels", response_model=ListChannelsResponse)
async def list_channels(request: Request) -> ListChannelsResponse:
    state = _state(request)
    channels = [Channel(id=c.id, name=c.name) for c in state.slack.list_channels()]
    return ListChannelsResponse(ok=True, channels=channels)


@router.get("/search", name="search_messages", response_model=SearchResponse)
async def search_messages(
    request: Request,
    query: str = Query(..., min_length=1, description="Substring to match in message text"),
) -> SearchResponse:
    state = _state(request)
    matches = state.slack.search_messages(query)
    messages: list[SearchMessage] = []
    for m in matches:
        ch = state.slack.get_channel_by_id(m.channel)
        messages.append(
            SearchMessage(
                ts=m.ts,
                channel=m.channel,
                channel_name=ch.name if ch is not None else None,
                user=m.user,
                text=m.text,
            )
        )
    return SearchResponse(ok=True, messages=messages)
