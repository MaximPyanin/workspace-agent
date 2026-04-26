from pydantic import BaseModel, Field


class Channel(BaseModel):
    id: str
    name: str


class SearchMessage(BaseModel):
    ts: str
    channel: str
    channel_name: str | None = None
    user: str | None = None
    text: str


class SendMessageRequest(BaseModel):
    channel: str = Field(..., description="Channel name (prefer '#name') or channel id")
    text: str = Field(..., description="Message body")
    user: str | None = None


class SendMessageResponse(BaseModel):
    ok: bool = True
    ts: str
    channel: str


class ListChannelsResponse(BaseModel):
    ok: bool = True
    channels: list[Channel]


class SearchResponse(BaseModel):
    ok: bool = True
    messages: list[SearchMessage]
