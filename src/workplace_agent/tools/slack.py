from typing import Any

import httpx
from pydantic import BaseModel, Field

from workplace_agent.tools._http import http_error
from workplace_agent.tools.registry import ToolErrorResult, ToolRegistry, ToolSpec


class SlackSendMessageParams(BaseModel):
    channel: str = Field(
        ...,
        description=(
            "Slack channel name (prefer the '#name' form) or channel id; "
            "an unknown '#name' channel is auto-created."
        ),
    )
    text: str = Field(..., description="Message body. Must be non-empty.")
    user: str | None = Field(default=None, description="Optional sender user id (e.g. 'U001').")


class SlackSendMessageResult(BaseModel):
    ok: bool = True
    ts: str
    channel: str


class SlackListChannelsParams(BaseModel):
    pass


class SlackChannel(BaseModel):
    id: str
    name: str


class SlackListChannelsResult(BaseModel):
    ok: bool = True
    channels: list[SlackChannel]


class SlackSearchMessagesParams(BaseModel):
    query: str = Field(..., min_length=1, description="Substring matched in message text.")


class SlackSearchHit(BaseModel):
    ts: str
    channel: str
    channel_name: str | None = None
    user: str | None = None
    text: str


class SlackSearchMessagesResult(BaseModel):
    ok: bool = True
    messages: list[SlackSearchHit]


def make_slack_tools(client: httpx.AsyncClient) -> list[ToolSpec[Any, Any]]:
    async def send_message(
        params: SlackSendMessageParams,
    ) -> SlackSendMessageResult | ToolErrorResult:
        response = await client.post(
            "/slack/send_message",
            json=params.model_dump(exclude_none=True),
        )
        if response.status_code != 200:
            return http_error(response)
        return SlackSendMessageResult.model_validate(response.json())

    async def list_channels(
        _: SlackListChannelsParams,
    ) -> SlackListChannelsResult | ToolErrorResult:
        response = await client.get("/slack/channels")
        if response.status_code != 200:
            return http_error(response)
        return SlackListChannelsResult.model_validate(response.json())

    async def search_messages(
        params: SlackSearchMessagesParams,
    ) -> SlackSearchMessagesResult | ToolErrorResult:
        response = await client.get("/slack/search", params={"query": params.query})
        if response.status_code != 200:
            return http_error(response)
        return SlackSearchMessagesResult.model_validate(response.json())

    return [
        ToolSpec(
            name="slack_send_message",
            description=(
                "Send a message to a Slack channel. The 'channel' argument should be a name "
                "starting with '#' (e.g. '#engineering') or an existing channel id. "
                "Side effect: posts a new message visible to anyone in the channel."
            ),
            params_model=SlackSendMessageParams,
            result_model=SlackSendMessageResult,
            handler=send_message,
        ),
        ToolSpec(
            name="slack_list_channels",
            description="List all Slack channels currently known to the workspace.",
            params_model=SlackListChannelsParams,
            result_model=SlackListChannelsResult,
            handler=list_channels,
        ),
        ToolSpec(
            name="slack_search_messages",
            description=(
                "Case-insensitive substring search across all Slack messages. "
                "Returns matching messages with channel id and channel name."
            ),
            params_model=SlackSearchMessagesParams,
            result_model=SlackSearchMessagesResult,
            handler=search_messages,
        ),
    ]


def register_slack_tools(registry: ToolRegistry, client: httpx.AsyncClient) -> None:
    for spec in make_slack_tools(client):
        registry.register(spec)
