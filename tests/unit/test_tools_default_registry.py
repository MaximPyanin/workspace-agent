import httpx

from workplace_agent.tools import default_registry


async def test_default_registry_registers_all_eleven_tools() -> None:
    async with httpx.AsyncClient(base_url="http://mock") as client:
        registry = default_registry(client)
    expected = {
        "slack_send_message",
        "slack_list_channels",
        "slack_search_messages",
        "jira_create_issue",
        "jira_get_issue",
        "jira_transition_issue",
        "calendar_create_event",
        "calendar_list_events",
        "calendar_find_free_slot",
        "email_send",
        "email_search",
    }
    assert set(registry.names()) == expected
    defs = registry.list_definitions()
    assert len(defs) == 11
    assert all(d.strict is False for d in defs)
    for d in defs:
        schema = d.input_schema
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
