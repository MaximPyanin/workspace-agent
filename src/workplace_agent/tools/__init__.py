import httpx

from workplace_agent.tools.calendar import register_calendar_tools
from workplace_agent.tools.email import register_email_tools
from workplace_agent.tools.jira import register_jira_tools
from workplace_agent.tools.registry import ToolErrorResult, ToolRegistry, ToolSpec
from workplace_agent.tools.slack import register_slack_tools


def default_registry(http_client: httpx.AsyncClient) -> ToolRegistry:
    registry = ToolRegistry()
    register_slack_tools(registry, http_client)
    register_jira_tools(registry, http_client)
    register_calendar_tools(registry, http_client)
    register_email_tools(registry, http_client)
    return registry


__all__ = ["ToolErrorResult", "ToolRegistry", "ToolSpec", "default_registry"]
