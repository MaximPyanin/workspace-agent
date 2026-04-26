from typing import Any

import httpx

from workplace_agent.tools.registry import ToolErrorResult


def http_error(response: httpx.Response) -> ToolErrorResult:
    detail: dict[str, Any] = {"status_code": response.status_code}
    try:
        body = response.json()
    except ValueError:
        body = response.text
    detail["body"] = body
    error = "http_error"
    if isinstance(body, dict):
        nested = body.get("detail")
        if isinstance(nested, dict) and "error" in nested:
            error = str(nested["error"])
        elif "error" in body:
            error = str(body["error"])
    return ToolErrorResult(error=error, detail=detail)
