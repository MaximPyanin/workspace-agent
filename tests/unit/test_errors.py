from collections.abc import Iterable
from typing import cast

import httpx
import pytest
from fastapi import FastAPI

from workplace_agent.config import ErrorInjectionRule
from workplace_agent.mock_api import errors as errors_module
from workplace_agent.mock_api.errors import (
    ErrorInjectionMiddleware,
    add_error_injection,
)


def _build_app(
    rules: Iterable[ErrorInjectionRule] | None = None,
    *,
    timeout_hang_seconds: float = 0.3,
) -> FastAPI:
    app = FastAPI()

    @app.post("/slack/send_message", name="send_message")
    async def slack_send_message() -> dict[str, object]:
        return {"ok": True, "ts": "1.0", "channel": "C1"}

    @app.get("/slack/channels", name="list_channels")
    async def slack_list_channels() -> dict[str, object]:
        return {"ok": True, "channels": []}

    @app.get("/jira/issues/{key}", name="get_issue")
    async def jira_get_issue(key: str) -> dict[str, object]:
        return {"ok": True, "key": key}

    add_error_injection(app, rules, timeout_hang_seconds=timeout_hang_seconds)
    return app


def _client(app: FastAPI, *, timeout: float = 5.0) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test", timeout=timeout)


async def test_no_rules_is_passthrough() -> None:
    app = _build_app(rules=None)
    async with _client(app) as client:
        resp = await client.post("/slack/send_message")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_rate_limit_per_service_matches_all_actions() -> None:
    rules = [ErrorInjectionRule(service="slack", action=None, error_type="rate_limit")]
    app = _build_app(rules)
    async with _client(app) as client:
        send = await client.post("/slack/send_message")
        channels = await client.get("/slack/channels")
    for resp in (send, channels):
        assert resp.status_code == 429
        assert resp.json() == {"ok": False, "error": "rate_limited"}


async def test_rate_limit_per_action_only_targets_that_action() -> None:
    rules = [
        ErrorInjectionRule(service="slack", action="send_message", error_type="rate_limit"),
    ]
    app = _build_app(rules)
    async with _client(app) as client:
        send = await client.post("/slack/send_message")
        channels = await client.get("/slack/channels")
    assert send.status_code == 429
    assert send.json() == {"ok": False, "error": "rate_limited"}
    assert channels.status_code == 200


async def test_not_found_injection() -> None:
    rules = [ErrorInjectionRule(service="jira", action="get_issue", error_type="not_found")]
    app = _build_app(rules)
    async with _client(app) as client:
        resp = await client.get("/jira/issues/ENG-1")
    assert resp.status_code == 404
    assert resp.json() == {"ok": False, "error": "not_found"}


async def test_timeout_injection_sleeps_for_configured_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(errors_module.asyncio, "sleep", fake_sleep)

    rules = [
        ErrorInjectionRule(service="slack", action="send_message", error_type="timeout"),
    ]
    app = _build_app(rules, timeout_hang_seconds=42.0)
    async with _client(app) as client:
        resp = await client.post("/slack/send_message")
    assert sleep_calls == [42.0]
    assert resp.status_code == 504
    assert resp.json() == {"ok": False, "error": "timeout"}


def test_middleware_rejects_non_rule_objects() -> None:
    app = FastAPI()
    bad_rules = cast(
        "list[ErrorInjectionRule]",
        [{"service": "slack", "error_type": "rate_limit"}],
    )
    with pytest.raises(TypeError, match="ErrorInjectionRule"):
        ErrorInjectionMiddleware(app, rules=bad_rules)


def test_middleware_rejects_invalid_timeout_hang_seconds() -> None:
    app = FastAPI()
    with pytest.raises(ValueError, match="timeout_hang_seconds"):
        ErrorInjectionMiddleware(app, rules=[], timeout_hang_seconds=0)


def test_add_error_injection_installs_middleware() -> None:
    app = FastAPI()
    before = len(app.user_middleware)
    add_error_injection(
        app,
        [ErrorInjectionRule(service="slack", action=None, error_type="rate_limit")],
    )
    assert len(app.user_middleware) == before + 1
