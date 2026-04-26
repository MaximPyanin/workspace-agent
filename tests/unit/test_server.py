import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from workplace_agent.config import ErrorInjectionRule
from workplace_agent.mock_api.server import create_app, find_free_port, run_in_thread
from workplace_agent.mock_api.state import AppState


def _fixed_clock():
    fixed = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    return lambda: fixed


async def test_health_endpoint_returns_ok_and_counts() -> None:
    state = AppState.empty(current_time=_fixed_clock())
    state.slack.add_channel("#general")
    app = create_app(state, seed_dir=None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["channels"] == 1
    assert body["issues"] == 0
    assert body["events"] == 0
    assert body["emails"] == 0


def test_find_free_port_returns_usable_port() -> None:
    port = find_free_port()
    assert 1024 < port < 65536


def test_run_in_thread_serves_health_and_shuts_down() -> None:
    state = AppState.empty(current_time=_fixed_clock())
    app = create_app(state, seed_dir=None)

    handle = run_in_thread(app)
    try:
        resp = httpx.get(f"{handle.base_url}/health", timeout=5.0)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
    finally:
        handle.shutdown()

    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{handle.base_url}/health", timeout=0.5)


async def test_create_app_with_error_rules_injects_errors_on_health() -> None:
    state = AppState.empty(current_time=_fixed_clock())
    rules = [ErrorInjectionRule(service="health", action="health", error_type="rate_limit")]
    app = create_app(state, seed_dir=None, error_rules=rules)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 429
    assert resp.json() == {"ok": False, "error": "rate_limited"}


def test_run_in_thread_triggers_lifespan_seeding(tmp_path: Path) -> None:
    (tmp_path / "slack.json").write_text(
        json.dumps({"channels": ["engineering", "product"]}), encoding="utf-8"
    )
    state = AppState.empty(current_time=_fixed_clock())
    app = create_app(state, seed_dir=tmp_path)

    handle = run_in_thread(app)
    try:
        resp = httpx.get(f"{handle.base_url}/health", timeout=5.0)
    finally:
        handle.shutdown()

    body = resp.json()
    assert body["channels"] == 2
    assert state.seeded is True
