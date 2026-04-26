import socket
import threading
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from workplace_agent.config import ErrorInjectionRule
from workplace_agent.mock_api.errors import (
    DEFAULT_TIMEOUT_HANG_SECONDS,
    add_error_injection,
)
from workplace_agent.mock_api.routers import calendar as calendar_router
from workplace_agent.mock_api.routers import email as email_router
from workplace_agent.mock_api.routers import jira as jira_router
from workplace_agent.mock_api.routers import slack as slack_router
from workplace_agent.mock_api.state import DEFAULT_SEED_DIR, AppState


def create_app(
    state: AppState,
    *,
    seed_dir: Path | None = DEFAULT_SEED_DIR,
    title: str = "workplace-agent mock API",
    error_rules: Iterable[ErrorInjectionRule] | None = None,
    timeout_hang_seconds: float = DEFAULT_TIMEOUT_HANG_SECONDS,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if seed_dir is not None:
            state.load_seeds(seed_dir)
        yield

    app = FastAPI(title=title, lifespan=lifespan)
    app.state.app_state = state

    @app.get("/health", name="health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "seeded": state.seeded,
            "channels": len(state.slack.list_channels()),
            "issues": len(state.jira.list_issues()),
            "events": len(state.calendar.list_events()),
            "emails": len(state.email.list_emails()),
        }

    app.include_router(slack_router.router)
    app.include_router(jira_router.router)
    app.include_router(calendar_router.router)
    app.include_router(email_router.router)

    add_error_injection(app, error_rules, timeout_hang_seconds=timeout_hang_seconds)

    return app


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerHandle:
    def __init__(
        self,
        server: uvicorn.Server,
        thread: threading.Thread,
        host: str,
        port: int,
    ) -> None:
        self._server = server
        self._thread = thread
        self.host = host
        self.port = port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                return
            if not self._thread.is_alive():
                raise RuntimeError("uvicorn thread exited before server was ready")
            time.sleep(0.02)
        raise TimeoutError(f"uvicorn did not start within {timeout}s")

    def shutdown(self, timeout: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout)


def run_in_thread(
    app: FastAPI,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    log_level: str = "warning",
    ready_timeout: float = 10.0,
) -> ServerHandle:
    bind_port = port or find_free_port()
    config = uvicorn.Config(
        app,
        host=host,
        port=bind_port,
        log_level=log_level,
        lifespan="on",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="mock-api-uvicorn")
    thread.start()
    handle = ServerHandle(server, thread, host=host, port=bind_port)
    try:
        handle.wait_until_ready(timeout=ready_timeout)
    except Exception:
        handle.shutdown()
        raise
    return handle
