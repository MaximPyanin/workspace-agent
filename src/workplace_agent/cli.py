import argparse
import asyncio
import time
from collections.abc import Awaitable, Callable

import httpx
from rich.console import Console

from workplace_agent.agent.loop import Agent
from workplace_agent.agent.prompts import build_system_prompt
from workplace_agent.agent.summary import print_response
from workplace_agent.config import Settings
from workplace_agent.llm.client import AnthropicClient, LLMClient
from workplace_agent.llm.fake import FakeLLM
from workplace_agent.logging_setup import configure_logging, get_logger
from workplace_agent.mock_api.server import ServerHandle, create_app, run_in_thread
from workplace_agent.mock_api.state import AppState
from workplace_agent.tools import default_registry

ReadPrompt = Callable[[], Awaitable[str]]
HealthGet = Callable[[str], httpx.Response]
Sleep = Callable[[float], None]
Monotonic = Callable[[], float]

EXIT_WORDS = frozenset({"exit", "quit", ":q"})

DEFAULT_REPL_PROMPT = "[bold cyan]>[/bold cyan] "


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="workplace-agent",
        description="Workplace AI agent with mock Slack/Jira/Calendar/Email APIs.",
    )
    parser.add_argument(
        "--mock-only",
        action="store_true",
        help="Launch only the mock API server and block until Ctrl-C.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override mock API bind host (default: from settings).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override mock API bind port (default: from settings; 0 picks a free port).",
    )
    return parser.parse_args(argv)


def wait_for_health(
    base_url: str,
    *,
    timeout: float = 10.0,
    interval: float = 0.05,
    get_fn: HealthGet | None = None,
    sleep_fn: Sleep | None = None,
    monotonic_fn: Monotonic | None = None,
) -> None:
    get = get_fn if get_fn is not None else _default_health_get
    sleep = sleep_fn if sleep_fn is not None else time.sleep
    now = monotonic_fn if monotonic_fn is not None else time.monotonic

    deadline = now() + timeout
    last_error: str | None = None
    while True:
        try:
            resp = get(f"{base_url}/health")
            if resp.status_code == 200:
                return
            last_error = f"status {resp.status_code}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if now() >= deadline:
            raise TimeoutError(
                f"mock API at {base_url} did not become healthy within {timeout}s "
                f"(last error: {last_error})"
            )
        sleep(interval)


def _default_health_get(url: str) -> httpx.Response:
    return httpx.get(url, timeout=2.0)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.agent_use_fake_llm:
        return FakeLLM()
    return AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        beta_header=settings.anthropic_beta_header,
    )


async def _default_read_prompt(console: Console) -> str:
    return await asyncio.to_thread(console.input, DEFAULT_REPL_PROMPT)


async def repl_async(
    agent: Agent,
    console: Console,
    *,
    read_prompt: ReadPrompt | None = None,
) -> None:
    reader: ReadPrompt = (
        read_prompt if read_prompt is not None else (lambda: _default_read_prompt(console))
    )
    console.print(
        "[dim]Workplace agent ready. Type 'exit', 'quit', or press Ctrl-D to leave.[/dim]"
    )
    while True:
        try:
            raw = await reader()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        prompt = raw.strip()
        if not prompt:
            continue
        if prompt.lower() in EXIT_WORDS:
            return
        try:
            response = await agent.run(prompt)
        except Exception as exc:
            console.print(f"[red]Agent error:[/red] {exc}")
            continue
        print_response(response, console=console)


def run_mock_only(
    settings: Settings,
    *,
    host: str | None = None,
    port: int | None = None,
    console: Console | None = None,
    block: Callable[[ServerHandle], None] | None = None,
) -> None:
    out = console if console is not None else Console()
    state = AppState.empty()
    app = create_app(state, error_rules=settings.mock_api_error_injection)
    handle = run_in_thread(
        app,
        host=host if host is not None else settings.mock_api_host,
        port=port if port is not None else settings.mock_api_port,
    )
    out.print(f"[green]Mock API running at {handle.base_url}[/green]")
    out.print("[dim]Press Ctrl-C to stop.[/dim]")
    try:
        if block is not None:
            block(handle)
        else:
            _block_until_interrupt()
    finally:
        handle.shutdown()
        out.print("[dim]Mock API stopped.[/dim]")


def _block_until_interrupt() -> None:
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return


async def _run_agent_session(
    settings: Settings,
    handle: ServerHandle,
    console: Console,
) -> None:
    wait_for_health(handle.base_url)
    async with httpx.AsyncClient(base_url=handle.base_url, timeout=15.0) as client:
        registry = default_registry(client)
        llm = build_llm_client(settings)
        agent = Agent(
            llm=llm,
            registry=registry,
            system_prompt=build_system_prompt(),
            max_steps=settings.agent_max_steps,
        )
        await repl_async(agent, console)


def run_agent(
    settings: Settings,
    *,
    host: str | None = None,
    port: int | None = None,
    console: Console | None = None,
) -> None:
    out = console if console is not None else Console()
    state = AppState.empty()
    app = create_app(state, error_rules=settings.mock_api_error_injection)
    handle = run_in_thread(
        app,
        host=host if host is not None else settings.mock_api_host,
        port=port if port is not None else settings.mock_api_port,
    )
    log = get_logger("cli")
    log.info("mock_api_started", base_url=handle.base_url)
    try:
        asyncio.run(_run_agent_session(settings, handle, out))
    finally:
        handle.shutdown()
        log.info("mock_api_stopped")


def main(argv: list[str] | None = None, *, settings: Settings | None = None) -> None:
    args = parse_args(argv)
    cfg = settings if settings is not None else _load_settings()
    configure_logging(level=cfg.log_level, log_file=cfg.log_file)

    if args.mock_only:
        run_mock_only(cfg, host=args.host, port=args.port)
        return
    run_agent(cfg, host=args.host, port=args.port)


def _load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        raise SystemExit(f"Failed to load settings: {exc}") from exc
