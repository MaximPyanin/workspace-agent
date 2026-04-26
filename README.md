# workplace-agent

A CLI-driven AI agent that takes plain-English prompts, picks the right tools, calls
mocked workplace APIs (Slack / Jira / Calendar / Email), and reports what it did.

The agent runs a hand-rolled ReAct-style loop on top of Anthropic Claude with native
tool use and structured outputs. The mock workplace lives in a local FastAPI server
spawned automatically on startup; everything is in-memory and reseeded from JSON on
each launch.

## What it does

- accepts a natural-language prompt (`> Send "build is green" to #engineering`)
- picks tools, invokes them through the registry, observes results, and loops until
  done — or terminates with a clarifying question if the request is ambiguous
- emits a structured terminal response (`done` vs `needs_clarification`) plus an
  action log of every tool call

## Prerequisites

- Python 3.13
- `uv` (recommended) — install from <https://docs.astral.sh/uv/>
- An Anthropic API key (only required for live runs; tests and the FakeLLM path do
  not need one)

## Install

```bash
uv sync
```

This creates `.venv/`, installs runtime + dev dependencies, and locks them in
`uv.lock`.

If you prefer pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

If you skip activating the venv, prefix every command below with `uv run`
(e.g. `uv run python agent.py`, `uv run pytest`).

## Configure

Copy `.env.example` to `.env` and set the Anthropic key:

```bash
cp .env.example .env
# edit .env, set ANTHROPIC_API_KEY=sk-...
```

Environment variables:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(empty)_ | required unless `AGENT_USE_FAKE_LLM=1` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | any model that supports structured outputs |
| `ANTHROPIC_BETA_HEADER` | `structured-outputs-2025-11-13` | sent on every request |
| `MOCK_API_HOST` | `127.0.0.1` | bind host for the in-process mock server |
| `MOCK_API_PORT` | `8765` | use `0` to pick a free port automatically |
| `MOCK_API_ERROR_INJECTION` | _(empty)_ | comma-separated rules, e.g. `slack:send_message:rate_limit,jira:get_issue:not_found` |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose structured logs |
| `LOG_FILE` | _(empty)_ | optional path; logs also stream to stderr |
| `AGENT_MAX_STEPS` | `12` | hard cap on ReAct iterations |
| `LLM_TIMEOUT_SECONDS` | `30` | per-request timeout for Anthropic calls |
| `LLM_MAX_RETRIES` | `3` | retries for transient API errors (rate limit, 5xx, connection) |
| `AGENT_USE_FAKE_LLM` | `0` | when `1`, the API-key requirement is dropped so `--mock-only` runs offline; not intended for the live REPL (it would build a FakeLLM with no scripted responses) |

## Run

```bash
uv run python agent.py
```

(or `source .venv/bin/activate && python agent.py` if you prefer to activate the venv).

This:

1. loads settings from `.env`,
2. starts the FastAPI mock server in a background thread (waits for `/health`),
3. opens a `rich`-powered REPL — type a prompt, get a structured summary,
4. on `Ctrl-D` / `exit` / `quit` it stops the server and closes resources.

To launch only the mock API (e.g. for `curl` exploration or Swagger):

```bash
uv run python agent.py --mock-only --port 8765
```

Then open <http://127.0.0.1:8765/docs> for the interactive Swagger UI exposing
all 11 mock endpoints (request/response schemas, "Try it out" buttons).
The raw OpenAPI spec is at `/openapi.json`.

CLI flags (apply to both modes):

- `--mock-only` — start only the FastAPI mock server, then block until Ctrl-C.
- `--host HOST` — override `MOCK_API_HOST` for this run.
- `--port PORT` — override `MOCK_API_PORT`; `0` picks a free port automatically.

## Mock API surface

Eleven tools across four services, mapped 1:1 onto FastAPI routes:

| Tool | Route |
|---|---|
| `slack_send_message` | `POST /slack/send_message` |
| `slack_list_channels` | `GET /slack/channels` |
| `slack_search_messages` | `GET /slack/search?query=` |
| `jira_create_issue` | `POST /jira/issues` |
| `jira_get_issue` | `GET /jira/issues/{key}` |
| `jira_transition_issue` | `POST /jira/issues/{key}/transition` |
| `calendar_create_event` | `POST /calendar/events` |
| `calendar_list_events` | `GET /calendar/events?from=&to=` |
| `calendar_find_free_slot` | `POST /calendar/find_free_slot` |
| `email_send` | `POST /email/send` |
| `email_search` | `GET /email/search?query=` |

Seed data (3 channels, 2 users, 2 Jira tickets, 3 calendar events, 2 emails)
lives in `src/workplace_agent/mock_api/seeds/`.

## Architecture

```
+-------------------+
|     CLI (rich)    |  agent.py -> workplace_agent.cli:main
+---------+---------+
          |
          v
+-------------------+      +-------------------+
|    Agent Loop     |<---->|    LLM Client     |---> Anthropic API
|  (ReAct)          |      |  (retry/timeout)  |
+---------+---------+      +-------------------+
          |
          v
+-------------------+
|  Tool Registry    |  one Pydantic-typed handler per tool
+---------+---------+
          |
          v  (httpx, in-process)
+-------------------+
|  Mock API server  |  FastAPI, sub-routers per service
|  (background      |  /slack/* /jira/* /calendar/* /email/*
|   uvicorn thread) |  in-memory stores, error injection middleware
+-------------------+
```

Source layout under `src/workplace_agent/`:

- `cli.py` — REPL, argparse, server lifecycle
- `config.py` — `pydantic-settings` env loader, error-injection rule parser
- `logging_setup.py` — `structlog` configuration
- `agent/loop.py` — ReAct loop, terminates with `messages.parse(AgentFinalResponse)`
- `agent/prompts.py` — system prompt + `build_system_prompt(now=...)` helper that
  injects today's date so date-relative requests resolve to the actual week
- `agent/summary.py` — `rich` rendering of the final response and action log
- `llm/client.py` — `LLMClient` protocol, `AnthropicClient` (retry/backoff), `FakeLLM`
- `llm/agent_response.py` — the `AgentFinalResponse` Pydantic schema enforced on the
  terminal turn
- `tools/registry.py` — `ToolSpec`, dispatch, structured `ToolErrorResult`
- `tools/{slack,jira,calendar,email}.py` — typed params/results + handlers
- `mock_api/server.py` — `create_app(state, error_rules=...)`, `run_in_thread()`
- `mock_api/state.py` — in-memory stores with deterministic ID generation and
  injectable `current_time`
- `mock_api/errors.py` — middleware that turns config rules into `429`/`404`/timeout
- `mock_api/routers/*` — FastAPI routers per service
- `mock_api/seeds/*.json` — seed data loaded at app startup

## Demo scenarios

The four demo scenarios from the brief are exercised by the integration suite in
`tests/integration/test_scenarios.py` with a scripted FakeLLM. They illustrate the
intended behaviour end-to-end:

- **A — single-tool:** `Send "Build is green" to #engineering`
  -> `slack_send_message` -> message visible in `/slack/search`.
- **B — multi-step:** `QA found a regression in the login button — file a bug, assign
  alice, mark it in progress, and tell #engineering`
  -> `jira_create_issue` (`ENG-3`), `jira_transition_issue` (`In Progress`, alice),
  `slack_send_message` referencing `ENG-3`.
- **C — search + aggregate:** `Find me a 30-minute slot on Monday during business hours`
  -> `calendar_list_events` then `calendar_find_free_slot` returning a non-overlapping
  slot.
- **D — ambiguous:** `Send an update`
  -> agent terminates with `kind="needs_clarification"` and a single concrete
  question; no tool calls are made.

## Tests

```bash
uv run pytest                                                # full suite
uv run pytest --cov=src/workplace_agent --cov-report=term-missing  # with coverage
uv run pytest tests/integration/test_scenarios.py            # demo scenarios only
```

Coverage is 89% overall.

Lint and typecheck:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

The LLM is always mocked in tests via `FakeLLM`; no test ever hits Anthropic.

## Error injection

Set `MOCK_API_ERROR_INJECTION` to verify the agent surfaces recoverable failures:

```bash
MOCK_API_ERROR_INJECTION=slack:send_message:rate_limit uv run python agent.py
```

The middleware turns matching requests into realistic `429 {"ok": false, "error":
"rate_limited"}` responses; the tool result comes back with `is_error=True` and the
agent's structured summary reflects what happened. See
`test_error_injection_surfaces_recoverable_error_in_summary` for a worked example.

## Notes and limitations

- State is in-memory only — restarting the server reseeds from the JSON files.
- The mock server runs on a background thread inside the same Python process; there
  is no multi-process or hot-reload setup.
- LLM responses are not streamed; the CLI prints the full structured summary once the
  loop ends.
- No real Slack/Jira/Calendar/Email integration. Every tool talks to the local mock
  via `httpx`.
- Out of scope by design: web UI, multi-user concurrency, Docker, persistent storage,
  production-grade auth.
- `ty` is currently a beta type checker; see `DECISIONS.md` for the rationale.
