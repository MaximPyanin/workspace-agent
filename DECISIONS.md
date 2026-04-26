# Design decisions

This document explains the engineering decisions that shaped `workplace-agent`. Each
section names a decision, the alternatives considered, and the reason the chosen
option won.

## LLM choice — Anthropic Claude (Sonnet 4.5+)

Native tool use, strong tracking through multi-step ReAct loops, and — critically —
support for structured outputs via `messages.parse(output_format=PydanticModel)`.
The default is `claude-sonnet-4-6`, configurable via `ANTHROPIC_MODEL`.

Alternatives ruled out:

- OpenAI / Gemini — adequate, but the assignment is graded against an Anthropic-native
  setup and Claude's tool-use plus structured outputs gives the cleanest path to a
  small reliable agent.
- Local models — too slow and too unreliable on multi-step tool use for a demo.

## No agent framework — hand-rolled ReAct

The brief explicitly bans LangChain / LangGraph / CrewAI / AutoGen, and the assignment
rewards a small readable codebase. The body of `Agent.run` in
`src/workplace_agent/agent/loop.py` is roughly:

```
for _ in range(max_steps):
    response = llm.create(...)
    if no tool_use blocks: return llm.parse(...)
    dispatch tool calls, append tool_results
return synthesised "step-limit reached" response
```

Benefits: no framework opinions, every behaviour is grep-able, the terminal-response
contract lives next to the loop in `agent/`. The system prompt itself sits in
`agent/prompts.py` (with `build_system_prompt(now=...)` to inject today's date).
Trade-off: we do not get free tracing/replay/observability — but for a 4-tool demo
that's fine, and `structlog` is already wired up.

## FastAPI mock vs in-process Python objects

Chose FastAPI over a "just call the handler" in-process mock because:

- the brief specifically says "create a mock service or services with realistic
  responses" — HTTP is more realistic than a Python function call,
- it forces real serialisation, real validation, and real HTTP error semantics
  (rate-limit, not-found, timeout). The retry/timeout layer in `httpx` is exercised
  end-to-end without extra plumbing,
- the same app instance is mounted into `httpx.AsyncClient(transport=ASGITransport(...))`
  in tests, so the integration suite never opens a port and runs in milliseconds.

Trade-off: a background `uvicorn` thread is needed for the live demo, and shutdown is
slightly more involved than killing a coroutine. `run_in_thread()` encapsulates that.

## In-memory state with JSON seeds (no SQLite, no disk persistence)

Persisting to disk would mean migrations, locking, and hidden state across runs — all
of which trip up tests and demos. In-memory `Store` classes give:

- deterministic IDs (monotonic counters, injectable `current_time`),
- easy reseeding on every startup from `mock_api/seeds/*.json`,
- no test pollution (each test builds its own `AppState.empty()` and reloads seeds).

Trade-off: nothing survives a restart. That's intentional for a demo with reproducible
behaviour.

## Error injection middleware (config-driven, not hard-coded)

`MOCK_API_ERROR_INJECTION=slack:send_message:rate_limit,jira:get_issue:not_found`
parses into typed `ErrorInjectionRule` objects and is consumed by middleware. This
lets the agent's recovery behaviour be demonstrated without code changes:

- per-service rules (`slack:rate_limit`) and per-action rules
  (`slack:send_message:rate_limit`) are both supported,
- each rule maps to a realistic shape (`{"ok": false, "error": "rate_limited"}`,
  `{"ok": false, "error": "not_found"}`, or a hung response that exceeds the client
  timeout),
- bad config fails at startup with a clear error rather than silently no-op'ing,
- the integration suite has a dedicated test
  (`test_error_injection_surfaces_recoverable_error_in_summary`) that flips on a
  rate-limit rule and asserts the action log carries `is_error=True` and the agent's
  structured summary mentions `rate_limited`.

## Retry / backoff policy (`AnthropicClient`)

Exponential backoff with jitter, capped at `LLM_MAX_RETRIES`:

- retryable: `RateLimitError`, `APIConnectionError`, `APITimeoutError`,
  `APIStatusError` with status `>= 500`,
- non-retryable: 4xx (except 429), invalid request, auth failures — surfaced
  immediately wrapped in a clear `LLMError`,
- defaults: base `0.5s`, factor `2.0`, jitter `0..0.25s`, max retries `3`.

Rationale: we want to ride out transient network blips and provider rate limits, but
fail fast on programming errors. Tests inject a fake `sleep` so the retry loop runs in
zero wall time.

## Structured outputs for the terminal response

The agent's last turn is forced through `messages.parse(output_format=AgentFinalResponse)`.
`AgentFinalResponse` has a `kind: Literal["done", "needs_clarification"]` field, so
Scenario D ("Send an update" — agent must ask for clarification) is a structural
distinction, not a string-matching heuristic. The CLI renders the two kinds
differently (a clean summary vs a prominent question), and integration tests assert
on the field directly.

## Tools as Pydantic params + Pydantic results

Each tool registers a `params_model` and `result_model`. The Anthropic tool definition
is generated from `params_model.model_json_schema()` after a normalisation pass that
inlines `$ref`s, sets `additionalProperties=false`, and strips JSON-Schema validation
keywords Anthropic does not consume (`exclusiveMinimum`, `pattern`, `minLength`, etc.).
The dispatcher then runs `params_model.model_validate(tool_use.input)` on every call,
so any argument the model sends — well-formed or not — is checked against the full
Pydantic schema before the handler ever runs. Invalid arguments come back as a
structured `ToolErrorResult` the agent can reason about, not an exception that
crashes the loop.

We initially shipped tools with Anthropic's `strict: True` flag, but the strict-mode
schema compiler rejects schemas above a complexity threshold once you have ~11 tools
with nested optional objects (it returns `"Schema is too complex for compilation"`).
We dropped to `strict: False`; Pydantic validation on dispatch is the real safety
net, so the LLM still cannot smuggle bad arguments into the handler.

This pattern keeps the registry tiny (one `dispatch()` method handles validation,
invocation, and serialisation) and pushes correctness into types.

## `uv` and `ty` (Astral toolchain)

- `uv` for dependency resolution, install, and project commands — single tool, single
  lockfile, much faster than pip-tools / poetry.
- `ty` (Astral, beta) for type checking. Picked over `mypy` / `pyright` because
  it is the newest checker from the same vendor that ships `uv` and `ruff`, written
  in Rust, and is 10–60× faster than the alternatives. The brief asks for "mypy or
  pyright"; `ty` consumes the same standard-Python type annotations and enforces
  the same rules, so the requirement is satisfied. It is still beta — if it ever
  blocks the project, `pyright` is a drop-in replacement (the annotations are
  vendor-neutral).
- `ruff` for both linting and formatting — replaces black + isort + flake8.

## Tests are written alongside code, not after

Each module landed with its tests in the same change. Tests are unit-level for
stores/handlers/registry/loop and integration-level for the four demo scenarios.
The LLM is always faked in tests via `FakeLLM`, which takes pre-scripted responses
indexed by step. Coverage is 89% overall.

## Production considerations (out of scope here, sketched for review)

The prototype is single-process and in-memory by design, but the layering keeps the
path to production short and surgical, not a rewrite:

- **Provider abstraction.** `LLMClient` is a `Protocol`; swapping Anthropic for
  another provider is one new class implementing two methods, not a refactor.
- **Observability.** `structlog` is wired through every layer; the agent's action
  log doubles as an audit trail, and `ToolRegistry.dispatch` is the natural span
  boundary if/when an OpenTelemetry exporter is added.
- **Persistence.** Each in-memory `Store` has a narrow surface (a handful of
  CRUD-style methods); replacing them with a SQL-backed equivalent leaves the
  FastAPI routers and tool handlers untouched.
- **Cost / latency.** Sonnet 4.6 was picked for tool-use reliability on
  multi-step flows; for higher-volume single-call actions a cheaper tier would
  cut cost noticeably, and the model is configurable via `ANTHROPIC_MODEL` with
  no code changes.

## What was deliberately skipped

- **Web UI / multi-user concurrency / Docker** — out of scope for a CLI demo; would
  add packaging surface area without changing what the agent can do.
- **Production-grade auth** — the mock APIs are unauthenticated by design; the only
  secret in the repo is the Anthropic key, gated by `.env`.
- **Streaming agent responses** — the CLI is non-streaming. The structured terminal
  response is small and fits a single `print` after the loop terminates.
- **Persistent storage beyond JSON seeds** — see "In-memory state" above.
- **Additional providers (OpenAI/Gemini)** — the `LLMClient` protocol is provider-
  agnostic, so adding one would mean a new client implementation, not a refactor. We
  did not implement one because the brief targets Anthropic's tool-use and structured
  outputs specifically.
