from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from workplace_agent.config import ErrorInjectionRule, Settings

_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_BETA_HEADER",
    "MOCK_API_HOST",
    "MOCK_API_PORT",
    "MOCK_API_ERROR_INJECTION",
    "LOG_LEVEL",
    "LOG_FILE",
    "AGENT_MAX_STEPS",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "AGENT_USE_FAKE_LLM",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[None]:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


def _make(**overrides: Any) -> Settings:
    overrides.setdefault("anthropic_api_key", "sk-test")
    return Settings.model_validate(overrides)


def test_defaults_applied() -> None:
    s = _make()
    assert s.anthropic_model == "claude-sonnet-4-6"
    assert s.anthropic_beta_header == "structured-outputs-2025-11-13"
    assert s.mock_api_host == "127.0.0.1"
    assert s.mock_api_port == 8765
    assert s.mock_api_error_injection == []
    assert s.log_level == "INFO"
    assert s.log_file is None
    assert s.agent_max_steps == 12
    assert s.llm_timeout_seconds == 30.0
    assert s.llm_max_retries == 3
    assert s.agent_use_fake_llm is False


def test_env_vars_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    monkeypatch.setenv("MOCK_API_PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    s = Settings()
    assert s.anthropic_api_key == "sk-from-env"
    assert s.mock_api_port == 9000
    assert s.log_level == "DEBUG"
    assert s.agent_max_steps == 7


def test_error_injection_three_part_rule() -> None:
    s = _make(mock_api_error_injection="slack:send_message:rate_limit")
    assert s.mock_api_error_injection == [
        ErrorInjectionRule(service="slack", action="send_message", error_type="rate_limit"),
    ]


def test_error_injection_multiple_rules() -> None:
    raw = "slack:rate_limit, jira:get_issue:not_found ,calendar:timeout"
    s = _make(mock_api_error_injection=raw)
    assert s.mock_api_error_injection == [
        ErrorInjectionRule(service="slack", action=None, error_type="rate_limit"),
        ErrorInjectionRule(service="jira", action="get_issue", error_type="not_found"),
        ErrorInjectionRule(service="calendar", action=None, error_type="timeout"),
    ]


def test_error_injection_empty_string_is_empty_list() -> None:
    assert _make(mock_api_error_injection="").mock_api_error_injection == []


def test_error_injection_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError) as exc:
        _make(mock_api_error_injection="slack:explode")
    assert "error_type" in str(exc.value)


def test_missing_api_key_raises_when_not_fake() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings()
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "AGENT_USE_FAKE_LLM" in msg


def test_missing_api_key_ok_when_fake() -> None:
    s = Settings.model_validate({"agent_use_fake_llm": True})
    assert s.anthropic_api_key == ""
    assert s.agent_use_fake_llm is True


def test_agent_max_steps_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        _make(agent_max_steps=0)


def test_error_injection_loaded_from_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-test\n"
        "MOCK_API_ERROR_INJECTION=slack:send_message:rate_limit,jira:get_issue:not_found\n"
    )
    s = Settings()
    assert s.mock_api_error_injection == [
        ErrorInjectionRule(service="slack", action="send_message", error_type="rate_limit"),
        ErrorInjectionRule(service="jira", action="get_issue", error_type="not_found"),
    ]


def test_llm_timeout_seconds_rejects_non_positive() -> None:
    with pytest.raises(ValidationError):
        _make(llm_timeout_seconds=0)
    with pytest.raises(ValidationError):
        _make(llm_timeout_seconds=-0.5)
