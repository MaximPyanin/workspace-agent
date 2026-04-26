import json
import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from workplace_agent.logging_setup import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logging() -> Generator[None]:
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)


def _read_event(line: str) -> dict[str, object]:
    return json.loads(line.strip())


def test_format_includes_expected_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO")
    log = get_logger("workplace_agent.test")
    log.info("hello world", user="alice")

    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    payload = _read_event(line)

    assert payload["event"] == "hello world"
    assert payload["level"] == "info"
    assert payload["logger"] == "workplace_agent.test"
    assert payload["user"] == "alice"
    assert "timestamp" in payload
    assert isinstance(payload["timestamp"], str)


def test_level_honored_filters_debug(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO")
    log = get_logger("workplace_agent.test")
    log.debug("hidden")
    log.info("visible")

    captured = capsys.readouterr()
    lines = [ln for ln in captured.err.strip().splitlines() if ln]
    events = [_read_event(ln)["event"] for ln in lines]
    assert "hidden" not in events
    assert "visible" in events


def test_debug_level_emits_debug(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="DEBUG")
    log = get_logger("workplace_agent.test")
    log.debug("now visible")

    captured = capsys.readouterr()
    lines = [ln for ln in captured.err.strip().splitlines() if ln]
    events = [_read_event(ln)["event"] for ln in lines]
    assert "now visible" in events


def test_log_level_case_insensitive(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="warning")
    log = get_logger("workplace_agent.test")
    log.info("not shown")
    log.warning("shown")

    captured = capsys.readouterr()
    events = [_read_event(ln)["event"] for ln in captured.err.strip().splitlines() if ln]
    assert events == ["shown"]


def test_log_file_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"
    configure_logging(level="INFO", log_file=str(log_file))
    log = get_logger("workplace_agent.test")
    log.info("to-file", k=1)

    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "expected at least one log line"
    payload = _read_event(contents[-1])
    assert payload["event"] == "to-file"
    assert payload["k"] == 1


def test_unknown_level_raises() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="LOUD")
