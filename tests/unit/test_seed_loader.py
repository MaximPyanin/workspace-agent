import json
from datetime import UTC, datetime
from pathlib import Path

from workplace_agent.mock_api.state import DEFAULT_SEED_DIR, AppState


def _write_seed(path: Path, name: str, payload: dict) -> None:
    (path / name).write_text(json.dumps(payload), encoding="utf-8")


def _fixed_clock():
    fixed = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    return lambda: fixed


def test_state_starts_empty() -> None:
    s = AppState.empty()
    assert s.slack.list_channels() == []
    assert s.slack.list_users() == []
    assert s.jira.list_issues() == []
    assert s.calendar.list_events() == []
    assert s.email.list_emails() == []
    assert s.seeded is False


def test_load_seeds_populates_from_directory(tmp_path: Path) -> None:
    _write_seed(
        tmp_path,
        "slack.json",
        {
            "channels": [{"name": "engineering"}, "product"],
            "users": [{"id": "U1", "name": "alice"}],
        },
    )
    _write_seed(
        tmp_path,
        "jira.json",
        {
            "issues": [
                {
                    "key": "ENG-1",
                    "project": "ENG",
                    "summary": "thing",
                    "status": "Open",
                    "created_at": "2026-04-01T00:00:00+00:00",
                    "updated_at": "2026-04-01T00:00:00+00:00",
                }
            ]
        },
    )
    _write_seed(
        tmp_path,
        "calendar.json",
        {
            "events": [
                {
                    "title": "standup",
                    "start": "2026-04-27T09:30:00+00:00",
                    "end": "2026-04-27T10:00:00+00:00",
                    "attendees": ["alice@example.com"],
                }
            ]
        },
    )
    _write_seed(
        tmp_path,
        "email.json",
        {
            "emails": [
                {
                    "sender": "ceo@x",
                    "recipients": ["all@x"],
                    "subject": "hi",
                    "body": "team",
                    "sent_at": "2026-04-20T09:00:00+00:00",
                    "thread_id": "t1",
                }
            ]
        },
    )

    s = AppState.empty(current_time=_fixed_clock())
    s.load_seeds(tmp_path)

    assert {c.name for c in s.slack.list_channels()} == {"engineering", "product"}
    assert [u.name for u in s.slack.list_users()] == ["alice"]
    issue = s.jira.get_issue("ENG-1")
    assert issue is not None and issue.summary == "thing"
    events = s.calendar.list_events()
    assert len(events) == 1 and events[0].title == "standup"
    emails = s.email.list_emails()
    assert len(emails) == 1 and emails[0].thread_id == "t1"
    assert s.seeded is True


def test_load_seeds_idempotent(tmp_path: Path) -> None:
    _write_seed(tmp_path, "slack.json", {"channels": ["engineering"]})
    s = AppState.empty(current_time=_fixed_clock())
    s.load_seeds(tmp_path)
    s.load_seeds(tmp_path)
    s.load_seeds(tmp_path)
    assert len(s.slack.list_channels()) == 1


def test_load_seeds_force_reseed_does_not_duplicate_jira_keys(tmp_path: Path) -> None:
    _write_seed(
        tmp_path,
        "jira.json",
        {
            "issues": [
                {
                    "key": "ENG-1",
                    "project": "ENG",
                    "summary": "x",
                    "created_at": "2026-04-01T00:00:00+00:00",
                    "updated_at": "2026-04-01T00:00:00+00:00",
                }
            ]
        },
    )
    s = AppState.empty(current_time=_fixed_clock())
    s.load_seeds(tmp_path)

    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        s.load_seeds(tmp_path, force=True)


def test_load_seeds_missing_directory_is_noop(tmp_path: Path) -> None:
    s = AppState.empty()
    s.load_seeds(tmp_path / "does-not-exist")
    assert s.slack.list_channels() == []
    assert s.seeded is True


def test_load_seeds_partial_files(tmp_path: Path) -> None:
    _write_seed(tmp_path, "slack.json", {"channels": ["engineering"]})
    s = AppState.empty()
    s.load_seeds(tmp_path)
    assert len(s.slack.list_channels()) == 1
    assert s.jira.list_issues() == []


def test_default_seed_dir_loads_packaged_seeds() -> None:
    s = AppState.empty(current_time=_fixed_clock())
    s.load_seeds(DEFAULT_SEED_DIR)
    assert len(s.slack.list_channels()) == 3
    assert len(s.slack.list_users()) == 2
    assert len(s.jira.list_issues()) == 2
    assert len(s.calendar.list_events()) == 3
    assert len(s.email.list_emails()) == 2


def test_jira_seed_advances_counter_for_subsequent_creates() -> None:
    s = AppState.empty(current_time=_fixed_clock())
    s.load_seeds(DEFAULT_SEED_DIR)
    nxt = s.jira.create_issue(project="ENG", summary="new one")
    assert nxt.key == "ENG-3"
