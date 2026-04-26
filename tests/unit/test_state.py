from datetime import UTC, datetime, timedelta

import pytest

from workplace_agent.mock_api.state import (
    CalendarStore,
    EmailStore,
    JiraStore,
    SlackStore,
)


def _fixed_clock(start: datetime | None = None, *, step_seconds: float = 1.0):
    base = start or datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    counter = {"i": 0}

    def now() -> datetime:
        ts = base + timedelta(seconds=step_seconds * counter["i"])
        counter["i"] += 1
        return ts

    return now


def test_slack_add_channel_normalizes_and_dedupes() -> None:
    s = SlackStore(current_time=_fixed_clock())
    a = s.add_channel("#engineering")
    b = s.add_channel("engineering")
    assert a.id == b.id
    assert a.name == "engineering"
    assert len(s.list_channels()) == 1


def test_slack_send_message_appends_and_assigns_ts() -> None:
    s = SlackStore(current_time=_fixed_clock())
    s.add_channel("#general")
    msg = s.send_message(channel="#general", text="hello", user="U001")
    assert msg.text == "hello"
    assert msg.user == "U001"
    assert "." in msg.ts
    assert s.list_messages(channel="#general") == [msg]


def test_jira_create_issue_assigns_ascending_keys_per_project() -> None:
    j = JiraStore(current_time=_fixed_clock())
    a = j.create_issue(project="ENG", summary="one")
    b = j.create_issue(project="ENG", summary="two")
    c = j.create_issue(project="OPS", summary="x")
    d = j.create_issue(project="OPS", summary="y")
    assert (a.key, b.key, c.key, d.key) == ("ENG-1", "ENG-2", "OPS-1", "OPS-2")


def test_jira_get_issue_returns_full_payload() -> None:
    j = JiraStore(current_time=_fixed_clock())
    issue = j.create_issue(project="ENG", summary="thing", description="desc", assignee="alice")
    fetched = j.get_issue(issue.key)
    assert fetched == issue
    assert fetched is not None
    assert fetched.assignee == "alice"
    assert fetched.description == "desc"
    assert fetched.status == "Open"


def test_jira_transition_updates_status_and_assignee() -> None:
    j = JiraStore(current_time=_fixed_clock())
    issue = j.create_issue(project="ENG", summary="x")
    transitioned = j.transition_issue(key=issue.key, status="In Progress", assignee="bob")
    assert transitioned.status == "In Progress"
    assert transitioned.assignee == "bob"
    assert transitioned.updated_at >= issue.updated_at


def _dt(h: int, m: int = 0) -> datetime:
    return datetime(2026, 4, 27, h, m, tzinfo=UTC)


def test_calendar_create_event_validates_end_after_start() -> None:
    c = CalendarStore(current_time=_fixed_clock())
    with pytest.raises(ValueError, match="end must be after start"):
        c.create_event(title="t", start=_dt(10), end=_dt(10))
    with pytest.raises(ValueError, match="end must be after start"):
        c.create_event(title="t", start=_dt(11), end=_dt(10))


def test_email_ids_monotonic() -> None:
    e = EmailStore(current_time=_fixed_clock())
    a = e.send(sender="a@x", recipients=["b@x"], subject="one", body="x")
    b = e.send(sender="a@x", recipients=["b@x"], subject="two", body="y")
    assert a.id < b.id
