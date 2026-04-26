import itertools
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

CurrentTime = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


DEFAULT_SEED_DIR: Path = Path(__file__).resolve().parent / "seeds"


class Channel(BaseModel):
    id: str
    name: str


class SlackUser(BaseModel):
    id: str
    name: str
    email: str | None = None


class SlackMessage(BaseModel):
    ts: str
    channel: str
    user: str | None = None
    text: str


class JiraIssue(BaseModel):
    key: str
    project: str
    summary: str
    description: str = ""
    status: str = "Open"
    assignee: str | None = None
    created_at: datetime
    updated_at: datetime


class CalendarEvent(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[str] = Field(default_factory=list)
    description: str = ""


class Email(BaseModel):
    id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    sent_at: datetime
    thread_id: str | None = None


def _normalize_channel_name(name: str) -> str:
    return name.lstrip("#").strip()


class SlackStore:
    def __init__(self, *, current_time: CurrentTime = _utc_now) -> None:
        self._current_time = current_time
        self._channels_by_id: dict[str, Channel] = {}
        self._channels_by_name: dict[str, Channel] = {}
        self._messages: list[SlackMessage] = []
        self._users: dict[str, SlackUser] = {}
        self._channel_seq = itertools.count(1)
        self._message_seq = itertools.count(1)
        self._lock = threading.Lock()

    def add_channel(self, name: str) -> Channel:
        norm = _normalize_channel_name(name)
        if not norm:
            raise ValueError("channel name cannot be empty")
        with self._lock:
            existing = self._channels_by_name.get(norm)
            if existing is not None:
                return existing
            cid = f"C{next(self._channel_seq):05d}"
            channel = Channel(id=cid, name=norm)
            self._channels_by_id[cid] = channel
            self._channels_by_name[norm] = channel
            return channel

    def get_channel_by_name(self, name: str) -> Channel | None:
        return self._channels_by_name.get(_normalize_channel_name(name))

    def get_channel_by_id(self, cid: str) -> Channel | None:
        return self._channels_by_id.get(cid)

    def resolve_channel(self, name_or_id: str) -> Channel | None:
        return self.get_channel_by_name(name_or_id) or self.get_channel_by_id(name_or_id)

    def list_channels(self) -> list[Channel]:
        return list(self._channels_by_id.values())

    def add_user(self, *, id: str, name: str, email: str | None = None) -> SlackUser:
        user = SlackUser(id=id, name=name, email=email)
        with self._lock:
            self._users[user.id] = user
            return user

    def list_users(self) -> list[SlackUser]:
        return list(self._users.values())

    def send_message(
        self, *, channel: str, text: str, user: str | None = None, ts: str | None = None
    ) -> SlackMessage:
        if not text or not text.strip():
            raise ValueError("message text cannot be empty")
        ch = self.resolve_channel(channel)
        if ch is None:
            raise KeyError(f"unknown channel: {channel!r}")
        with self._lock:
            seq = next(self._message_seq)
            actual_ts = (
                ts if ts is not None else f"{int(self._current_time().timestamp())}.{seq:06d}"
            )
            msg = SlackMessage(ts=actual_ts, channel=ch.id, user=user, text=text)
            self._messages.append(msg)
            return msg

    def list_messages(self, *, channel: str | None = None) -> list[SlackMessage]:
        if channel is None:
            return list(self._messages)
        ch = self.resolve_channel(channel)
        if ch is None:
            return []
        return [m for m in self._messages if m.channel == ch.id]

    def search_messages(self, query: str) -> list[SlackMessage]:
        q = query.lower()
        return [m for m in self._messages if q in m.text.lower()]


class JiraStore:
    DEFAULT_STATUSES: tuple[str, ...] = ("Open", "In Progress", "Blocked", "Done")

    def __init__(self, *, current_time: CurrentTime = _utc_now) -> None:
        self._current_time = current_time
        self._issues: dict[str, JiraIssue] = {}
        self._project_counters: dict[str, int] = {}
        self._lock = threading.Lock()
        self.allowed_statuses: tuple[str, ...] = self.DEFAULT_STATUSES

    def _bump_counter(self, project: str, n: int) -> None:
        cur = self._project_counters.get(project, 0)
        if n > cur:
            self._project_counters[project] = n

    def create_issue(
        self,
        *,
        project: str,
        summary: str,
        description: str = "",
        assignee: str | None = None,
        status: str = "Open",
    ) -> JiraIssue:
        project = project.strip()
        if not project:
            raise ValueError("project must be non-empty")
        if not summary or not summary.strip():
            raise ValueError("summary must be non-empty")
        if status not in self.allowed_statuses:
            raise ValueError(f"invalid status {status!r}; allowed: {self.allowed_statuses}")
        with self._lock:
            n = self._project_counters.get(project, 0) + 1
            self._project_counters[project] = n
            key = f"{project}-{n}"
            now = self._current_time()
            issue = JiraIssue(
                key=key,
                project=project,
                summary=summary,
                description=description,
                status=status,
                assignee=assignee,
                created_at=now,
                updated_at=now,
            )
            self._issues[key] = issue
            return issue

    def insert_issue(self, issue: JiraIssue) -> JiraIssue:
        with self._lock:
            if issue.key in self._issues:
                raise ValueError(f"duplicate jira key {issue.key!r}")
            self._issues[issue.key] = issue
            try:
                seq = int(issue.key.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                seq = 0
            self._bump_counter(issue.project, seq)
            return issue

    def get_issue(self, key: str) -> JiraIssue | None:
        return self._issues.get(key)

    def list_issues(self, *, project: str | None = None) -> list[JiraIssue]:
        if project is None:
            return list(self._issues.values())
        return [i for i in self._issues.values() if i.project == project]

    def transition_issue(
        self,
        *,
        key: str,
        status: str | None = None,
        assignee: str | None = None,
    ) -> JiraIssue:
        if status is None and assignee is None:
            raise ValueError("transition requires status or assignee")
        if status is not None and status not in self.allowed_statuses:
            raise ValueError(f"invalid status {status!r}; allowed: {self.allowed_statuses}")
        with self._lock:
            issue = self._issues.get(key)
            if issue is None:
                raise KeyError(key)
            updates: dict[str, Any] = {"updated_at": self._current_time()}
            if status is not None:
                updates["status"] = status
            if assignee is not None:
                updates["assignee"] = assignee
            new_issue = issue.model_copy(update=updates)
            self._issues[key] = new_issue
            return new_issue


class CalendarStore:
    def __init__(self, *, current_time: CurrentTime = _utc_now) -> None:
        self._current_time = current_time
        self._events: dict[str, CalendarEvent] = {}
        self._seq = itertools.count(1)
        self._lock = threading.Lock()

    def create_event(
        self,
        *,
        title: str,
        start: datetime,
        end: datetime,
        attendees: list[str] | None = None,
        description: str = "",
    ) -> CalendarEvent:
        if end <= start:
            raise ValueError("event end must be after start")
        if not title or not title.strip():
            raise ValueError("event title required")
        with self._lock:
            eid = f"evt_{next(self._seq):06d}"
            event = CalendarEvent(
                id=eid,
                title=title,
                start=start,
                end=end,
                attendees=list(attendees or []),
                description=description,
            )
            self._events[eid] = event
            return event

    def get_event(self, eid: str) -> CalendarEvent | None:
        return self._events.get(eid)

    def list_events(
        self,
        *,
        from_: datetime | None = None,
        to: datetime | None = None,
    ) -> list[CalendarEvent]:
        result: list[CalendarEvent] = []
        for ev in self._events.values():
            if from_ is not None and ev.end <= from_:
                continue
            if to is not None and ev.start >= to:
                continue
            result.append(ev)
        return sorted(result, key=lambda e: e.start)


class EmailStore:
    def __init__(self, *, current_time: CurrentTime = _utc_now) -> None:
        self._current_time = current_time
        self._emails: dict[str, Email] = {}
        self._seq = itertools.count(1)
        self._lock = threading.Lock()

    def send(
        self,
        *,
        sender: str,
        recipients: list[str],
        subject: str,
        body: str,
        thread_id: str | None = None,
        sent_at: datetime | None = None,
    ) -> Email:
        if not sender or not sender.strip():
            raise ValueError("sender required")
        if not recipients:
            raise ValueError("at least one recipient is required")
        with self._lock:
            eid = f"em_{next(self._seq):06d}"
            email = Email(
                id=eid,
                sender=sender,
                recipients=list(recipients),
                subject=subject,
                body=body,
                sent_at=sent_at if sent_at is not None else self._current_time(),
                thread_id=thread_id,
            )
            self._emails[eid] = email
            return email

    def get(self, eid: str) -> Email | None:
        return self._emails.get(eid)

    def list_emails(self) -> list[Email]:
        return list(self._emails.values())

    def search(self, query: str) -> list[Email]:
        q = query.lower().strip()
        if not q:
            return []
        scored: list[tuple[int, Email]] = []
        for em in self._emails.values():
            score = 0
            if q in em.subject.lower():
                score += 2
            if q in em.body.lower():
                score += 1
            if score:
                scored.append((score, em))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [em for _, em in scored]


class AppState:
    def __init__(
        self,
        *,
        slack: SlackStore,
        jira: JiraStore,
        calendar: CalendarStore,
        email: EmailStore,
        current_time: CurrentTime,
    ) -> None:
        self.slack = slack
        self.jira = jira
        self.calendar = calendar
        self.email = email
        self.current_time = current_time
        self._seeded = False
        self._seed_lock = threading.Lock()

    @classmethod
    def empty(cls, *, current_time: CurrentTime | None = None) -> Self:
        ct: CurrentTime = current_time if current_time is not None else _utc_now
        return cls(
            slack=SlackStore(current_time=ct),
            jira=JiraStore(current_time=ct),
            calendar=CalendarStore(current_time=ct),
            email=EmailStore(current_time=ct),
            current_time=ct,
        )

    @property
    def seeded(self) -> bool:
        return self._seeded

    def load_seeds(self, seed_dir: Path | None = None, *, force: bool = False) -> None:
        path = seed_dir if seed_dir is not None else DEFAULT_SEED_DIR
        with self._seed_lock:
            if self._seeded and not force:
                return
            self._load_slack(path / "slack.json")
            self._load_jira(path / "jira.json")
            self._load_calendar(path / "calendar.json")
            self._load_email(path / "email.json")
            self._seeded = True

    def _load_slack(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data.get("channels", []):
            name = raw if isinstance(raw, str) else raw["name"]
            self.slack.add_channel(name)
        for raw in data.get("users", []):
            self.slack.add_user(id=raw["id"], name=raw["name"], email=raw.get("email"))
        for raw in data.get("messages", []):
            self.slack.send_message(
                channel=raw["channel"],
                text=raw["text"],
                user=raw.get("user"),
                ts=raw.get("ts"),
            )

    def _load_jira(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        now = self.current_time()
        for raw in data.get("issues", []):
            issue = JiraIssue(
                key=raw["key"],
                project=raw["project"],
                summary=raw["summary"],
                description=raw.get("description", ""),
                status=raw.get("status", "Open"),
                assignee=raw.get("assignee"),
                created_at=raw.get("created_at") or now,
                updated_at=raw.get("updated_at") or now,
            )
            self.jira.insert_issue(issue)

    def _load_calendar(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data.get("events", []):
            self.calendar.create_event(
                title=raw["title"],
                start=_parse_dt(raw["start"]),
                end=_parse_dt(raw["end"]),
                attendees=list(raw.get("attendees", [])),
                description=raw.get("description", ""),
            )

    def _load_email(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data.get("emails", []):
            sent_at = _parse_dt(raw["sent_at"]) if raw.get("sent_at") else None
            self.email.send(
                sender=raw["sender"],
                recipients=list(raw["recipients"]),
                subject=raw["subject"],
                body=raw["body"],
                thread_id=raw.get("thread_id"),
                sent_at=sent_at,
            )


def _parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)
