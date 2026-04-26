import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Final, get_args

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.types import ASGIApp

from workplace_agent.config import ErrorInjectionRule, ErrorType

DEFAULT_TIMEOUT_HANG_SECONDS: Final[float] = 5.0

ERROR_BODIES: Final[dict[str, tuple[int, dict[str, object]]]] = {
    "rate_limit": (429, {"ok": False, "error": "rate_limited"}),
    "not_found": (404, {"ok": False, "error": "not_found"}),
    "timeout": (504, {"ok": False, "error": "timeout"}),
}


def _service_from_path(path: str) -> str | None:
    parts = [p for p in path.split("/") if p]
    return parts[0] if parts else None


def _resolve_action(request: Request) -> str | None:
    app = request.scope.get("app")
    router = getattr(app, "router", None)
    if router is None:
        return None
    for route in router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            name = getattr(route, "name", None)
            return name or None
    return None


class ErrorInjectionMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        rules: Iterable[ErrorInjectionRule] | None = None,
        *,
        timeout_hang_seconds: float = DEFAULT_TIMEOUT_HANG_SECONDS,
    ) -> None:
        super().__init__(app)
        rule_list = list(rules or [])
        valid_types = set(get_args(ErrorType))
        for rule in rule_list:
            if not isinstance(rule, ErrorInjectionRule):
                raise TypeError(
                    f"ErrorInjectionMiddleware: expected ErrorInjectionRule, "
                    f"got {type(rule).__name__}"
                )
            if rule.error_type not in valid_types:
                raise ValueError(
                    f"ErrorInjectionMiddleware: invalid error_type {rule.error_type!r}; "
                    f"allowed: {sorted(valid_types)}"
                )
            if not rule.service:
                raise ValueError("ErrorInjectionMiddleware: rule.service must be non-empty")
        if timeout_hang_seconds <= 0:
            raise ValueError("timeout_hang_seconds must be > 0")
        self._rules: list[ErrorInjectionRule] = rule_list
        self._timeout_hang_seconds = timeout_hang_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self._rules:
            return await call_next(request)
        service = _service_from_path(request.scope.get("path", ""))
        if service is None:
            return await call_next(request)
        action = _resolve_action(request)
        if action is None:
            return await call_next(request)
        for rule in self._rules:
            if rule.matches(service, action):
                return await self._inject(rule)
        return await call_next(request)

    async def _inject(self, rule: ErrorInjectionRule) -> Response:
        if rule.error_type == "timeout":
            await asyncio.sleep(self._timeout_hang_seconds)
        status, body = ERROR_BODIES[rule.error_type]
        return JSONResponse(body, status_code=status)


def add_error_injection(
    app: FastAPI,
    rules: Iterable[ErrorInjectionRule] | None,
    *,
    timeout_hang_seconds: float = DEFAULT_TIMEOUT_HANG_SECONDS,
) -> None:
    rule_list = list(rules or [])
    if not rule_list:
        return
    app.add_middleware(
        ErrorInjectionMiddleware,
        rules=rule_list,
        timeout_hang_seconds=timeout_hang_seconds,
    )
