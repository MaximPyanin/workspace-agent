import httpx

from workplace_agent.tools._http import http_error


def _response(*, status: int, body: str, content_type: str = "application/json") -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=body.encode("utf-8"),
        headers={"content-type": content_type},
    )


def test_http_error_extracts_top_level_error() -> None:
    resp = _response(status=429, body='{"ok": false, "error": "rate_limited"}')

    result = http_error(resp)

    assert result.error == "rate_limited"
    assert result.detail["status_code"] == 429
    assert result.detail["body"] == {"ok": False, "error": "rate_limited"}


def test_http_error_returns_default_when_body_lacks_error_key() -> None:
    resp = _response(status=500, body='{"ok": false}')

    result = http_error(resp)

    assert result.error == "http_error"
    assert result.detail["body"] == {"ok": False}
