"""API-key guard shared by every HTTP surface.

Module-level, not a closure over ``create_app``'s ``settings``: the admin
router lives in its own module and the route split (`server/routes/*`) needs
the same guard, so the key is read from ``request.app.state.settings`` at call
time.  That also means a settings object swapped in by a test is honoured.

Two header spellings are accepted because real clients disagree: the
Anthropic SDK and Claude Code send ``x-api-key``, OpenAI-compatible clients
send ``Authorization: Bearer``.  Comparison is constant-time — a plain ``!=``
on a secret leaks its prefix through timing.
"""

from __future__ import annotations

import hmac

from fastapi import Request
from fastapi.responses import JSONResponse

from . import openai as oai


def expected_key(request: Request) -> str:
    """The configured API key, or ``""`` when the gateway is open."""
    settings = getattr(request.app.state, "settings", None)
    return getattr(settings, "api_key", None) or ""


def supplied_key(request: Request) -> str:
    """The caller's key from ``x-api-key`` or an ``Authorization`` bearer."""
    key = request.headers.get("x-api-key") or ""
    if key:
        return key
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def unauthorized(request: Request) -> JSONResponse | None:
    """``None`` when the caller may proceed, else a 401 response to return."""
    expected = expected_key(request)
    if not expected:
        return None
    if not hmac.compare_digest(supplied_key(request), expected):
        return JSONResponse(
            status_code=401,
            content=oai.openai_error(
                "invalid api key", err_type="authentication_error"
            ),
        )
    return None
