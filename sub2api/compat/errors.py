"""Turn upstream ``event: error`` frames into real, classifiable failures.

The AdaL cloud proxy answers a rejected Anthropic request with HTTP 200 and
an ``event: error`` frame inside the SSE stream (measured: ``Messages.stream()
got an unexpected keyword argument 'context_management'`` with
``error.type == "invalid_request_error"``). Relayed verbatim, that looks like
a successful empty response to the client. These helpers extract the error
object, map its ``type`` to the HTTP status the official Anthropic API would
have returned, and build the native error envelope for the client.
"""

from __future__ import annotations

# err_type -> HTTP status, matching the official Anthropic API.
_STATUS_BY_TYPE: dict[str, int] = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "rate_limit_error": 429,
    "overloaded_error": 529,
}

# Safe shape for a malformed error frame: the request did fail upstream even
# when the proxy's error envelope is unusable, and an unclassifiable failure
# must not be reported as the client's fault (hence api_error / 502).
_UNCLASSIFIED: dict[str, str] = {"type": "api_error", "message": "upstream error"}


def parse_error_frame(frames: list[dict]) -> dict | None:
    """Return the first Anthropic or Responses stream error object.

    ``frames`` are decoded SSE ``data:`` payloads. Returns ``None`` when no
    frame carries an error. A frame whose ``error`` value is missing, not a
    dict, or empty yields the safe default ``{"type": "api_error",
    "message": "upstream error"}`` instead of raising or leaking an
    unshaped dict to callers.
    """
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        event_type = frame.get("type")
        if event_type == "response.failed":
            response = frame.get("response")
            error = response.get("error") if isinstance(response, dict) else None
        elif event_type in ("error", "response.error"):
            error = frame.get("error")
            if error is None and isinstance(frame.get("message"), str):
                error = {
                    key: frame[key]
                    for key in ("message", "code", "param")
                    if key in frame
                }
        else:
            continue
        if (
            isinstance(error, dict)
            and isinstance(error.get("type"), str)
            and error["type"]
        ):
            return error
        if (
            isinstance(error, dict)
            and isinstance(error.get("message"), str)
            and error["message"]
        ):
            normalized = dict(error)
            normalized["type"] = (
                "rate_limit_error"
                if error.get("code") == "rate_limit_exceeded"
                else "api_error"
            )
            return normalized
        return dict(_UNCLASSIFIED)
    return None


def status_for_anthropic_error(err_type: str) -> int:
    """Map an Anthropic ``error.type`` to the HTTP status it deserves.

    Missing, empty, and unknown types map to 502: the failure is real but
    unclassifiable, so it surfaces as a bad gateway rather than a client
    error.
    """
    return _STATUS_BY_TYPE.get(err_type, 502)


def anthropic_error(message: str, err_type: str = "invalid_request_error") -> dict:
    """Build the native Anthropic error envelope."""
    return {"type": "error", "error": {"type": err_type, "message": message}}
