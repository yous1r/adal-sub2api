"""Subscription quota and credit-balance probes for the AdaL cloud channel.

Separate module because these are the *billing* surface, not the proxy: they
never touch the model catalog and never take a slot, and the admin UI reads
``fetch_credits_raw`` directly.  ``fetch_account_quota`` and
``fetch_credits_*`` never raise — a quota lookup must not be able to fail the
gateway's usage endpoint or a timer-driven credit sync.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from typing import Any

import httpx

from .auth import ADAL_APP_URL, clerk_id_from_token

_pkg = sys.modules[__package__]

# -- subscription quota ------------------------------------------------------
# The platform publishes live subscription state at two endpoints keyed by the
# internal user id:
#
#   GET /api/user/by-clerk-id/{clerk_id} -> {"id": <uuid>, "email", "tier", …}
#   GET /api/subscription/user/{uuid}    -> {"tier", "status",
#                                            "monthly_credits",
#                                            "credits_used_this_period",
#                                            "credits_remaining",
#                                            "current_period_end", …}
#
# Neither *requires* a bearer, but ``by-clerk-id`` is not indifferent to one:
# with an ``Authorization`` header naming a different account it answers
# 403 ``{"error_type": "auth_forbidden", "message": "Cannot query other
# users"}``.  So the lookup sends the queried account's own token, which also
# keeps working when that token is expired — identity comes from the ``sub``
# claim, not from signature freshness.  ``/api/subscription/user/{uuid}``
# ignores the bearer entirely (measured: anonymous, own and alien bearers all
# return the same 200 body).  Credits are dollar-denominated (the free tier's
# 2.0 credits is described upstream as "$2/month"), hence ``USAGE_UNIT``.
#
# A **trialing** subscription is not allowed to spend its monthly allocation.
# Measured on a live trial account (tier ``pro``, ``status: trialing``,
# ``is_trialing: true``): the subscription endpoint reports
# ``monthly_credits 80.0 / credits_remaining 60.171408`` while
# ``/api/credits/balance`` reports ``weekly_usage {spent: 19.828592,
# limit: 20.0, remaining: 0.171408}`` — the same period
# (``2026-09-02 → 2026-09-09``) and the same spend, capped at 20.0.  The
# weekly ceiling is the binding one, so for a trial the flat payload reports
# it; reporting 60 remaining when 0.17 is spendable is how a client ends up
# firing requests into a 429.

USAGE_CACHE_TTL = 30.0  # cc-switch polls on a per-minute timer
TIERS_CACHE_TTL = 600.0  # the tier catalog is effectively static
USAGE_FETCH_CONCURRENCY = 8  # parallel account lookups for large pools
USAGE_UNIT = "USD"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def fetch_account_quota(
    client: httpx.AsyncClient,
    token: str,
    *,
    app_url: str = ADAL_APP_URL,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Resolve one account's subscription credits from its JWT.

    Returns ``{"ok": True, ...}`` with the credit figures, or
    ``{"ok": False, "error": <reason>}``.  Never raises: a quota lookup must
    not be able to fail the gateway's usage endpoint.

    ``total``/``used``/``remaining`` are what the account may actually spend.
    For a **trialing** subscription that is the weekly ceiling, not the
    monthly allocation (see the module comment), which costs one extra
    ``/api/credits/balance`` call — trial accounts only.  ``limit_basis``
    names which basis was used, and the monthly figures are preserved under
    ``monthly_*`` when they differ.
    """
    if not token:
        return {"ok": False, "error": "no auth token"}
    clerk_id = clerk_id_from_token(token)
    if not clerk_id:
        return {"ok": False, "error": "token carries no clerk id"}
    # Authenticate as the account being queried.  The endpoint accepts an
    # anonymous caller, but answers 403 ``Cannot query other users`` when the
    # bearer belongs to a *different* user — so inheriting whatever
    # ``Authorization`` the shared client happens to carry would zero a pool
    # account's credits.  An expired token still resolves its own quota:
    # identity comes from the ``sub`` claim, not from signature freshness.
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        resp = await client.get(
            f"{app_url}/api/user/by-clerk-id/{clerk_id}",
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"user lookup failed: {exc}"}
    if resp.status_code != 200:
        return {"ok": False, "error": f"user lookup failed: HTTP {resp.status_code}"}
    try:
        user = resp.json()
        user_id = str(user["id"])
    except (ValueError, KeyError, TypeError):
        return {"ok": False, "error": "user lookup returned no id"}
    email = str(user.get("email") or "")
    try:
        resp = await client.get(
            f"{app_url}/api/subscription/user/{user_id}",
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": f"subscription lookup failed: {exc}",
            "email": email,
        }
    if resp.status_code == 404:
        return {"ok": False, "error": "no subscription", "email": email}
    if resp.status_code != 200:
        return {
            "ok": False,
            "error": f"subscription lookup failed: HTTP {resp.status_code}",
            "email": email,
        }
    try:
        sub = resp.json()
    except ValueError:
        return {
            "ok": False,
            "error": "subscription lookup returned non-JSON",
            "email": email,
        }
    row: dict[str, Any] = {
        "ok": True,
        "email": email,
        "tier": str(sub.get("tier") or user.get("subscription_tier") or ""),
        "status": str(sub.get("status") or ""),
        "billing_interval": str(sub.get("billing_interval") or ""),
        "total": _as_float(sub.get("monthly_credits")),
        "used": _as_float(sub.get("credits_used_this_period")),
        "remaining": _as_float(sub.get("credits_remaining")),
        "is_trialing": bool(sub.get("is_trialing")),
        "period_end": str(sub.get("current_period_end") or ""),
        "limit_basis": "monthly",
    }
    if row["is_trialing"] or row["status"] == "trialing":
        cap = await _weekly_cap(client, token, app_url=app_url, timeout=timeout)
        if cap is not None:
            # Keep the subscription figures for inspection; they are what the
            # platform *allocated*, not what this account may still spend.
            row["monthly_total"] = row["total"]
            row["monthly_used"] = row["used"]
            row["monthly_remaining"] = row["remaining"]
            row["total"] = cap["weekly_limit"]
            row["used"] = cap["weekly_spent"]
            row["remaining"] = cap["weekly_remaining"]
            row["limit_basis"] = "weekly-trial"
            if cap["resets_at"]:
                # The weekly ceiling is what resets, so that is the deadline
                # worth reporting (measured: identical to the trial's
                # current_period_end, since a trial period is one week).
                row["period_end"] = cap["resets_at"]
    return row


async def _weekly_cap(
    client: httpx.AsyncClient,
    token: str,
    *,
    app_url: str,
    timeout: float,
) -> dict[str, Any] | None:
    """The weekly spend ceiling for one account, or ``None`` if unreadable.

    ``None`` covers both a failed probe and a payload with no weekly ceiling
    (``limit`` 0 or absent), so the caller keeps the monthly figures rather
    than inventing a cap.
    """
    payload = await _pkg.fetch_credits_raw(
        client, token, app_url=app_url, timeout=timeout
    )
    snapshot = _pkg.parse_credits_balance(payload)
    if snapshot is None or snapshot["weekly_limit"] <= 0:
        return None
    return snapshot


def parse_credits_balance(payload: Any) -> dict[str, Any] | None:
    """Flatten a ``/api/credits/balance`` payload into snapshot columns.

    Returns ``None`` when the payload is not the measured shape.  The weekly
    block is authoritative for spend limits: the account's real limits
    (80.0 monthly / 20.0 weekly) differ from the tier catalogue's, and only
    this endpoint reports ``is_usage_limited``.
    """
    if not isinstance(payload, dict):
        return None
    weekly = payload.get("weekly_usage")
    weekly = weekly if isinstance(weekly, dict) else {}
    return {
        "total": _as_float(payload.get("total")),
        "monthly_allocation": _as_float(payload.get("monthly_allocation")),
        "monthly_used": _as_float(payload.get("monthly_used")),
        "weekly_spent": _as_float(weekly.get("spent")),
        "weekly_limit": _as_float(weekly.get("limit")),
        "weekly_remaining": _as_float(weekly.get("remaining")),
        "is_usage_limited": bool(weekly.get("is_usage_limited")),
        "limit_reason": str(weekly.get("limit_reason") or ""),
        "resets_at": str(weekly.get("resets_at") or ""),
    }


async def fetch_credits_raw(
    client: httpx.AsyncClient,
    token: str,
    *,
    app_url: str = ADAL_APP_URL,
    timeout: float = 8.0,
) -> dict[str, Any] | None:
    """One account's ``/api/credits/balance`` payload verbatim, or ``None``.

    The endpoint authenticates from the bearer alone and ignores
    ``?user_id=``/``?clerk_id=``; anonymous callers get 401
    ``{"error_type": "auth_required"}``.  Never raises — a credit probe must
    not be able to fail a request path.  The admin UI shows this payload as
    measured, including the percentage fields sub2api does not store.
    """
    if not token:
        return None
    try:
        resp = await client.get(
            f"{app_url}/api/credits/balance",
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


async def fetch_credits_balance(
    client: httpx.AsyncClient,
    token: str,
    *,
    app_url: str = ADAL_APP_URL,
    timeout: float = 8.0,
) -> dict[str, Any] | None:
    """Flattened credit snapshot for one account, or ``None`` on any failure."""
    payload = await _pkg.fetch_credits_raw(
        client, token, app_url=app_url, timeout=timeout
    )
    return _pkg.parse_credits_balance(payload)


def _invalid_message(
    rows: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
    parked: list[dict[str, Any]],
) -> str:
    """Explain why the gateway cannot currently spend its subscription."""
    if not rows:
        return "no adal account configured"
    if not resolved:
        errors = sorted({str(r.get("error") or "unknown error") for r in rows})
        return "; ".join(errors[:3])
    reasons = sorted({str(r["dead_reason"]) for r in parked})
    return f"all {len(rows)} account(s) parked: {', '.join(reasons)}"


def aggregate_quota(
    rows: list[dict[str, Any]],
    tiers: dict[str, Any],
    *,
    channel: str = "",
    pool_enabled: bool = False,
) -> dict[str, Any]:
    """Fold per-account quota rows into one flat cc-switch usage payload.

    The top-level keys are exactly the fields a cc-switch usage-script
    extractor reads (``isValid``, ``invalidMessage``, ``planName``, ``used``,
    ``total``, ``remaining``, ``unit``, ``extra``).  sub2api's own detail is
    nested under ``sub2api``, which cc-switch ignores.

    Credit figures sum every account whose subscription resolved, including
    parked ones — ``isValid``/``invalidMessage`` carry the "cannot spend it"
    signal instead of silently zeroing the numbers.  A trialing account
    contributes its weekly ceiling (see :func:`fetch_account_quota`), so the
    totals stay "what this gateway can actually spend"; ``extra`` says so.
    """
    resolved = [r for r in rows if r.get("ok")]
    parked = [r for r in rows if r.get("dead_reason")]
    counts = Counter(
        name
        for name in (
            _pkg.tier_display_name(tiers, str(r.get("tier") or "")) for r in resolved
        )
        if name
    )
    period_ends = sorted(str(r.get("period_end") or "") for r in resolved)
    notes: list[str] = []
    if len(resolved) == 1 and resolved[0].get("status"):
        notes.append(str(resolved[0]["status"]))
    elif pool_enabled:
        notes.append(f"{len(resolved)}/{len(rows)} accounts")
    trials = sum(1 for r in resolved if r.get("limit_basis") == "weekly-trial")
    if trials:
        notes.append(
            "trial weekly cap"
            if trials == len(resolved)
            else f"{trials}/{len(resolved)} on trial weekly cap"
        )
    if parked:
        reasons = sorted({str(r["dead_reason"]) for r in parked})
        notes.append(f"{len(parked)} parked ({', '.join(reasons)})")
    next_reset = next((end for end in period_ends if end), "")
    if next_reset:
        notes.append(f"resets {next_reset}")
    is_valid = bool(resolved) and len(parked) < len(rows)
    payload: dict[str, Any] = {
        "isValid": is_valid,
        "planName": ", ".join(
            name if count == 1 else f"{name} x{count}"
            for name, count in sorted(counts.items())
        ),
        "total": round(sum(_as_float(r.get("total")) for r in resolved), 6),
        "used": round(sum(_as_float(r.get("used")) for r in resolved), 6),
        "remaining": round(sum(_as_float(r.get("remaining")) for r in resolved), 6),
        "unit": USAGE_UNIT,
        "extra": " · ".join(notes),
    }
    if not is_valid:
        payload["invalidMessage"] = _invalid_message(rows, resolved, parked)
    payload["sub2api"] = {
        "channel": channel,
        "pool_enabled": pool_enabled,
        "accounts": len(rows),
        "accounts_resolved": len(resolved),
        "accounts_parked": len(parked),
        "queried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detail": rows,
    }
    return payload
