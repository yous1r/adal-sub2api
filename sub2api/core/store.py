"""Durable storage for accounts, usage events, and credit snapshots.

SQLite (stdlib ``sqlite3`` only — every blocking call runs in a worker thread
via ``asyncio.to_thread``) because the pool, the admin UI, and the usage
metering must all observe the same rows across server restarts, and the
registrar already writes the same ``accounts`` schema to
``~/.adal/registrar.sqlite3``: mirroring it verbatim lets registrar-produced
rows import 1:1.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    session_id TEXT PRIMARY KEY, token TEXT NOT NULL, max_concurrent INTEGER NOT NULL,
    cookies TEXT NOT NULL, email TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'unknown', reason TEXT NOT NULL DEFAULT 'not_checked',
    detail TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, session_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL, provider TEXT NOT NULL, path TEXT NOT NULL, stream INTEGER NOT NULL,
    status INTEGER NOT NULL, latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0.0,
    rate_estimated INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS usage_events_ts ON usage_events(ts);
CREATE TABLE IF NOT EXISTS credit_snapshots (
    session_id TEXT NOT NULL, ts REAL NOT NULL, total REAL, monthly_allocation REAL,
    monthly_used REAL, weekly_spent REAL, weekly_limit REAL, weekly_remaining REAL,
    is_usage_limited INTEGER NOT NULL DEFAULT 0, limit_reason TEXT NOT NULL DEFAULT '',
    resets_at TEXT NOT NULL DEFAULT '', PRIMARY KEY (session_id, ts));
CREATE TABLE IF NOT EXISTS pool_settings (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS catalog_cache (k TEXT PRIMARY KEY, v TEXT NOT NULL, updated_at REAL NOT NULL);
"""

_USAGE_COLUMNS = (
    "ts",
    "session_id",
    "model",
    "provider",
    "path",
    "stream",
    "status",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "cost_usd",
    "rate_estimated",
)

_INT_USAGE_KEYS = frozenset(
    {
        "stream",
        "status",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "rate_estimated",
    }
)

_CREDIT_FLOAT_KEYS = (
    "total",
    "monthly_allocation",
    "monthly_used",
    "weekly_spent",
    "weekly_limit",
    "weekly_remaining",
)


def default_db_path() -> Path:
    """``$SUB2API_DB`` read at call time, else ``~/.adal/sub2api.sqlite3``."""
    override = os.environ.get("SUB2API_DB")
    if override:
        return Path(override)
    return Path.home() / ".adal" / "sub2api.sqlite3"


def _decode_cookies(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def accounts_from_db(path: Path | str | None = None) -> list[dict] | None:
    """Sync: account rows (``cookies`` decoded, plus ``status``) for the pool.

    Read-only — never creates the database. Returns ``None`` when the file is
    missing, unreadable, lacks the ``accounts`` table, or has no live row.
    """
    target = Path(path) if path is not None else default_db_path()
    if not target.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT session_id, token, max_concurrent, cookies, email, status"
                " FROM accounts WHERE status != 'dead'"
                " ORDER BY created_at, session_id"
            ).fetchall()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None
    if not rows:
        return None
    return [
        {
            "session_id": row["session_id"],
            "token": row["token"],
            "max_concurrent": row["max_concurrent"],
            "cookies": _decode_cookies(row["cookies"]),
            "email": row["email"],
            "status": row["status"],
        }
        for row in rows
    ]


def pool_settings_from_db(
    path: Path | str | None = None,
) -> tuple[str, int, float]:
    """Sync: ``(strategy, max_failures, cooldown_seconds)`` from the store.

    Read-only, never creates the database, and falls back to the pool's own
    defaults for every key the store does not hold — so a DB written by an
    older build still loads.
    """
    defaults = ("round-robin", 3, 60.0)
    target = Path(path) if path is not None else default_db_path()
    if not target.exists():
        return defaults
    try:
        conn = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT k, v FROM pool_settings").fetchall()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return defaults
    settings = {r["k"]: r["v"] for r in rows}
    try:
        return (
            settings.get("strategy") or defaults[0],
            int(settings.get("max_failures", defaults[1])),
            float(settings.get("cooldown_seconds", defaults[2])),
        )
    except (TypeError, ValueError):
        return defaults


class Store:
    """Async facade over one shared SQLite connection."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = _open(self._path)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- accounts ------------------------------------------------------------

    async def accounts(self) -> list[dict]:
        def run() -> list[dict]:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM accounts ORDER BY created_at, session_id"
                ).fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "token": r["token"],
                    "max_concurrent": r["max_concurrent"],
                    "cookies": _decode_cookies(r["cookies"]),
                    "email": r["email"],
                    "status": r["status"],
                    "reason": r["reason"],
                    "detail": r["detail"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

        return await asyncio.to_thread(run)

    async def upsert_account(
        self,
        session_id: str,
        token: str,
        cookies: list[dict] | None = None,
        email: str = "",
        max_concurrent: int = 4,
        status: str | None = None,
        reason: str | None = None,
        detail: str | None = None,
    ) -> None:
        now = time.time()
        cookies_json = json.dumps(cookies or [])

        def run() -> None:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO accounts
                        (session_id, token, max_concurrent, cookies, email,
                         status, reason, detail, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, COALESCE(?, 'unknown'),
                            COALESCE(?, 'not_checked'), COALESCE(?, ''), ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        token = excluded.token,
                        max_concurrent = excluded.max_concurrent,
                        cookies = excluded.cookies,
                        email = excluded.email,
                        status = COALESCE(?, accounts.status),
                        reason = COALESCE(?, accounts.reason),
                        detail = COALESCE(?, accounts.detail),
                        updated_at = excluded.updated_at
                    """,
                    (
                        session_id,
                        token,
                        max_concurrent,
                        cookies_json,
                        email,
                        status,
                        reason,
                        detail,
                        now,
                        now,
                        status,
                        reason,
                        detail,
                    ),
                )
                self._conn.commit()

        await asyncio.to_thread(run)

    async def delete_account(self, session_id: str) -> bool:
        def run() -> bool:
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM accounts WHERE session_id = ?", (session_id,)
                )
                self._conn.commit()
            return cur.rowcount > 0

        return await asyncio.to_thread(run)

    async def import_accounts(self, payload: dict) -> dict:
        accounts = payload.get("accounts") or []
        strategy = payload.get("strategy", "round-robin")
        max_failures = int(payload.get("max_failures", 3))
        cooldown_seconds = float(payload.get("cooldown_seconds", 60.0))
        imported = 0
        skipped = 0
        for entry in accounts:
            if (
                not isinstance(entry, dict)
                or not entry.get("token")
                or not entry.get("session_id")
            ):
                skipped += 1
                continue
            await self.upsert_account(
                session_id=entry["session_id"],
                token=entry["token"],
                cookies=entry.get("cookies"),
                email=entry.get("email", ""),
                max_concurrent=int(entry.get("max_concurrent", 4)),
            )
            imported += 1
        await self._set_pool_settings(strategy, max_failures, cooldown_seconds)
        return {"imported": imported, "skipped": skipped}

    async def export_accounts(self) -> dict:
        strategy, max_failures, cooldown_seconds = await self._pool_settings()
        rows = await self.accounts()
        return {
            "strategy": strategy,
            "max_failures": max_failures,
            "cooldown_seconds": cooldown_seconds,
            "accounts": [
                {
                    "token": r["token"],
                    "session_id": r["session_id"],
                    "max_concurrent": r["max_concurrent"],
                    "cookies": r["cookies"],
                    "email": r["email"],
                }
                for r in rows
            ],
        }

    async def _set_pool_settings(
        self, strategy: str, max_failures: int, cooldown_seconds: float
    ) -> None:
        def run() -> None:
            with self._lock:
                for k, v in (
                    ("strategy", strategy),
                    ("max_failures", str(max_failures)),
                    ("cooldown_seconds", str(cooldown_seconds)),
                ):
                    self._conn.execute(
                        "INSERT INTO pool_settings (k, v) VALUES (?, ?)"
                        " ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                        (k, v),
                    )
                self._conn.commit()

        await asyncio.to_thread(run)

    async def _pool_settings(self) -> tuple[str, int, float]:
        def run() -> tuple[str, int, float]:
            with self._lock:
                rows = self._conn.execute("SELECT k, v FROM pool_settings").fetchall()
            settings = {r["k"]: r["v"] for r in rows}
            return (
                settings.get("strategy", "round-robin"),
                int(settings.get("max_failures", "3")),
                float(settings.get("cooldown_seconds", "60.0")),
            )

        return await asyncio.to_thread(run)

    # -- catalog -------------------------------------------------------------

    async def load_catalog(self, key: str = "adal") -> dict[str, Any] | None:
        """Last cached model catalog JSON for ``key``, or ``None``."""

        def run() -> dict[str, Any] | None:
            with self._lock:
                row = self._conn.execute(
                    "SELECT v FROM catalog_cache WHERE k = ?", (key,)
                ).fetchone()
            if row is None:
                return None
            try:
                parsed = json.loads(row["v"])
            except ValueError:
                return None
            return parsed if isinstance(parsed, dict) else None

        return await asyncio.to_thread(run)

    async def save_catalog(self, catalog: dict[str, Any], key: str = "adal") -> None:
        """Persist the model catalog JSON under ``key``."""

        def run() -> None:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO catalog_cache (k, v, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(k) DO UPDATE SET
                        v = excluded.v, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(catalog), time.time()),
                )
                self._conn.commit()

        await asyncio.to_thread(run)

    # -- usage ---------------------------------------------------------------

    async def record_usage(self, **row: Any) -> None:
        values = []
        for key in _USAGE_COLUMNS:
            raw = row.get(key)
            if key == "ts":
                values.append(time.time() if raw is None else float(raw))
            elif key in _INT_USAGE_KEYS:
                values.append(int(raw) if raw is not None else 0)
            elif key in ("cost_usd", "latency_ms"):
                values.append(float(raw) if raw is not None else 0.0)
            else:
                values.append(str(raw) if raw is not None else "")
        # Unknown keys (anything outside _USAGE_COLUMNS) are ignored.

        def run() -> None:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO usage_events ({', '.join(_USAGE_COLUMNS)})"
                    f" VALUES ({', '.join('?' for _ in _USAGE_COLUMNS)})",
                    values,
                )
                self._conn.commit()

        try:
            await asyncio.to_thread(run)
        except sqlite3.Error:
            pass  # metering must never break a response

    async def usage_totals(self, since: float) -> dict:
        def run() -> dict:
            with self._lock:
                r = self._conn.execute(
                    """
                    SELECT COUNT(*) AS requests,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read,
                           COALESCE(SUM(cache_write_tokens), 0) AS cache_write,
                           COALESCE(SUM(reasoning_tokens), 0) AS reasoning,
                           COALESCE(SUM(cost_usd), 0.0) AS cost_usd
                    FROM usage_events WHERE ts >= ?
                    """,
                    (since,),
                ).fetchone()
            return {
                "requests": int(r["requests"]),
                "tokens": {
                    "input": int(r["input_tokens"]),
                    "output": int(r["output_tokens"]),
                    "cache_read": int(r["cache_read"]),
                    "cache_write": int(r["cache_write"]),
                    "reasoning": int(r["reasoning"]),
                },
                "cost_usd": round(float(r["cost_usd"]), 6),
            }

        return await asyncio.to_thread(run)

    async def usage_by_session(self, since: float) -> dict[str, dict]:
        """Per-account local metering totals for the trailing window.

        Keyed by ``session_id`` — the pool's account identity — so the admin
        UI can bill each account for what it actually consumed, next to its
        subscription quota.  Accounts with no traffic in the window are
        simply absent; an empty dict means nothing was metered.
        """

        def run() -> dict[str, dict]:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT session_id,
                           COUNT(*) AS requests,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read,
                           COALESCE(SUM(reasoning_tokens), 0) AS reasoning,
                           COALESCE(SUM(cost_usd), 0.0) AS cost_usd
                    FROM usage_events WHERE ts >= ?
                    GROUP BY session_id ORDER BY cost_usd DESC
                    """,
                    (since,),
                ).fetchall()
            return {
                r["session_id"]: {
                    "requests": int(r["requests"]),
                    "input_tokens": int(r["input_tokens"]),
                    "output_tokens": int(r["output_tokens"]),
                    "cache_read_tokens": int(r["cache_read"]),
                    "reasoning_tokens": int(r["reasoning"]),
                    "cost_usd": round(float(r["cost_usd"]), 6),
                }
                for r in rows
                if r["session_id"]
            }

        return await asyncio.to_thread(run)

    # -- credits -------------------------------------------------------------

    async def record_credits(self, session_id: str, **fields: Any) -> None:
        ts = float(fields.get("ts") or time.time())
        floats = [
            float(fields.get(k)) if fields.get(k) is not None else None
            for k in _CREDIT_FLOAT_KEYS
        ]
        is_usage_limited = 1 if fields.get("is_usage_limited") else 0
        limit_reason = str(fields.get("limit_reason") or "")
        resets_at = str(fields.get("resets_at") or "")

        def run() -> None:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO credit_snapshots
                        (session_id, ts, total, monthly_allocation, monthly_used,
                         weekly_spent, weekly_limit, weekly_remaining,
                         is_usage_limited, limit_reason, resets_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, ts) DO UPDATE SET
                        total = excluded.total,
                        monthly_allocation = excluded.monthly_allocation,
                        monthly_used = excluded.monthly_used,
                        weekly_spent = excluded.weekly_spent,
                        weekly_limit = excluded.weekly_limit,
                        weekly_remaining = excluded.weekly_remaining,
                        is_usage_limited = excluded.is_usage_limited,
                        limit_reason = excluded.limit_reason,
                        resets_at = excluded.resets_at
                    """,
                    (
                        session_id,
                        ts,
                        *floats,
                        is_usage_limited,
                        limit_reason,
                        resets_at,
                    ),
                )
                self._conn.commit()

        await asyncio.to_thread(run)

    async def latest_credits(self) -> list[dict]:
        def run() -> list[dict]:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT c.* FROM credit_snapshots c
                    JOIN (SELECT session_id, MAX(ts) AS max_ts
                          FROM credit_snapshots GROUP BY session_id) m
                    ON c.session_id = m.session_id AND c.ts = m.max_ts
                    ORDER BY c.session_id
                    """
                ).fetchall()
            return [dict(r) for r in rows]

        return await asyncio.to_thread(run)

    async def close(self) -> None:
        def run() -> None:
            with self._lock:
                self._conn.close()

        await asyncio.to_thread(run)
