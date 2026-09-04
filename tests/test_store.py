"""Store tests: accounts round-trip, usage metering, credit snapshots.

Every database lives under ``tmp_path`` — never the real ``~/.adal``.
"""

from __future__ import annotations

import json
import time
import asyncio
from pathlib import Path

import pytest

from sub2api.core.store import Store, accounts_from_db, default_db_path

# -- fixtures -----------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Store:
    store = Store(tmp_path / "sub2api.sqlite3")
    yield store
    # close() is async; run it on a fresh loop for the sync fixture teardown
    asyncio.run(store.close())


@pytest.fixture
def payload() -> dict:
    return {
        "strategy": "least-connections",
        "max_failures": 5,
        "cooldown_seconds": 90.5,
        "accounts": [
            {
                "token": "tok-1",
                "session_id": "sess-1",
                "max_concurrent": 8,
                "cookies": [
                    {
                        "name": "__session",
                        "value": "abc",
                        "domain": "adal.sylph.ai",
                        "path": "/",
                    },
                    {
                        "name": "__cf",
                        "value": "def",
                        "domain": "adal.sylph.ai",
                        "path": "/",
                    },
                ],
                "email": "a@example.com",
            },
            {
                "token": "tok-2",
                "session_id": "sess-2",
                "max_concurrent": 2,
                "cookies": [
                    {
                        "name": "__session",
                        "value": "xyz",
                        "domain": "adal.sylph.ai",
                        "path": "/",
                    },
                ],
                "email": "b@example.com",
            },
        ],
    }


# -- import/export round trip -------------------------------------------------


@pytest.mark.anyio
async def test_import_export_round_trip_matches_original(db: Store, payload: dict):
    result = await db.import_accounts(payload)
    assert result == {"imported": 2, "skipped": 0}
    exported = await db.export_accounts()

    # account entries carry exactly the accounts.json keys
    assert set(exported["accounts"][0]) == {
        "token",
        "session_id",
        "max_concurrent",
        "cookies",
        "email",
    }
    assert json.loads(json.dumps(exported)) == json.loads(json.dumps(payload))


@pytest.mark.anyio
async def test_export_defaults_when_pool_settings_never_set(db: Store):
    await db.upsert_account(session_id="s", token="t")
    exported = await db.export_accounts()
    assert exported["strategy"] == "round-robin"
    assert exported["max_failures"] == 3
    assert exported["cooldown_seconds"] == 60.0


@pytest.mark.anyio
async def test_import_skips_entries_without_token(db: Store):
    payload = {
        "strategy": "round-robin",
        "max_failures": 3,
        "cooldown_seconds": 60.0,
        "accounts": [
            {"session_id": "s1", "token": "t1"},
            {"session_id": "s2"},  # no token
            {"token": "t3"},  # no session_id
        ],
    }
    result = await db.import_accounts(payload)
    assert result == {"imported": 1, "skipped": 2}
    rows = await db.accounts()
    assert [r["session_id"] for r in rows] == ["s1"]


# -- accounts CRUD ------------------------------------------------------------


@pytest.mark.anyio
async def test_upsert_preserves_created_at_and_advances_updated_at(db: Store):
    await db.upsert_account(session_id="s1", token="t1", email="first@x.com")
    before = (await db.accounts())[0]
    assert before["status"] == "unknown"

    await asyncio.sleep(1.1)  # ensure updated_at strictly advances
    await db.upsert_account(session_id="s1", token="t2", email="second@x.com")
    rows = await db.accounts()
    assert len(rows) == 1
    after = rows[0]
    assert after["created_at"] == before["created_at"]
    assert after["updated_at"] > before["updated_at"]
    assert after["token"] == "t2"
    assert after["email"] == "second@x.com"


@pytest.mark.anyio
async def test_upsert_none_status_leaves_existing_value(db: Store):
    await db.upsert_account(
        session_id="s1", token="t1", status="ok", reason="ok", detail="fine"
    )
    await db.upsert_account(session_id="s1", token="t2")
    row = (await db.accounts())[0]
    assert row["status"] == "ok"
    assert row["reason"] == "ok"
    assert row["detail"] == "fine"


@pytest.mark.anyio
async def test_delete_account_returns_true_then_false(db: Store):
    await db.upsert_account(session_id="s1", token="t1")
    assert await db.delete_account("s1") is True
    assert await db.delete_account("s1") is False


@pytest.mark.anyio
async def test_accounts_decodes_cookies_and_tolerates_malformed(db: Store):
    await db.upsert_account(
        session_id="s1", token="t1", cookies=[{"name": "c", "value": "v"}]
    )
    # write a malformed cookies string directly
    import sqlite3

    def corrupt():
        conn = sqlite3.connect(str(db._path))
        conn.execute(
            "UPDATE accounts SET cookies = ? WHERE session_id = 's1'", ("{not json",)
        )
        conn.commit()
        conn.close()

    await asyncio.to_thread(corrupt)
    rows = await db.accounts()
    assert rows[0]["cookies"] == []


# -- usage metering -----------------------------------------------------------


@pytest.mark.anyio
async def test_usage_totals_sums_and_respects_since(db: Store):
    base = time.time() - 100
    await db.record_usage(
        ts=base,
        model="m",
        provider="p",
        path="/v1/messages",
        input_tokens=100,
        output_tokens=10,
        cost_usd=0.010000,
    )
    await db.record_usage(
        ts=base + 10,
        model="m",
        provider="p",
        path="/v1/messages",
        cache_read_tokens=50,
        cache_write_tokens=20,
        reasoning_tokens=5,
        cost_usd=0.002500,
    )
    await db.record_usage(
        ts=base + 20,
        model="m",
        provider="p",
        path="/v1/messages",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.000001,
    )

    totals = await db.usage_totals(since=0)
    assert totals["requests"] == 3
    assert totals["tokens"] == {
        "input": 101,
        "output": 11,
        "cache_read": 50,
        "cache_write": 20,
        "reasoning": 5,
    }
    assert totals["cost_usd"] == 0.012501

    future = await db.usage_totals(since=base + 1000)
    assert future["requests"] == 0
    assert future["tokens"]["input"] == 0
    assert future["cost_usd"] == 0.0


@pytest.mark.anyio
async def test_usage_by_session_bills_each_account_separately(db: Store):
    base = time.time() - 100
    for sid, cost, n in (("sess_a", 0.0100, 3), ("sess_b", 0.0002, 1)):
        for i in range(n):
            await db.record_usage(
                ts=base + i,
                session_id=sid,
                model="m",
                provider="p",
                path="/v1/messages",
                input_tokens=10,
                output_tokens=1,
                cost_usd=cost / n,
            )
    # A row with no session (single-account mode before pooling) is not an
    # account and must not create a bogus entry.
    await db.record_usage(
        ts=base, model="m", provider="p", path="/v1/messages", cost_usd=9.9
    )

    by_session = await db.usage_by_session(since=0)
    assert set(by_session) == {"sess_a", "sess_b"}
    assert by_session["sess_a"]["requests"] == 3
    assert by_session["sess_a"]["cost_usd"] == pytest.approx(0.0100)
    assert by_session["sess_b"]["requests"] == 1
    assert by_session["sess_b"]["input_tokens"] == 10
    # Ordered by spend, most expensive first.
    assert list(by_session) == ["sess_a", "sess_b"]

    later = await db.usage_by_session(since=base + 1000)
    assert later == {}


@pytest.mark.anyio
async def test_usage_by_session_empty_without_traffic(db: Store):
    assert await db.usage_by_session(since=0) == {}


# -- credit snapshots ---------------------------------------------------------


@pytest.mark.anyio
async def test_record_credits_latest_credits_returns_newest_per_session(db: Store):
    await db.record_credits("s1", weekly_spent=1.0, ts=time.time() - 50)
    await db.record_credits("s1", weekly_spent=2.0, ts=time.time())
    await db.record_credits("s2", weekly_spent=9.9, ts=time.time() - 5)

    latest = await db.latest_credits()
    by_session = {r["session_id"]: r for r in latest}
    assert len(latest) == 2
    assert by_session["s1"]["weekly_spent"] == 2.0
    assert by_session["s2"]["weekly_spent"] == 9.9


# -- pool helper --------------------------------------------------------------


def test_accounts_from_db_missing_file_is_none_and_creates_nothing(tmp_path: Path):
    missing = tmp_path / "nonexistent.sqlite3"
    assert accounts_from_db(missing) is None
    assert not missing.exists()


def test_accounts_from_db_skips_dead_rows(db: Store):
    async def seed():
        await db.upsert_account(session_id="live", token="t-live", status="ok")
        await db.upsert_account(session_id="dead", token="t-dead", status="dead")

    asyncio.run(seed())
    rows = accounts_from_db(db._path)
    assert [r["session_id"] for r in rows] == ["live"]
    assert rows[0]["cookies"] == []
    assert rows[0]["status"] == "ok"


def test_accounts_from_db_none_when_all_dead(db: Store):
    async def seed():
        await db.upsert_account(session_id="dead1", token="t1", status="dead")
        await db.upsert_account(session_id="dead2", token="t2", status="dead")

    asyncio.run(seed())
    assert accounts_from_db(db._path) is None


def test_default_db_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SUB2API_DB", str(tmp_path / "override.sqlite3"))
    assert default_db_path() == tmp_path / "override.sqlite3"
    monkeypatch.delenv("SUB2API_DB")
    assert default_db_path() == Path.home() / ".adal" / "sub2api.sqlite3"
