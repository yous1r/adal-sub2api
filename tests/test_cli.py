"""Tests for the CLI entrypoint: flag plumbing and the default-deny guard.

``uvicorn.run`` is patched out — the point is what ``main`` decides, not that
a socket binds.
"""

from __future__ import annotations

import pytest

import sub2api.__main__ as cli


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "SUB2API_API_KEY",
        "SUB2API_ALLOW_ANONYMOUS",
        "SUB2API_WEB",
        "SUB2API_RESPONSES_ONLY",
        "SUB2API_DB",
        "SUB2API_CHANNEL",
        "SUB2API_HOST",
        "SUB2API_PORT",
        "SUB2API_ACCOUNTS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def captured(monkeypatch):
    """Capture the settings ``main`` would serve, without starting a server."""
    seen: dict = {}

    def fake_create_app(settings):
        seen["settings"] = settings
        return object()

    def fake_run(app, host, port):
        seen["host"] = host
        seen["port"] = port

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    return seen


def test_refuses_to_start_without_api_key(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--host", "127.0.0.1", "--port", "48099"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "refusing to start without SUB2API_API_KEY" in err
    assert "SUB2API_ALLOW_ANONYMOUS=1" in err


def test_refusal_happens_before_the_app_is_built(monkeypatch):
    built = []
    monkeypatch.setattr(cli, "create_app", lambda s: built.append(s))
    with pytest.raises(SystemExit):
        cli.main([])
    assert built == []


def test_allow_anonymous_opt_out_starts(monkeypatch, captured):
    monkeypatch.setenv("SUB2API_ALLOW_ANONYMOUS", "1")
    cli.main(["--port", "48099"])
    assert captured["settings"].api_key is None
    assert captured["port"] == 48099


def test_api_key_from_env_starts(monkeypatch, captured):
    monkeypatch.setenv("SUB2API_API_KEY", "sk-live")
    cli.main([])
    assert captured["settings"].api_key == "sk-live"


def test_flags_override_environment(monkeypatch, captured):
    monkeypatch.setenv("SUB2API_API_KEY", "sk-live")
    monkeypatch.setenv("SUB2API_HOST", "0.0.0.0")
    monkeypatch.setenv("SUB2API_PORT", "9999")
    monkeypatch.setenv("SUB2API_CHANNEL", "echo")
    cli.main(
        ["--host", "127.0.0.1", "--port", "48080", "--channel", "adal-cloud", "--web"]
    )
    settings = captured["settings"]
    assert (settings.host, settings.port) == ("127.0.0.1", 48080)
    assert settings.channel == "adal-cloud"
    assert settings.web is True
    assert captured["host"] == "127.0.0.1"


def test_web_defaults_off_and_env_can_enable(monkeypatch, captured):
    monkeypatch.setenv("SUB2API_API_KEY", "sk-live")
    cli.main([])
    assert captured["settings"].web is False
    monkeypatch.setenv("SUB2API_WEB", "1")
    cli.main([])
    assert captured["settings"].web is True


def test_db_flag_exports_env_so_pool_and_store_agree(monkeypatch, captured, tmp_path):
    monkeypatch.setenv("SUB2API_API_KEY", "sk-live")
    db = tmp_path / "cli.sqlite3"
    cli.main(["--db", str(db)])
    assert captured["settings"].db == str(db)
    # default_db_path() and the zero-arg pool loader read the env var, so the
    # flag must land there too or --db would only move the metering store.
    from sub2api.core.store import default_db_path

    assert default_db_path() == db
