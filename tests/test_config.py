from __future__ import annotations

from sub2api.core.config import AppSettings


def test_from_env_defaults():
    settings = AppSettings.from_env(environ={})
    assert settings.channel == "echo"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8080
    assert settings.channel_config.workspace == "."
    assert settings.channel_config.auth_token is None
    assert settings.channel_config.options == {}


def test_from_env_overrides():
    settings = AppSettings.from_env(
        environ={
            "SUB2API_CHANNEL": "adal-cli",
            "SUB2API_HOST": "0.0.0.0",
            "SUB2API_PORT": "9000",
            "SUB2API_WORKSPACE": "/work",
            "SUB2API_AUTH_TOKEN": "jwt",
            "SUB2API_RUNTIME_PATH": "/usr/bin/adal",
            "SUB2API_CHANNEL_OPTIONS": '{"timeout": 30}',
        }
    )
    assert settings.channel == "adal-cli"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    cfg = settings.channel_config
    assert cfg.workspace == "/work"
    assert cfg.options == {"timeout": 30}


def test_from_env_openai_and_guard_fields():
    settings = AppSettings.from_env(environ={})
    assert settings.api_key is None
    assert settings.openai_permission_mode == "yolo"
    assert settings.enabled_tools is None

    settings = AppSettings.from_env(
        environ={
            "SUB2API_API_KEY": "sk-x",
            "SUB2API_OPENAI_PERMISSION_MODE": "default",
            "SUB2API_ENABLED_TOOLS": "Read, Search",
        }
    )
    assert settings.api_key == "sk-x"
    assert settings.openai_permission_mode == "default"
    assert settings.enabled_tools == ("Read", "Search")
