"""`sub2api` command-line entrypoint: starts the HTTP gateway."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from .core.config import AppSettings
from .server.app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sub2api", description=__doc__)
    parser.add_argument(
        "--host",
        default=None,
        help="bind host (default from SUB2API_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="bind port (default from SUB2API_PORT or 8080)",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="channel to activate (default from SUB2API_CHANNEL or echo)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="mount the /admin management UI (also SUB2API_WEB=1)",
    )
    parser.add_argument(
        "--responses-only",
        action="store_true",
        help="use Responses upstream, bridging Chat Completions (also SUB2API_RESPONSES_ONLY=1)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite store path (default from SUB2API_DB or ~/.adal/sub2api.sqlite3)",
    )
    args = parser.parse_args(argv)

    settings = AppSettings.from_env()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.channel:
        settings.channel = args.channel
    if args.web:
        settings.web = True
    if args.responses_only:
        settings.responses_only = True
    if args.db:
        settings.db = args.db
        # default_db_path() reads SUB2API_DB at call time, and the pool loader
        # takes no arguments (its zero-arg signature is pinned by tests that
        # monkeypatch it).  Exporting the flag is what makes --db one source of
        # truth for the metering store AND for accounts_from_db/pool_settings.
        os.environ["SUB2API_DB"] = args.db

    # Default-deny: an unkeyed gateway hands anyone who can reach the port a
    # working subscription, and with --web their credentials too. This lives
    # here rather than in create_app because building an app object (tests,
    # embedding) is not the same as exposing a listening socket.
    if not settings.api_key and os.environ.get("SUB2API_ALLOW_ANONYMOUS") != "1":
        print(
            "sub2api: refusing to start without SUB2API_API_KEY"
            " (set SUB2API_ALLOW_ANONYMOUS=1 to opt out)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
