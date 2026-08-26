"""`sub2api` command-line entrypoint: starts the HTTP gateway."""

from __future__ import annotations

import argparse

import uvicorn

from .core.config import AppSettings
from .server.app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sub2api", description=__doc__)
    parser.add_argument("--host", default=None, help="bind host (default from SUB2API_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default from SUB2API_PORT or 8080)")
    parser.add_argument("--channel", default=None, help="channel to activate (default from SUB2API_CHANNEL or echo)")
    args = parser.parse_args(argv)

    settings = AppSettings.from_env()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.channel:
        settings.channel = args.channel

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
