"""A tripwire that makes network use in the orchestrator fail loudly.

The orchestrator (the ``asif`` process itself) never needs the network: model
inference, transfers and model downloads each run in their own short-lived
child process. ``install()`` adds a Python audit hook that raises on any
attempt to create a socket, resolve a name or open a URL in *this* process.
Child processes are unaffected.

This is defence in depth, not a sandbox: native code can bypass audit hooks.
The real walls are the per-stage sandboxes (M3). What this buys is that a
future bug or dependency that quietly reaches for the network in the
orchestrator crashes instead of succeeding. CI separately checks that the
orchestrator never even imports a networking module (tests/test_netguard.py).
"""

from __future__ import annotations

import sys
from typing import Any

DENIED_EVENTS = frozenset(
    {
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "socket.sendto",
        "socket.sendmsg",
        "urllib.Request",
        "http.client.connect",
        "ftplib.connect",
        "smtplib.connect",
        "poplib.connect",
        "imaplib.open",
        "webbrowser.open",
    }
)


class NetworkDenied(PermissionError):
    """Raised when orchestrator code tries to use the network."""


_installed = False


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if event in DENIED_EVENTS:
        raise NetworkDenied(
            f"As If's orchestrator has no network capability (blocked {event!r}). "
            "This is a bug; please report it."
        )


def install() -> None:
    """Install the tripwire for the rest of this process. Idempotent; can't be undone."""
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


def installed() -> bool:
    return _installed
