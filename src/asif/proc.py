"""Process hygiene shared by the orchestrator and the inference worker.

Three things keep a photo from outliving the process that held it:

* ``disable_core_dumps``: a crash (or Ctrl+\\) would otherwise write the
  process's memory, photo pixels included, to disk. The limit is inherited
  by every child (worker, llama-server, exiftool) and can't be raised again.
* ``install_stop_signals``: SIGTERM, SIGHUP and SIGQUIT normally end Python
  on the spot, skipping every ``finally``. Turning them into
  ``KeyboardInterrupt`` (like Ctrl+C) lets the RAM copy be wiped and the
  children closed, and a second signal can't cut that cleanup short.
* ``die_with_parent``: a ``preexec_fn`` so a child gets SIGTERM if we are
  SIGKILLed, which no handler can catch (Linux only).
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import threading
from collections.abc import Callable
from types import FrameType

_PR_SET_PDEATHSIG = 1
_STOP_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")
    if hasattr(signal, name)
)


def disable_core_dumps() -> bool:
    """Set the core-file limit to 0 (soft and hard). Returns False if it couldn't."""
    try:
        import resource
    except ImportError:  # not on POSIX
        return False
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        return False
    return True


def _stop(signum: int, frame: FrameType | None) -> None:
    # Ignore further stop signals so they can't cut the cleanup short.
    for sig in _STOP_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)
    raise KeyboardInterrupt(f"stopped by signal {signum}")


def install_stop_signals() -> None:
    """Make Ctrl+C, SIGTERM, SIGHUP and SIGQUIT unwind cleanly (main thread only).

    A signal that is already ignored stays ignored, so ``nohup asif ...``
    keeps surviving a closed terminal.
    """
    for sig in _STOP_SIGNALS:
        if signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, _stop)


def die_with_parent() -> Callable[[], None] | None:
    """A ``preexec_fn`` that makes the child get SIGTERM when its parent dies.

    Linux only; None elsewhere, and None while other threads are running,
    because a ``preexec_fn`` can deadlock the child then (CPython's warning).
    The kernel delivers the signal when the *thread* that spawned the child
    exits, so spawn from the main thread.
    """
    if not sys.platform.startswith("linux") or threading.active_count() > 1:
        return None
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    parent = os.getpid()
    sig = int(signal.SIGTERM)

    def preexec() -> None:
        prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
        if os.getppid() != parent:  # the parent died before prctl took effect
            os._exit(1)

    return preexec
