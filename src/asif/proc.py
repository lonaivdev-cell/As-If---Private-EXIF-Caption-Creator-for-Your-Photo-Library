"""Process hygiene shared by the orchestrator and the inference worker.

Three things keep a photo from outliving the process that held it:

* ``disable_core_dumps``: a crash (or Ctrl+\\) would otherwise write the
  process's memory, photo pixels included, to disk, or hand it to a crash
  reporter (apport, systemd-coredump). The limit is inherited by every child
  (worker, llama-server, exiftool) and can't be raised again.
* ``install_stop_signals``: SIGTERM, SIGHUP and SIGQUIT normally end Python
  on the spot, skipping every ``finally``. Turning them into
  ``KeyboardInterrupt`` (like Ctrl+C) lets the RAM copy be wiped and the
  children closed, and a second signal can't cut that cleanup short.
* ``with_parent_death``: wraps a child's command so it gets SIGTERM if we
  are SIGKILLed, which no handler can catch (Linux only). It works however
  many threads we have, unlike ``die_with_parent``, a ``preexec_fn`` that is
  only safe to use before any thread starts.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from types import FrameType, TracebackType

_PR_SET_PDEATHSIG = 1
_PR_SET_DUMPABLE = 4

# Runs in the child after fork: arm the death signal, make sure the parent
# didn't die before that took effect, then become the real program.
# PR_SET_PDEATHSIG survives execve (for programs that aren't setuid).
_TRAMPOLINE = (
    "import ctypes, os, sys\n"
    "ctypes.CDLL(None, use_errno=True).prctl(1, 15, 0, 0, 0)\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    os._exit(1)\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)
_STOP_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")
    if hasattr(signal, name)
)


def disable_core_dumps(*, this_process: bool = True) -> bool:
    """Keep crash dumps (and the photo pixels in them) off disk. False if it couldn't.

    The core limit becomes 1 byte, soft and hard: too small for a core file,
    and the kernel's marker for "don't pipe a dump to a crash reporter
    either" (a limit of 0 only stops core files). If the hard limit is
    already 0, it stays 0. Both are inherited and can't be raised again.
    On Linux the dump filter is also emptied (inherited too), so a dump that
    still happens carries no memory contents; with ``this_process``, this
    process is also made non-dumpable, which additionally stops other
    programs of the same user from reading its memory.
    """
    try:
        import resource
    except ImportError:  # not on POSIX
        return False
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (1, 1))
    except (ValueError, OSError):
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            return False
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError), open("/proc/self/coredump_filter", "w") as fh:
            fh.write("0")
        if this_process:
            with contextlib.suppress(OSError, AttributeError):
                ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)
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


def with_parent_death(argv: list[str]) -> list[str]:
    """``argv``, wrapped so the child gets SIGTERM when this process dies.

    Linux only (elsewhere ``argv`` comes back as is). A tiny Python
    trampoline sets the death signal after fork and then execs the real
    program, so nothing runs between fork and exec in this process, which is
    what makes it safe with other threads running.
    """
    if not sys.platform.startswith("linux"):
        return argv
    return [sys.executable, "-I", "-c", _TRAMPOLINE, str(os.getpid()), *argv]


@contextlib.contextmanager
def shielded() -> Iterator[None]:
    """Let the block finish even if a stop signal arrives; stop right after.

    For cleanup that must not be cut short half-way (wiping the RAM work
    dir). A signal that arrives meanwhile is remembered and acted on, as a
    ``KeyboardInterrupt``, once the block is done. Main thread only;
    elsewhere it does nothing special.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending: list[int] = []
    previous: dict[int, Callable[[int, FrameType | None], object] | int | None] = {}

    def remember(signum: int, frame: FrameType | None) -> None:
        pending.append(signum)

    for sig in _STOP_SIGNALS:
        before = signal.getsignal(sig)
        if before is not signal.SIG_IGN:
            previous[sig] = before
            signal.signal(sig, remember)
    try:
        yield
    finally:
        for sig, before in previous.items():
            signal.signal(sig, before)
    if pending:
        _stop(pending[0], None)


class ShieldedExitStack(contextlib.ExitStack):
    """An ``ExitStack`` whose unwinding can't be cut short by a stop signal."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        with shielded():
            suppressed = super().__exit__(exc_type, exc, tb)
        return bool(suppressed)


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
