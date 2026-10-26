"""The orchestrator's side of the inference pipe. No networking code here.

``InferenceClient`` runs the inference worker (``asif.infer.worker``) as a
child process and speaks its JSON Lines protocol over the worker's stdin and
stdout; the protocol is documented in the worker module. The worker is
untrusted by design: every line it sends is checked, a line that doesn't
answer the request that was just sent kills it (an answer must never be
attributed to the wrong photo), and the model's text is handed back raw for
``asif.infer.schema`` to validate.

This module must never import ``socket``, ``http``, ``urllib`` or anything
that does: CI imports the whole orchestrator and fails if one shows up.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal

SLACK_S = 30.0
"""Extra wait on top of the worker's own load and request timeouts."""
SHUTDOWN_GRACE_S = 8.0
"""How long ``close()`` waits for a clean exit before terminating the worker."""
TERM_GRACE_S = 5.0
"""How long it then waits after SIGTERM before SIGKILL."""
STDERR_LINES = 200
DETAIL_LINES = 40
"""How much of the worker's stderr goes into an ``InferenceError`` detail."""

# Python's -I: the current directory isn't put on sys.path (so a stray
# ``asif/`` or ``json.py`` next to your photos can't be imported in its
# place) and PYTHON* environment variables are ignored.
DEFAULT_COMMAND = (sys.executable, "-I", "-m", "asif.infer.worker")

_RESULT_ERRORS = frozenset({"image_decode", "server", "timeout", "bad_request"})
_MAX_LINE = 4 << 20
_STDERR_LINE_MAX = 4096
_SHOWN_LINE_MAX = 500


class InferenceError(RuntimeError):
    """The worker couldn't describe a photo, or can't run at all.

    ``kind`` is one of ``start`` (it never became ready), ``image_decode``,
    ``server``, ``timeout``, ``bad_request`` (the worker's own per-request
    errors), ``protocol`` (it said something unexpected and was killed) or
    ``dead`` (it isn't running). ``detail`` is a plain-language explanation,
    often with the tail of the worker's log.
    """

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class WorkerConfig:
    backend: Literal["llama", "stub"]
    lang: str
    max_edge_px: int
    llama_server: str = "llama-server"
    model: str = ""
    mmproj: str = ""
    ctx: int = 4096
    threads: int = 0
    gpu_layers: int | None = None
    server_args: tuple[str, ...] = ()
    load_timeout_s: int = 300
    request_timeout_s: int = 600
    command: tuple[str, ...] | None = field(default=None)
    """Override the worker command (default: this Python, ``-I -m asif.infer.worker``)."""

    def to_message(self) -> dict[str, object]:
        """The ``config`` line sent to the worker, by this process (its parent)."""
        return {
            "type": "config",
            "backend": self.backend,
            "lang": self.lang,
            "max_edge_px": self.max_edge_px,
            "llama_server": self.llama_server,
            "model": self.model,
            "mmproj": self.mmproj,
            "ctx": self.ctx,
            "threads": self.threads,
            "gpu_layers": self.gpu_layers,
            "server_args": list(self.server_args),
            "load_timeout_s": self.load_timeout_s,
            "request_timeout_s": self.request_timeout_s,
            # The worker arms its own parent-death signal, then checks we're
            # still its parent, in case we died before that took effect.
            "parent_pid": os.getpid(),
        }


class _StderrTail:
    """The worker's stderr, drained by a thread into a ring buffer.

    Draining continuously means a worker that spews can never fill the pipe
    and block (which would deadlock it against us); the ring buffer keeps
    memory bounded.
    """

    def __init__(self) -> None:
        self._lines: collections.deque[str] = collections.deque(maxlen=STDERR_LINES)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def follow(self, stream: IO[bytes]) -> None:
        self._thread = threading.Thread(
            target=self._pump, args=(stream,), name="asif-worker-stderr", daemon=True
        )
        self._thread.start()

    def _pump(self, stream: IO[bytes]) -> None:
        # This thread owns the stream: closing it from another thread while a
        # read is blocked would hang on the reader's lock.
        with stream:
            while chunk := stream.readline(_STDERR_LINE_MAX):
                line = chunk.decode("utf-8", "replace").rstrip()[:_SHOWN_LINE_MAX]
                with self._lock:
                    self._lines.append(line)

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)


class InferenceClient:
    """Runs one inference worker and asks it to describe photos, one at a time.

    Not thread-safe. Use as a context manager (``with InferenceClient(cfg) as
    client:`` starts it) or call ``start()`` and ``close()`` yourself.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen[bytes] | None = None
        self._selector: selectors.BaseSelector | None = None
        self._buf = bytearray()
        self._stderr = _StderrTail()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Spawn the worker, configure it and wait until it's ready.

        Returns the worker's ``ready`` message. Raises ``InferenceError``
        with kind ``start`` if it reports a fatal error, exits, says
        something unexpected, or isn't ready within ``load_timeout_s`` plus
        ``SLACK_S`` seconds.

        Call it from the main thread: on Linux the worker (and llama-server
        with it) gets SIGTERM when the thread that started it ends, which is
        what stops it if this process is SIGKILLed. Nothing runs in the child
        before exec (no ``preexec_fn``), so other threads may be running.
        """
        if self._proc is not None or self._closed:
            raise RuntimeError("an InferenceClient can only be started once")
        command = list(self.config.command or DEFAULT_COMMAND)
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise InferenceError(
                "start", f"the inference worker could not be run: {exc.strerror or exc}"
            ) from exc
        assert proc.stdout is not None and proc.stderr is not None
        self._proc = proc
        self._stderr.follow(proc.stderr)
        self._selector = selectors.DefaultSelector()
        self._selector.register(proc.stdout, selectors.EVENT_READ)

        deadline = time.monotonic() + self.config.load_timeout_s + SLACK_S
        try:
            self._send(self.config.to_message())
            msg = self._receive(deadline)
        except InferenceError as exc:
            self._kill()
            detail = exc.detail
        else:
            if msg.get("type") == "ready":
                return msg
            if msg.get("type") == "fatal":
                self._reap()
                detail = _fatal_detail(msg)
            else:
                self._kill()
                detail = "the inference worker sent something other than 'ready'"
        detail = self._with_stderr(detail)
        self.close()
        raise InferenceError("start", detail)

    @property
    def alive(self) -> bool:
        return not self._closed and self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        """Ask the worker to stop, then terminate or kill it if it doesn't. Idempotent."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                with contextlib.suppress(InferenceError):
                    self._send({"type": "shutdown"})
                with contextlib.suppress(OSError):
                    if proc.stdin is not None:
                        proc.stdin.close()  # EOF means "shut down" too
                self._stop(proc, SHUTDOWN_GRACE_S)
        finally:
            self._release(proc)

    def __enter__(self) -> InferenceClient:
        if self._proc is None:
            self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- requests ----------------------------------------------------------

    def describe(self, image: Path, img_id: str, attempt: int = 0) -> str:
        """Ask for a description of ``image``; returns the model's raw text.

        ``attempt`` > 0 asks for a retry that may differ. Raises
        ``InferenceError`` with the worker's error kind; ``timeout`` (and the
        worker is killed) if there is no answer within ``request_timeout_s``
        plus ``SLACK_S`` seconds; ``protocol`` (worker killed) if the answer
        is malformed or for another request; ``dead`` if the worker isn't
        running.
        """
        if not self.alive:
            raise InferenceError("dead", self._with_stderr("the inference worker isn't running"))
        request = {
            "type": "describe",
            "id": img_id,
            "image": os.path.abspath(image),
            "attempt": attempt,
        }
        deadline = time.monotonic() + self.config.request_timeout_s + SLACK_S
        try:
            self._send(request)
            msg = self._receive(deadline)
        except InferenceError as exc:
            if exc.kind == "dead":
                self._reap()
            else:  # timeout or protocol: whatever it says next can't be trusted
                self._kill()
            raise InferenceError(exc.kind, self._with_stderr(exc.detail)) from None

        if msg.get("type") == "fatal":
            self._reap()
            raise InferenceError("dead", self._with_stderr(_fatal_detail(msg)))
        if msg.get("type") != "result" or msg.get("id") != img_id:
            self._kill()
            raise InferenceError("protocol", "the inference worker answered a different request")
        ok, raw, error = msg.get("ok"), msg.get("raw"), msg.get("error")
        if ok is True and isinstance(raw, str):
            return raw
        if ok is False and error in _RESULT_ERRORS:
            detail = msg.get("detail")
            raise InferenceError(str(error), detail if isinstance(detail, str) else "")
        self._kill()
        raise InferenceError("protocol", "the inference worker sent a malformed result")

    def stderr_tail(self) -> list[str]:
        """The last lines (up to ``STDERR_LINES``) the worker wrote to stderr."""
        return self._stderr.lines()

    # -- plumbing ----------------------------------------------------------

    def _send(self, msg: dict[str, Any]) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        line = json.dumps(msg, ensure_ascii=True) + "\n"
        try:
            proc.stdin.write(line.encode("ascii"))
            proc.stdin.flush()
        except OSError as exc:  # BrokenPipeError and friends: it has exited
            raise InferenceError("dead", "the inference worker exited") from exc

    def _receive(self, deadline: float) -> dict[str, Any]:
        line = self._read_line(deadline)
        try:
            msg = json.loads(line)
        except ValueError:
            msg = None
        if not isinstance(msg, dict):
            raise InferenceError("protocol", "the inference worker sent a line that isn't JSON")
        return msg

    def _read_line(self, deadline: float) -> bytes:
        proc, selector = self._proc, self._selector
        assert proc is not None and proc.stdout is not None and selector is not None
        # Raw reads on the fd, never through the buffered file object, so
        # select() and our own buffer always agree on what's been read.
        fd = proc.stdout.fileno()
        while (end := self._buf.find(b"\n")) < 0:
            if len(self._buf) > _MAX_LINE:
                raise InferenceError("protocol", "the inference worker sent an over-long line")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InferenceError("timeout", "no answer from the inference worker in time")
            if not selector.select(remaining):
                continue
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                self._stop(proc, TERM_GRACE_S)  # it closed stdout, so it's on its way out
                raise InferenceError(
                    "dead", f"the inference worker exited (code {proc.returncode})"
                )
            self._buf += chunk
        line = bytes(self._buf[:end])
        del self._buf[: end + 1]
        return line

    def _with_stderr(self, detail: str) -> str:
        """``detail`` plus the end of the worker's stderr, if it wrote any."""
        self._stderr.join(timeout=0.5 if self._proc and self._proc.poll() is not None else 0)
        tail = self._stderr.lines()[-DETAIL_LINES:]
        if not tail:
            return detail
        return f"{detail}\nworker stderr:\n" + "\n".join(tail)

    @staticmethod
    def _stop(proc: subprocess.Popen[bytes], grace: float) -> None:
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            proc.terminate()
        try:
            proc.wait(timeout=TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _reap(self) -> None:
        """The worker said it's exiting: give it a moment, then make sure."""
        if self._proc is not None:
            self._stop(self._proc, TERM_GRACE_S)

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()

    def _release(self, proc: subprocess.Popen[bytes]) -> None:
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        self._stderr.join(timeout=1)


def _fatal_detail(msg: dict[str, Any]) -> str:
    error, detail = msg.get("error"), msg.get("detail")
    error = error if isinstance(error, str) else "fatal"
    return f"{error}: {detail}" if isinstance(detail, str) and detail else error
