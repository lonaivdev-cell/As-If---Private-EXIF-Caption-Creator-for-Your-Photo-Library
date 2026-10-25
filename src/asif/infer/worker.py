"""The inference worker: a separate process that turns photos into JSON.

Run as ``python -m asif.infer.worker``. From M3 on it runs inside the
inference sandbox (no network namespace route out). It is the only module in
As If that imports networking code, and the only process that talks to
llama-server, over loopback.

Protocol: JSON Lines over stdin/stdout, one object per line, UTF-8. stdout
carries *only* protocol lines: the worker keeps a private copy of fd 1 for
them and points fd 1 at stderr, so a stray ``print``, a library warning or
a child's output can never corrupt the stream. Diagnostics go to stderr.

1. Orchestrator -> worker, first line::

     {"type": "config", "backend": "llama" | "stub", "lang": "pt-BR",
      "max_edge_px": 1024,
      # llama backend only:
      "llama_server": "/usr/bin/llama-server", "model": "/m/x.gguf",
      "mmproj": "/m/mmproj.gguf", "ctx": 4096, "threads": 0,
      "gpu_layers": null, "server_args": [], "load_timeout_s": 300,
      "request_timeout_s": 600}

2. Worker -> orchestrator, once ready::

     {"type": "ready", "backend": "llama", "prompt_version": "caption_v1"}

   or, if it can't start::

     {"type": "fatal", "error": "<short kind>", "detail": "<last log lines>"}

   and then exits non-zero. Kinds: ``bad_config`` (including
   ``server_args`` that would swap the model, move the address or key, or
   download something), ``missing_binary``, ``missing_model``,
   ``server_start``, ``server_exited`` (the model failed to load),
   ``load_timeout``, ``port_taken`` and ``internal``.

3. Requests, one at a time::

     {"type": "describe", "id": "i_000001", "image": "/abs/path.jpg", "attempt": 0}

   Responses, one per request, same ``id``::

     {"type": "result", "id": "i_000001", "ok": true, "raw": "<model text>", "ms": 1234}
     {"type": "result", "id": "i_000001", "ok": false,
      "error": "image_decode" | "server" | "timeout" | "bad_request", "detail": "..."}

   A line that isn't a valid request gets a ``bad_request`` result whose
   ``id`` is the request's, or null if it has none. If llama-server dies,
   the worker answers with a ``fatal`` line instead and exits non-zero: it
   can't serve anything more, and a fresh worker is the clean way back.

   ``raw`` is the model's text verbatim. The worker does *not* validate it;
   the orchestrator does (the worker is untrusted by design).
   ``attempt`` > 0 means "retry": the temperature goes from 0 to 0.3 and
   the seed is the attempt number, so a retry can differ.

4. ``{"type": "shutdown"}``, EOF on stdin or SIGTERM: stop llama-server,
   exit (0 for the first two).

Backends:

* ``stub``: no model. Deterministic, schema-valid English output derived
  from the image's average colour, brightness and shape, with the keyword
  ``stub`` always present so stub captions are easy to find and remove. For
  trying the pipeline and for tests.
* ``llama``: ``llama-server`` from llama.cpp, bound to 127.0.0.1 on a free
  port, with a random API key passed through the environment (never argv),
  so other local processes can't use it. On Linux the worker also checks
  that the process answering on that port really is its llama-server
  before sending it anything, and llama-server gets SIGTERM if the worker
  dies, however it dies. llama-server's output goes to a bounded in-memory
  ring buffer, never to our stdout; its tail goes into ``fatal`` details.
  Requests go through ``http.client``, which, unlike ``urllib``, never
  routes through a proxy from the environment.
"""

from __future__ import annotations

import base64
import collections
import colorsys
import ctypes
import http.client
import io
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO, Any, Protocol

from PIL import Image, ImageStat

from asif.infer.preprocess import ImageDecodeError, prepare_image
from asif.infer.schema import PROMPT_VERSION, caption_schema, load_prompt

LOOPBACK = "127.0.0.1"
COMPLETIONS_PATH = "/v1/chat/completions"
MAX_TOKENS = 512
RETRY_TEMPERATURE = 0.3
USER_TEXT = "Describe this photo."

LOG_LINES = 200
"""llama-server output kept in memory, in lines."""
FATAL_TAIL_LINES = 40
"""How much of that goes into a ``fatal`` detail."""
_LOG_LINE_MAX = 4096
_SHOWN_LINE_MAX = 500
_MAX_RESPONSE = 4 << 20
_HEALTH_TIMEOUT_S = 2.0
_STOP_GRACE_S = 3.0

# Flags that would undo what the worker promises: the model the orchestrator
# hashed for provenance, loopback only, our own key, and no downloads.
_FORBIDDEN_SERVER_ARGS = (
    "-m",
    "--model",
    "--mmproj",
    "--mmproj-url",
    "--host",
    "--port",
    "--api-key",
    "--api-key-file",
    "-hf",
    "-hfr",
    "--hf-repo",
    "-hff",
    "--hf-file",
    "-mu",
    "--model-url",
)
_PR_SET_PDEATHSIG = 1


class _WorkerError(Exception):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class _Fatal(_WorkerError):
    """The worker can't (or can no longer) serve. Becomes a ``fatal`` line."""


class _RequestError(_WorkerError):
    """One request failed; the worker carries on."""


@dataclass(frozen=True)
class _Config:
    backend: str
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


def _parse_config(msg: dict[str, Any]) -> _Config:
    if msg.get("type") != "config":
        raise _Fatal("bad_config", "the first line must be a config message")
    backend = msg.get("backend")
    if backend not in ("llama", "stub"):
        raise _Fatal("bad_config", "backend must be 'llama' or 'stub'")

    def integer(key: str, default: int, minimum: int) -> int:
        value = msg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise _Fatal("bad_config", f"{key} must be an integer >= {minimum}")
        return value

    def text(key: str, default: str) -> str:
        value = msg.get(key, default)
        if not isinstance(value, str):
            raise _Fatal("bad_config", f"{key} must be a string")
        return value

    gpu_layers = msg.get("gpu_layers")
    if gpu_layers is not None:
        gpu_layers = integer("gpu_layers", 0, 0)
    server_args = msg.get("server_args", [])
    if not isinstance(server_args, list) or not all(isinstance(a, str) for a in server_args):
        raise _Fatal("bad_config", "server_args must be a list of strings")
    lang = text("lang", "")
    if not lang:
        raise _Fatal("bad_config", "lang must not be empty")
    return _Config(
        backend=backend,
        lang=lang,
        max_edge_px=integer("max_edge_px", 0, 1),
        llama_server=text("llama_server", "llama-server"),
        model=text("model", ""),
        mmproj=text("mmproj", ""),
        ctx=integer("ctx", 4096, 1),
        threads=integer("threads", 0, 0),
        gpu_layers=gpu_layers,
        server_args=tuple(server_args),
        load_timeout_s=integer("load_timeout_s", 300, 1),
        request_timeout_s=integer("request_timeout_s", 600, 1),
    )


class _Backend(Protocol):
    def start(self) -> None: ...
    def describe(self, image: Path, attempt: int) -> str: ...
    def close(self) -> None: ...


# -- stub -------------------------------------------------------------------

# Twelve 30-degree hue sectors, starting at red.
_HUES = (
    "red",
    "orange",
    "yellow",
    "green",
    "green",
    "green",
    "teal",
    "blue",
    "blue",
    "purple",
    "pink",
    "red",
)


def stub_caption(jpeg: bytes) -> dict[str, Any]:
    """A deterministic, schema-valid fake answer for a prepared JPEG."""
    with Image.open(io.BytesIO(jpeg)) as im:
        width, height = im.size
        red, green, blue = ImageStat.Stat(im.convert("RGB")).mean
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    if saturation < 0.15:
        colour = "white" if value > 0.85 else "black" if value < 0.2 else "grey"
    else:
        colour = _HUES[round(hue * 12) % 12]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    tone = "dark" if luma < 85 else "bright" if luma > 170 else "mid-tone"
    shape = "wide" if width > height * 1.1 else "tall" if height > width * 1.1 else "square"
    return {
        "caption": f"Stub caption: a {tone}, {shape} picture, mostly {colour}. "
        "Not a real description.",
        "keywords": ["stub", colour, tone, shape],
        "has_text": False,
        "people_count": 0,
        "setting": "unclear",
    }


class _StubBackend:
    def __init__(self, cfg: _Config) -> None:
        self.max_edge_px = cfg.max_edge_px

    def start(self) -> None:
        pass

    def describe(self, image: Path, attempt: int) -> str:
        return json.dumps(stub_caption(prepare_image(image, self.max_edge_px)))

    def close(self) -> None:
        pass


# -- llama-server -------------------------------------------------------------


class _LogTail:
    """The last lines of a child's output, in memory only, drained by a thread.

    Draining continuously means the child can never block on a full pipe,
    however much it logs; the ring buffer means memory stays bounded.
    """

    def __init__(self, lines: int = LOG_LINES) -> None:
        self._lines: collections.deque[str] = collections.deque(maxlen=lines)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def follow(self, stream: IO[bytes]) -> None:
        self._thread = threading.Thread(
            target=self._pump, args=(stream,), name="llama-log", daemon=True
        )
        self._thread.start()

    def _pump(self, stream: IO[bytes]) -> None:
        with stream:
            while chunk := stream.readline(_LOG_LINE_MAX):
                line = chunk.decode("utf-8", "replace").rstrip()
                with self._lock:
                    self._lines.append(line)

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def tail(self, n: int = FATAL_TAIL_LINES) -> str:
        with self._lock:
            lines = list(self._lines)[-n:]
        # A verbose llama-server may log request bodies: never pass an image on.
        shown = (
            "[line hidden: it contained image data]" if "data:image" in line else line
            for line in lines
        )
        return "\n".join(line[:_SHOWN_LINE_MAX] for line in shown)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        port: int = sock.getsockname()[1]
        return port


def _owns_port(pid: int, port: int) -> bool:
    """Is every socket listening on ``port`` held by process ``pid``?

    Between our finding a free port and llama-server binding it, another
    local program could take the port and pose as llama-server to receive
    photos. On Linux this asks the kernel who really listens; elsewhere it
    can't tell and says yes (the M3 sandbox gives llama-server a private
    loopback anyway).
    """
    tables = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))
    if not tables[0].exists():
        return True
    listeners = set()
    for table in tables:
        try:
            rows = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()  # sl, local_address, rem_address, st, ..., inode
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                listeners.add(f"socket:[{fields[9]}]")
    held = set()
    fd_dir = Path(f"/proc/{pid}/fd")
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return False
    for fd in fds:
        try:
            held.add(os.readlink(fd_dir / fd))
        except OSError:
            continue
    return bool(listeners) and listeners <= held


def _die_with_parent() -> Callable[[], None] | None:
    """A ``preexec_fn`` that makes the child get SIGTERM when we die (Linux).

    The kernel sends the signal even if this worker is SIGKILLed, so
    llama-server can't outlive it. It fires when the *thread* that spawned
    the child exits; the worker only ever spawns from its main thread.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    parent = os.getpid()
    sig = int(signal.SIGTERM)

    def preexec() -> None:
        prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0)
        if os.getppid() != parent:  # the worker died before prctl took effect
            os._exit(1)

    return preexec


def _check_server_args(args: tuple[str, ...]) -> None:
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in _FORBIDDEN_SERVER_ARGS:
            raise _Fatal(
                "bad_config",
                f"server_args may not contain {flag}: As If sets the model, address and key "
                "itself, and llama-server never downloads anything.",
            )


def _server_error(data: bytes) -> str:
    """llama-server's own error message, if it sent one, on one short line."""
    try:
        message = json.loads(data)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return ""
    if not isinstance(message, str):
        return ""
    return ": " + " ".join(message.split())[:200]


class _LlamaBackend:
    def __init__(self, cfg: _Config) -> None:
        self.cfg = cfg
        self.prompt = ""
        self.key = secrets.token_urlsafe(32)
        self.port = 0
        self.proc: subprocess.Popen[bytes] | None = None
        self.log = _LogTail()

    # -- lifecycle --

    def start(self) -> None:
        cfg = self.cfg
        _check_server_args(cfg.server_args)
        self.prompt = load_prompt(cfg.lang)
        binary = shutil.which(cfg.llama_server)
        if binary is None:
            raise _Fatal(
                "missing_binary",
                f"llama-server was not found (looked for {cfg.llama_server!r}). Install "
                "llama.cpp, or set [model] llama_server to the full path of llama-server.",
            )
        for label, path in (("model", cfg.model), ("vision projector (mmproj)", cfg.mmproj)):
            if not path or not os.path.isfile(path):
                raise _Fatal("missing_model", f"the {label} file was not found: {path!r}")

        self.port = _free_port()
        argv = [binary, "-m", cfg.model, "--mmproj", cfg.mmproj]
        argv += ["--host", LOOPBACK, "--port", str(self.port), "-c", str(cfg.ctx), "-np", "1"]
        if cfg.threads > 0:
            argv += ["-t", str(cfg.threads)]
        if cfg.gpu_layers is not None:
            argv += ["-ngl", str(cfg.gpu_layers)]
        argv += cfg.server_args

        # LLAMA_ARG_* variables configure llama-server too; only ours get through.
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("LLAMA_ARG_") and k != "LLAMA_API_KEY"
        }
        env["LLAMA_API_KEY"] = env["LLAMA_ARG_API_KEY"] = self.key
        try:
            self.proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                close_fds=True,
                # preexec_fn is only unsafe with other threads running; none exist yet.
                preexec_fn=_die_with_parent(),
            )
        except OSError as exc:
            raise _Fatal(
                "server_start", f"llama-server could not be started: {exc.strerror or exc}"
            ) from exc
        assert self.proc.stdout is not None
        self.log.follow(self.proc.stdout)
        self._wait_ready()

    def _wait_ready(self) -> None:
        assert self.proc is not None
        deadline = time.monotonic() + self.cfg.load_timeout_s
        delay = 0.05
        while True:
            code = self.proc.poll()
            if code is not None:
                raise self._exited(code, "while loading the model")
            if self._health() == 200:
                self._ensure_ours()
                return
            now = time.monotonic()
            if now >= deadline:
                raise _Fatal(
                    "load_timeout",
                    f"the model didn't finish loading within {self.cfg.load_timeout_s} s. "
                    f"llama-server's last lines:\n{self.log.tail()}",
                )
            time.sleep(min(delay, deadline - now))
            delay = min(delay * 2, 0.5)

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.log.join(timeout=1)

    # -- requests --

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    def _exited(self, code: int, when: str) -> _Fatal:
        self.log.join(timeout=1)  # it's gone: let the log reader catch its last words
        return _Fatal(
            "server_exited",
            f"llama-server stopped {when} (exit code {code}). Its last lines:\n{self.log.tail()}",
        )

    def _ensure_ours(self) -> None:
        """Raise ``_Fatal`` unless our llama-server is alive and alone on its port.

        Checked before anything is sent with the key or a photo in it.
        """
        assert self.proc is not None
        code = self.proc.poll()
        if code is not None:
            raise self._exited(code, "unexpectedly")
        if not _owns_port(self.proc.pid, self.port):
            raise _Fatal(
                "port_taken",
                "another program answered on the port meant for llama-server, so As If "
                "stopped before sending it anything. Try again; if this keeps happening, "
                "something on this machine is waiting for it.",
            )

    def _health(self) -> int | None:
        # /health is public: no key, so a squatter on the port never sees it.
        conn = http.client.HTTPConnection(LOOPBACK, self.port, timeout=_HEALTH_TIMEOUT_S)
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read(_MAX_RESPONSE)
            return resp.status
        except (OSError, http.client.HTTPException):
            return None
        finally:
            conn.close()

    def request_body(self, jpeg: bytes, attempt: int) -> dict[str, Any]:
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        return {
            "messages": [
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": USER_TEXT},
                    ],
                },
            ],
            "temperature": 0.0 if attempt == 0 else RETRY_TEMPERATURE,
            "seed": attempt,
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "caption", "strict": True, "schema": caption_schema()},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def describe(self, image: Path, attempt: int) -> str:
        jpeg = prepare_image(image, self.cfg.max_edge_px)
        body = json.dumps(self.request_body(jpeg, attempt)).encode("utf-8")
        self._ensure_ours()
        status, data = self._post(body)
        if len(data) > _MAX_RESPONSE:
            raise _RequestError("server", "llama-server's answer was unreasonably large")
        if status != 200:
            raise _RequestError(
                "server", f"llama-server answered HTTP {status}{_server_error(data)}"
            )
        try:
            content = json.loads(data)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise _RequestError("server", "llama-server's answer wasn't a chat completion")
        return content

    def _post(self, body: bytes) -> tuple[int, bytes]:
        # http.client, not urllib: urllib honours *_proxy variables, and a
        # photo must never reach a proxy, not even by accident.
        timeout = self.cfg.request_timeout_s
        conn = http.client.HTTPConnection(LOOPBACK, self.port, timeout=timeout)
        try:
            conn.request(
                "POST",
                COMPLETIONS_PATH,
                body=body,
                headers={**self._headers(), "Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            return resp.status, resp.read(_MAX_RESPONSE + 1)
        except TimeoutError as exc:
            raise _RequestError("timeout", f"no answer from the model within {timeout} s") from exc
        except (OSError, http.client.HTTPException) as exc:
            self._check_server_alive()
            raise _RequestError(
                "server", f"talking to llama-server failed ({type(exc).__name__})"
            ) from exc
        finally:
            conn.close()

    def _check_server_alive(self) -> None:
        """Raise ``_Fatal`` if llama-server is gone (giving it a moment to finish dying)."""
        assert self.proc is not None
        try:
            code = self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return
        raise self._exited(code, "unexpectedly")


# -- the protocol loop --------------------------------------------------------


class _Out:
    """The protocol channel: a private copy of the original stdout."""

    def __init__(self) -> None:
        sys.stdout.flush()
        fd = os.dup(1)  # not inheritable, so llama-server can't write to it either
        os.dup2(2, 1)  # fd 1 (print, C libraries, warnings) now goes to stderr
        self._stream = os.fdopen(fd, "wb")

    def send(self, msg: dict[str, Any]) -> None:
        line = json.dumps(msg, ensure_ascii=True, separators=(",", ":")) + "\n"
        self._stream.write(line.encode("ascii"))
        self._stream.flush()


def _failure(img_id: object, kind: str, detail: str) -> dict[str, Any]:
    return {
        "type": "result",
        "id": img_id if isinstance(img_id, str) else None,
        "ok": False,
        "error": kind,
        "detail": detail,
    }


def _describe(backend: _Backend, msg: dict[str, Any]) -> dict[str, Any]:
    img_id = msg.get("id")
    image = msg.get("image")
    attempt = msg.get("attempt", 0)
    if not isinstance(img_id, str) or not img_id:
        return _failure(None, "bad_request", "a describe request needs a string id")
    if not isinstance(image, str) or not os.path.isabs(image):
        return _failure(img_id, "bad_request", "image must be an absolute path")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        return _failure(img_id, "bad_request", "attempt must be an integer >= 0")
    started = time.monotonic()
    try:
        raw = backend.describe(Path(image), attempt)
    except ImageDecodeError as exc:
        return _failure(img_id, "image_decode", str(exc))
    except _RequestError as exc:
        return _failure(img_id, exc.kind, exc.detail)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        return _failure(img_id, "bad_request", f"can't read the image file ({reason})")
    except _Fatal:
        raise
    except Exception as exc:
        traceback.print_exc()  # a worker bug: the traceback goes to stderr
        return _failure(img_id, "server", f"internal worker error ({type(exc).__name__})")
    ms = round((time.monotonic() - started) * 1000)
    return {"type": "result", "id": img_id, "ok": True, "raw": raw, "ms": ms}


def _parse_line(line: bytes) -> dict[str, Any] | None:
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    return msg if isinstance(msg, dict) else None


def _serve(backend: _Backend, stdin: IO[bytes], out: _Out) -> int:
    while line := stdin.readline():
        msg = _parse_line(line)
        if msg is None:
            out.send(_failure(None, "bad_request", "not a JSON object"))
            continue
        kind = msg.get("type")
        if kind == "shutdown":
            return 0
        if kind != "describe":
            out.send(_failure(msg.get("id"), "bad_request", "unknown message type"))
            continue
        try:
            out.send(_describe(backend, msg))
        except _Fatal as exc:
            out.send({"type": "fatal", "error": exc.kind, "detail": exc.detail})
            return 1
    return 0


def _on_sigterm(signum: int, frame: FrameType | None) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # let the cleanup finish
    raise SystemExit(128 + signum)


def main() -> int:
    out = _Out()
    signal.signal(signal.SIGTERM, _on_sigterm)
    stdin = sys.stdin.buffer
    backend: _Backend | None = None
    try:
        first = stdin.readline()
        if not first:
            return 0
        msg = _parse_line(first)
        try:
            if msg is None:
                raise _Fatal("bad_config", "the first line must be a JSON config message")
            cfg = _parse_config(msg)
            backend = _LlamaBackend(cfg) if cfg.backend == "llama" else _StubBackend(cfg)
            backend.start()
        except _Fatal as exc:
            out.send({"type": "fatal", "error": exc.kind, "detail": exc.detail})
            return 1
        except Exception as exc:
            traceback.print_exc()
            detail = f"the worker failed while starting ({type(exc).__name__}); see its stderr"
            out.send({"type": "fatal", "error": "internal", "detail": detail})
            return 1
        out.send({"type": "ready", "backend": cfg.backend, "prompt_version": PROMPT_VERSION})
        return _serve(backend, stdin, out)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 1  # the orchestrator went away; nobody is listening
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
