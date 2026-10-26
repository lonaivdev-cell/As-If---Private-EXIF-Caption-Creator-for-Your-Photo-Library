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
      "request_timeout_s": 600,
      # optional: the orchestrator's pid (see "Parent death" below)
      "parent_pid": 4242}

2. Worker -> orchestrator, once ready::

     {"type": "ready", "backend": "llama", "prompt_version": "caption_v1"}

   or, if it can't start::

     {"type": "fatal", "error": "<short kind>", "detail": "<last log lines>"}

   and then exits non-zero. Kinds: ``bad_config`` (including a
   ``server_args`` option that isn't on the list of known-safe speed,
   memory, GPU, sampling and logging options, and a ``llama_server`` script
   that doesn't ``exec`` llama-server), ``missing_binary``,
   ``missing_model``, ``server_start``, ``server_exited`` (the model failed
   to load), ``load_timeout``, ``port_taken`` and ``internal``. A
   system-wide llama.cpp settings file (``/etc/llama.cpp/config.ini``) is a
   ``bad_config`` too: see ``SYSTEM_LLAMA_CONFIG``.

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

4. ``{"type": "shutdown"}``, EOF on stdin, or SIGTERM, SIGHUP, SIGQUIT or
   SIGINT: stop llama-server, exit (0 for the first two, 130 for a signal).

Parent death: on Linux the worker asks the kernel for SIGTERM when the
thread that started it ends, so a SIGKILLed orchestrator can't leave it and
llama-server running with a photo in memory. It arms this itself, at
startup: the orchestrator can't (a ``preexec_fn`` isn't safe once it runs
other threads, and the CLI always does). If the config's ``parent_pid``
isn't its parent any more, the orchestrator died before that took effect,
and the worker exits 1 without starting anything.

Backends:

* ``stub``: no model. Deterministic, schema-valid English output derived
  from the image's average colour, brightness and shape, with the keyword
  ``stub`` always present so stub captions are easy to find and remove. For
  trying the pipeline and for tests.
* ``llama``: ``llama-server`` from llama.cpp, bound to 127.0.0.1 on a free
  port, with a random API key passed through the environment (never argv),
  so other local processes can't use it, and in offline mode. It sees none
  of the llama.cpp settings the user may have elsewhere: no ``LLAMA_*``,
  ``MTMD_*``, ``HF_*`` or ``HUGGING*`` variables but the worker's own, and
  an empty private ``XDG_CONFIG_HOME``, so the user's
  ``~/.config/llama.cpp/config.ini`` (which current builds apply before
  their environment and command line) is never read. On Linux the
  worker also checks that the process answering on that port really is its
  llama-server before sending it anything, and llama-server gets SIGTERM if
  the worker dies, however it dies. It runs in a process group of its own,
  which the worker stops as a whole. Neither the worker nor llama-server
  can write a core dump. llama-server's output goes to a bounded in-memory
  ring buffer, never to our stdout, with any line that carries image data
  hidden; its tail goes into ``fatal`` details. Requests go through
  ``http.client``, which, unlike ``urllib``, never routes through a proxy
  from the environment.
"""

from __future__ import annotations

import base64
import collections
import colorsys
import contextlib
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
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

from PIL import Image, ImageStat

from asif.infer.preprocess import ImageDecodeError, prepare_image
from asif.infer.schema import PROMPT_VERSION, caption_schema, load_prompt
from asif.proc import die_with_parent, disable_core_dumps, install_stop_signals

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
_IMAGE_MARK = b"data:image"
_HIDDEN_LINE = "[line hidden: it contained image data]"
_MAX_RESPONSE = 4 << 20
_HEALTH_TIMEOUT_S = 2.0
_STOP_GRACE_S = 3.0
_PR_SET_PDEATHSIG = 1

SYSTEM_LLAMA_CONFIG = "/etc/llama.cpp/config.ini"
"""llama.cpp's system-wide settings. From about b10400 every llama.cpp program
applies this file, then ``${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini``,
before its environment and command line, so either can set any option at
all (``rpc``, ``api-key``, ``log-file``...). The user's file is hidden by
pointing XDG_CONFIG_HOME elsewhere; no environment hides this one, so the
worker refuses to start llama-server while it exists (until the M3
sandbox can mask it)."""

# Environment variables llama.cpp reads as settings. LLAMA_* covers every
# option under each name it has had (LLAMA_ARG_*, and the LLAMA_LOG_FILE,
# LLAMA_OFFLINE... of older builds), MTMD_* the vision side, HF_* and
# HUGGING* where models come from. Only the worker's own get through.
_LLAMA_ENV_PREFIXES = ("LLAMA_", "MTMD_", "HF_", "HUGGING")

# The only llama-server options server_args may carry: ones that change speed,
# memory, GPU placement, sampling defaults or logging, and nothing else. An
# allow-list, because llama.cpp keeps adding options that load other weights
# (-mm, -md, --lora), download (-hf, -dr, the --*-default presets), reach other
# machines (--rpc, --connect, --tools-runtime ssh:...), run tools or serve
# files (--tools, --agent, --path, --media-path) or write prompts to disk
# (--log-file, --slot-save-path): a list of what to forbid is always one
# release behind. Every spelling llama.cpp accepts for an option is listed.
_ALLOWED_SERVER_ARGS = frozenset(
    " ".join(
        (
            # CPU threads, affinity and priority
            "-t --threads -tb --threads-batch -C --cpu-mask -Cr --cpu-range --cpu-strict",
            "--prio --poll -Cb --cpu-mask-batch -Crb --cpu-range-batch --cpu-strict-batch",
            "--prio-batch --poll-batch --numa",
            # context, batching, slots and the KV cache (all in memory)
            "-c --ctx-size -np --parallel -b --batch-size -ub --ubatch-size",
            "-cb --cont-batching -nocb --no-cont-batching -fa --flash-attn",
            "-ctk --cache-type-k -ctv --cache-type-v -kvo --kv-offload -nkvo --no-kv-offload",
            "-kvu --kv-unified -no-kvu --no-kv-unified --kv-unified-per-slot --swa-full",
            "-ctxcp --ctx-checkpoints --swa-checkpoints -cms --checkpoint-min-step",
            "-cram --cache-ram --cache-idle-slots --no-cache-idle-slots --cache-prompt",
            "--no-cache-prompt --cache-reuse --context-shift --no-context-shift",
            "-sps --slot-prompt-similarity",
            # how the model file As If chose is loaded, and where its layers go
            "-lm --load-mode -lzm --lazy-mode --mlock --mmap --no-mmap -dio --direct-io",
            "-ndio --no-direct-io --repack -nr --no-repack --no-host --check-tensors",
            "--warmup --no-warmup --op-offload --no-op-offload",
            "-ngl --gpu-layers --n-gpu-layers -dev --device -sm --split-mode",
            "-ts --tensor-split -mg --main-gpu -ot --override-tensor",
            "-cmoe --cpu-moe -ncmoe --n-cpu-moe --moe-cache-mib -ncffn --n-cpu-ffn",
            "-fit --fit -fitt --fit-target -fitc --fit-ctx",
            # the vision projector As If chose
            "--mmproj-offload --no-mmproj-offload -mmdev --mmproj-device",
            "--image-min-tokens --image-max-tokens --mtmd-batch-max-tokens",
            # chat template and sampling defaults (each request sets its own too)
            "--jinja --no-jinja --chat-template --reasoning-format -rea --reasoning",
            "--reasoning-budget -bs --backend-sampling --perf --no-perf",
            "--temp --temperature --top-k --top-p --min-p --repeat-penalty --repeat-last-n",
            "--presence-penalty --frequency-penalty",
            # logging (into the in-memory tail, never a file) and the HTTP side
            "-v --verbose --log-verbose -lv --verbosity --log-verbosity --log-disable",
            "--log-colors --log-prefix --no-log-prefix --log-timestamps --no-log-timestamps",
            "--log-jsonl --no-log-jsonl --offline -to --timeout --threads-http",
            "--no-webui --no-ui --no-slots",
        )
    ).split()
)


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
    parent_pid: int | None = None


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
    parent_pid = msg.get("parent_pid")
    if parent_pid is not None:
        parent_pid = integer("parent_pid", 0, 1)
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
        parent_pid=parent_pid,
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
        # A verbose llama-server logs whole request bodies, the photo's base64
        # included, on one line. Only the first chunk of a line is kept, and
        # the base64 always comes after "data:image" on its line: so either
        # that chunk holds the marker and is hidden, or none of the photo is
        # in it. The rest of the line is still read, so the pipe never fills.
        continuing = False
        with stream:
            while chunk := stream.readline(_LOG_LINE_MAX):
                if not continuing:
                    if _IMAGE_MARK in chunk:
                        line = _HIDDEN_LINE
                    else:
                        line = chunk.decode("utf-8", "replace").rstrip()[:_SHOWN_LINE_MAX]
                    with self._lock:
                        self._lines.append(line)
                continuing = not chunk.endswith(b"\n")

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def tail(self, n: int = FATAL_TAIL_LINES) -> str:
        with self._lock:
            return "\n".join(list(self._lines)[-n:])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        port: int = sock.getsockname()[1]
        return port


def _listening_sockets(port: int) -> set[str] | None:
    """The sockets listening on ``port``, named as ``/proc/<pid>/fd`` links
    name them. None where the kernel can't be asked (not Linux)."""
    tables = (Path("/proc/net/tcp"), Path("/proc/net/tcp6"))
    if not tables[0].exists():
        return None
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
    return listeners


def _held_by(pid: int) -> set[str]:
    """What process ``pid``'s file descriptors point at (empty if we can't look)."""
    fd_dir = Path(f"/proc/{pid}/fd")
    held: set[str] = set()
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return held
    for fd in fds:
        try:
            held.add(os.readlink(fd_dir / fd))
        except OSError:
            continue
    return held


def _owns_port(pid: int, port: int) -> bool:
    """Is every socket listening on ``port`` held by process ``pid``?

    Between our finding a free port and llama-server binding it, another
    local program could take the port and pose as llama-server to receive
    photos. On Linux this asks the kernel who really listens; elsewhere it
    can't tell and says yes (the M3 sandbox gives llama-server a private
    loopback anyway).
    """
    listeners = _listening_sockets(port)
    if listeners is None:
        return True
    return bool(listeners) and listeners <= _held_by(pid)


def _group_owns_port(pgid: int, port: int) -> bool:
    """Is every socket listening on ``port`` held by processes in group ``pgid``?

    llama-server leads a session of its own, so no outside process can join
    its group: a listener in it is a child of whatever ``llama_server`` ran.
    """
    listeners = _listening_sockets(port)
    if not listeners:
        return False
    held: set[str] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.getpgid(int(entry)) != pgid:
                continue
        except OSError:
            continue
        held |= _held_by(int(entry))
    return listeners <= held


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _option_name(arg: str) -> str | None:
    """The llama-server option ``arg`` names, as llama.cpp reads it; None for a value.

    llama.cpp treats every argument that starts with ``-`` as an option
    (unless an option before it takes it as its value), and turns ``_`` into
    ``-`` in ``--`` options before looking them up.
    """
    if not arg.startswith("-"):
        return None
    try:
        float(arg)  # a negative number, as in ``--prio -1``
    except ValueError:
        pass
    else:
        return None
    # No llama.cpp option name holds "=" or a space: cutting there keeps a
    # value (an API key, say) out of the error message.
    name = arg.split("=", 1)[0].split(None, 1)[0]
    return name.replace("_", "-") if name.startswith("--") else name


def _check_server_args(args: tuple[str, ...]) -> None:
    for arg in args:
        name = _option_name(arg)
        if name is not None and name not in _ALLOWED_SERVER_ARGS:
            raise _Fatal(
                "bad_config",
                f"[model] server_args can't include {name[:60]!r}. As If passes on only "
                "the llama-server options it knows are safe (speed, memory, GPU, sampling "
                "and logging settings): it chooses the model, the projector, the address "
                "and the key itself, and llama-server must never download anything, reach "
                "another machine, run tools or serve files. If this option is harmless, "
                "please report it so it can be added.",
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
        self.config_dir: str | None = None
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
        if os.path.lexists(SYSTEM_LLAMA_CONFIG):
            raise _Fatal(
                "bad_config",
                f"llama.cpp's system-wide settings file {SYSTEM_LLAMA_CONFIG} exists. Current "
                "llama-server builds apply it before anything As If sets, so it could make "
                "llama-server send work to another machine or accept other keys. As If won't "
                "start llama-server while it's there. If nothing else needs it, rename or "
                "remove it (that takes an administrator, e.g. with sudo mv).",
            )

        self.port = _free_port()
        argv = [binary, "-m", cfg.model, "--mmproj", cfg.mmproj]
        argv += ["--host", LOOPBACK, "--port", str(self.port), "-c", str(cfg.ctx), "-np", "1"]
        if cfg.threads > 0:
            argv += ["-t", str(cfg.threads)]
        if cfg.gpu_layers is not None:
            argv += ["-ngl", str(cfg.gpu_layers)]
        argv += cfg.server_args

        env = {k: v for k, v in os.environ.items() if not k.startswith(_LLAMA_ENV_PREFIXES)}
        env["LLAMA_API_KEY"] = env["LLAMA_ARG_API_KEY"] = self.key
        # Offline mode as a variable, under both of its names: builds from
        # about b5500 to b9320 read LLAMA_OFFLINE, later ones LLAMA_ARG_OFFLINE,
        # and each ignores the other (as older builds, which have no offline
        # mode and no download on start, ignore both). No option turns it off.
        env["LLAMA_OFFLINE"] = env["LLAMA_ARG_OFFLINE"] = "1"
        # An empty config dir of our own (0700, so nobody can plant a
        # config.ini in it), in place of the user's ~/.config. HOME stays: GPU
        # drivers keep their shader caches under it.
        try:
            self.config_dir = tempfile.mkdtemp(prefix="asif-llama-config-")
        except OSError as exc:
            raise _Fatal(
                "server_start",
                "no private temporary folder could be made for llama-server: "
                f"{exc.strerror or exc}",
            ) from exc
        env["XDG_CONFIG_HOME"] = self.config_dir
        try:
            self.proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                close_fds=True,
                # Its own session and process group, which nothing outside can
                # join: close() stops the whole group, wrapper scripts' children too.
                start_new_session=True,
                # No other thread runs yet (the log reader starts below), so
                # die_with_parent can safely hand over its preexec_fn.
                preexec_fn=die_with_parent(),
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
        if proc is not None:
            # The whole group, even if llama-server itself is gone: a wrapper
            # script may have left a child behind.
            _signal_group(proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=_STOP_GRACE_S)
            _signal_group(proc.pid, signal.SIGKILL)
            proc.wait()
            self.log.join(timeout=1)
        config_dir, self.config_dir = self.config_dir, None
        if config_dir is not None:
            shutil.rmtree(config_dir, ignore_errors=True)

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
            if _group_owns_port(self.proc.pid, self.port):
                # A shell script that runs llama-server without exec: the
                # death signal would reach only the script, so llama-server
                # could outlive us. close() stops it; refuse to go on.
                raise _Fatal(
                    "bad_config",
                    "[model] llama_server points at a script that starts llama-server "
                    "without exec, so As If can't make sure llama-server stops when it "
                    "does. In that script, put exec in front of the llama-server command, "
                    "or point llama_server at llama-server itself.",
                )
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


def _stop_with_parent() -> None:
    """Have the kernel send us SIGTERM when our parent dies (Linux only).

    Set here, not by the orchestrator: a ``preexec_fn`` isn't safe once a
    process runs other threads, and the CLI always does by then.
    """
    if not sys.platform.startswith("linux"):
        return
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)


def main() -> int:
    # Before any photo is read: a core dump would write its pixels to disk.
    # llama-server inherits the limit, and can't raise it again.
    disable_core_dumps()
    out = _Out()
    # llama-server has a session of its own, so a closed terminal or Ctrl+\
    # no longer reaches it: these signals must unwind into close() instead.
    install_stop_signals()
    # After the handlers, so a dead orchestrator's SIGTERM unwinds into
    # close() like any other, and before anything is started.
    _stop_with_parent()
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
            if cfg.parent_pid is not None and os.getppid() != cfg.parent_pid:
                # It died before the death signal was armed: nobody is listening.
                print("asif worker: the orchestrator has exited; stopping", file=sys.stderr)
                return 1
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
