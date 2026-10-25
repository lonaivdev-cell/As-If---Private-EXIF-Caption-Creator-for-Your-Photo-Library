"""A stand-in for llama.cpp's ``llama-server``, for tests. Standard library only.

It takes llama-server's command line (``-m``, ``--mmproj``, ``--host``,
``--port``, ``-c``, ``-np``, ``-t``, ``-ngl``; unknown extras are accepted),
refuses a non-loopback ``--host`` and any ``--api-key`` on the command line,
and serves ``GET /health`` (503 "Loading model" for the first
``FAKE_LLAMA_LOADING_POLLS`` polls, default 2, then 200) and
``POST /v1/chat/completions`` in OpenAI format.

Every completion request is checked the way a strict server would:

* the Bearer token equals ``LLAMA_API_KEY`` (and ``LLAMA_ARG_API_KEY``) from
  the environment;
* there is a non-empty system prompt;
* the image is one ``data:image/jpeg;base64,`` URL that decodes to a
  well-formed JPEG whose longest edge is at most ``FAKE_LLAMA_MAX_EDGE`` and
  that carries no metadata segments at all (no EXIF, XMP, ICC, IPTC or
  comments: only APP0/JFIF is allowed);
* ``response_format`` carries the caption JSON schema (compared with
  ``src/asif/infer/schema.json``), and thinking is switched off.

A failed check answers HTTP 400 with the reason, so tests see it.

Behaviour comes from ``FAKE_LLAMA_MODE``, or else from the first mode name
found in the model's file name (``model-slow.gguf``)::

    good              valid caption JSON (mentions seed, temperature, image size)
    invalid-json      text that isn't JSON
    schema-violation  JSON that breaks the caption schema
    slow              sleeps FAKE_LLAMA_DELAY_S (default 3) before answering
    crash-on-load     logs a load error and exits 1, like a bad model file
    never-ready       /health never stops answering 503
    http500           every completion is an HTTP 500
    die-on-request    exits abruptly when a completion request arrives
    spew              floods its log (FAKE_LLAMA_SPEW_KB, default 4096 KiB)

``FAKE_LLAMA_RECORD`` names a JSON Lines file the server appends to: one
``start`` record (pid, argv, names of LLAMA_* variables) and one ``request``
record per completion (seed, temperature, image size...). Never image data.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MODES = (
    "invalid-json",
    "schema-violation",
    "slow",
    "crash-on-load",
    "never-ready",
    "http500",
    "die-on-request",
    "spew",
    "good",
)
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
SCHEMA_FILE = Path(__file__).resolve().parents[1] / "src" / "asif" / "infer" / "schema.json"
DATA_URL_PREFIX = "data:image/jpeg;base64,"

_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_STANDALONE = frozenset({0x01, *range(0xD0, 0xD8)})


def install(directory: Path, name: str = "llama-server") -> Path:
    """Write an executable ``llama-server`` wrapper that execs this fake.

    ``exec`` matters: the fake then *is* the process the worker started, so
    signals (and PR_SET_PDEATHSIG) reach it directly.
    """
    script = directory / name
    script.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
        '"$@"\n'
    )
    script.chmod(0o755)
    return script


@dataclass
class FakeLlama:
    """Test helper: the fake installed in ``directory``, plus its record file.

    Tests point ``FAKE_LLAMA_RECORD`` at ``record_path`` (the worker passes
    its environment on to llama-server).
    """

    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binary = install(self.directory)
        self.record_path = self.directory / "record.jsonl"

    def model_files(self, mode: str = "good") -> tuple[Path, Path]:
        """Dummy weights whose name selects ``mode``, and a dummy projector."""
        model = self.directory / f"model-{mode}.gguf"
        mmproj = self.directory / "mmproj-F16.gguf"
        for path in (model, mmproj):
            if not path.exists():
                path.write_bytes(b"GGUF, but not really")
        return model, mmproj

    def records(self, event: str | None = None) -> list[dict[str, Any]]:
        if not self.record_path.exists():
            return []
        rows = [json.loads(line) for line in self.record_path.read_text().splitlines()]
        return [r for r in rows if event is None or r["event"] == event]

    def pids(self) -> list[int]:
        return [r["pid"] for r in self.records("start")]


def process_gone(pid: int) -> bool:
    """True once ``pid`` has exited (a zombie nobody reaped yet counts as gone)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return True
    return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")


def jpeg_info(data: bytes) -> tuple[int, int, list[int]]:
    """Walk a JPEG's segments. Returns (width, height, markers). Raises ValueError."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG (no SOI marker)")
    if data[-2:] != b"\xff\xd9":
        raise ValueError("JPEG doesn't end with an EOI marker")
    pos, markers, size, seen_sos = 2, [], None, False
    while pos < len(data) - 1:
        if data[pos] != 0xFF:
            raise ValueError(f"expected a marker at byte {pos}")
        marker = data[pos + 1]
        pos += 2
        if marker == 0xFF:
            pos -= 1  # fill byte
            continue
        if marker == 0xD9:
            break
        if marker in _STANDALONE:
            continue
        if pos + 2 > len(data):
            raise ValueError("JPEG truncated inside a segment header")
        length = int.from_bytes(data[pos : pos + 2], "big")
        if length < 2 or pos + length > len(data):
            raise ValueError("JPEG segment runs past the end of the file")
        markers.append(marker)
        if marker in _SOF_MARKERS:
            height = int.from_bytes(data[pos + 3 : pos + 5], "big")
            width = int.from_bytes(data[pos + 5 : pos + 7], "big")
            size = (width, height)
        pos += length
        if marker == 0xDA:  # entropy-coded data runs to the next real marker
            seen_sos = True
            while pos < len(data) - 1 and not (
                data[pos] == 0xFF and data[pos + 1] != 0 and data[pos + 1] not in _STANDALONE
            ):
                pos += 1
    if size is None or not seen_sos:
        raise ValueError("JPEG has no frame or no scan")
    if size[0] == 0 or size[1] == 0:
        raise ValueError("JPEG has a zero dimension")
    return size[0], size[1], markers


def check_request(body: Any, max_edge: int) -> tuple[list[str], dict[str, Any]]:
    """Problems with a chat-completion request, and facts about it for the record."""
    problems: list[str] = []
    info: dict[str, Any] = {}
    if not isinstance(body, dict):
        return ["body is not a JSON object"], info
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ["messages must be a list with a system and a user message"], info
    system = messages[0]
    if not (
        isinstance(system, dict)
        and system.get("role") == "system"
        and isinstance(system.get("content"), str)
        and system["content"].strip()
    ):
        problems.append("the first message must be a non-empty system prompt")
    images: list[str] = []
    for msg in messages[1:]:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            problems.append("only user messages may follow the system prompt")
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    images.append(url if isinstance(url, str) else "")
                elif isinstance(part, dict) and part.get("type") == "text":
                    info["user_text"] = part.get("text")
    if len(images) != 1:
        problems.append(f"expected exactly one image, got {len(images)}")
    else:
        url = images[0]
        if not url.startswith(DATA_URL_PREFIX):
            problems.append("the image must be a data:image/jpeg;base64 URL")
        else:
            try:
                jpeg = base64.b64decode(url[len(DATA_URL_PREFIX) :], validate=True)
                width, height, markers = jpeg_info(jpeg)
            except (binascii.Error, ValueError) as exc:
                problems.append(f"the image isn't a valid JPEG: {exc}")
            else:
                info.update(width=width, height=height, jpeg_bytes=len(jpeg))
                if max(width, height) > max_edge:
                    problems.append(f"image is {width}x{height}, larger than {max_edge} px")
                meta = sorted({m for m in markers if 0xE1 <= m <= 0xEF or m == 0xFE})
                if meta:
                    problems.append(f"image carries metadata segments: {[hex(m) for m in meta]}")
    expected_schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    expected_format = {
        "type": "json_schema",
        "json_schema": {"name": "caption", "strict": True, "schema": expected_schema},
    }
    if body.get("response_format") != expected_format:
        problems.append("response_format must carry the caption JSON schema")
    if body.get("chat_template_kwargs") != {"enable_thinking": False}:
        problems.append("chat_template_kwargs must switch thinking off")
    if body.get("stream") not in (None, False):
        problems.append("streaming is not expected")
    max_tokens = body.get("max_tokens")
    temperature = body.get("temperature")
    seed = body.get("seed")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        problems.append("max_tokens must be a positive integer")
    if not isinstance(temperature, int | float) or isinstance(temperature, bool):
        problems.append("temperature must be a number")
    if not isinstance(seed, int) or isinstance(seed, bool):
        problems.append("seed must be an integer")
    info.update(max_tokens=max_tokens, temperature=temperature, seed=seed)
    return problems, info


def good_answer(info: dict[str, Any]) -> str:
    return json.dumps(
        {
            "caption": (
                f"A fake caption for a {info.get('width')}x{info.get('height')} image "
                f"(seed {info.get('seed')}, temperature {info.get('temperature')})."
            ),
            "keywords": ["fake", "test", "photo"],
            "has_text": False,
            "people_count": 0,
            "setting": "unclear",
        }
    )


SCHEMA_VIOLATION = json.dumps(
    {"caption": "", "keywords": ["one"], "has_text": "maybe", "people_count": -1, "setting": "?"}
)


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], mode: str, model: str) -> None:
        super().__init__(address, Handler)
        self.mode = mode
        self.model = model
        self.api_key = os.environ.get("LLAMA_API_KEY", "")
        self.max_edge = int(os.environ.get("FAKE_LLAMA_MAX_EDGE", "4096"))
        self.loading_polls = int(os.environ.get("FAKE_LLAMA_LOADING_POLLS", "2"))
        self.delay_s = float(os.environ.get("FAKE_LLAMA_DELAY_S", "3"))
        self.health_polls = 0
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: FakeServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"srv  log_server_r: {format % args}\n")
        sys.stderr.flush()

    def _send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str) -> None:
        self._send_json(status, {"error": {"code": status, "message": message, "type": kind}})

    def do_GET(self) -> None:
        if self.path != "/health":
            self._error(404, "File Not Found", "not_found_error")
            return
        with self.server.lock:
            self.server.health_polls += 1
            loading = self.server.health_polls <= self.server.loading_polls
        if loading or self.server.mode == "never-ready":
            self._error(503, "Loading model", "unavailable_error")
        else:
            self._send_json(200, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._error(404, "File Not Found", "not_found_error")
            return
        expected = f"Bearer {self.server.api_key}"
        if not self.server.api_key or self.headers.get("Authorization") != expected:
            self._error(401, "Invalid API Key", "authentication_error")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            self._error(400, "request body is not JSON", "invalid_request_error")
            return
        problems, info = check_request(body, self.server.max_edge)
        record({"event": "request", "problems": problems, **info})
        if problems:
            self._error(400, "; ".join(problems), "invalid_request_error")
            return
        mode = self.server.mode
        if mode == "die-on-request":
            print("GGML_ASSERT failed: fake crash for tests", flush=True)
            os._exit(134)
        if mode == "http500":
            self._error(500, "fake internal error for tests", "server_error")
            return
        if mode == "slow":
            time.sleep(self.server.delay_s)
        if mode == "spew":
            spew(256)
        content = {"invalid-json": "Sure! Here is { not JSON", "schema-violation": SCHEMA_VIOLATION}
        self._send_json(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.server.model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": content.get(mode) or good_answer(info),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


def record(entry: dict[str, Any]) -> None:
    path = os.environ.get("FAKE_LLAMA_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def spew(kib: int) -> None:
    line = "llama_spew: " + "x" * 1011 + "\n"
    for i in range(kib):
        stream = sys.stdout if i % 2 else sys.stderr
        stream.write(line)
    sys.stdout.flush()
    sys.stderr.flush()


def pick_mode(model: str) -> str:
    if mode := os.environ.get("FAKE_LLAMA_MODE"):
        return mode
    name = Path(model).name
    return next((m for m in MODES if m in name), "good")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llama-server", allow_abbrev=False)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-c", "--ctx-size", type=int, default=4096)
    parser.add_argument("-np", "--parallel", type=int, default=1)
    parser.add_argument("-t", "--threads", type=int, default=-1)
    parser.add_argument("-ngl", "--n-gpu-layers", "--gpu-layers", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    args, extras = parser.parse_known_args(argv)

    if args.api_key is not None:
        print("error: fake llama-server: the API key must come from the environment", flush=True)
        return 2
    if args.host not in LOOPBACK_HOSTS:
        print(f"error: fake llama-server refuses non-loopback host {args.host!r}", flush=True)
        return 2
    if not (os.environ.get("LLAMA_API_KEY") and os.environ.get("LLAMA_ARG_API_KEY")):
        print("error: fake llama-server: no API key in the environment", flush=True)
        return 2
    if os.environ["LLAMA_API_KEY"] != os.environ["LLAMA_ARG_API_KEY"]:
        print("error: fake llama-server: LLAMA_API_KEY != LLAMA_ARG_API_KEY", flush=True)
        return 2

    mode = pick_mode(args.model)
    record(
        {
            "event": "start",
            "pid": os.getpid(),
            "argv": sys.argv[1:] if argv is None else argv,
            "extras": extras,
            "mode": mode,
            "llama_env": sorted(k for k in os.environ if k.startswith("LLAMA")),
        }
    )
    print("build: fake llama-server for As If tests", flush=True)
    print(f"llama_model_load: loading model from {args.model}", flush=True)
    if mode == "crash-on-load":
        print("llama_model_load: error loading model: fake crash for tests", flush=True)
        print("main: exiting due to model loading error", flush=True)
        return 1
    for path in (args.model, args.mmproj):
        if not os.path.isfile(path):
            print(f"error: failed to open {path}", flush=True)
            return 1
    if mode == "spew":
        spew(int(os.environ.get("FAKE_LLAMA_SPEW_KB", "4096")))

    server = FakeServer((args.host, args.port), mode, Path(args.model).name)
    print(f"main: server is listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
