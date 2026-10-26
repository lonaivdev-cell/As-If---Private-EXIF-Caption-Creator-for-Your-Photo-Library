"""The inference worker, driven over its real stdin/stdout protocol.

The llama backend runs against ``tests/fake_llama_server.py``, a strict
stand-in for llama-server (llama.cpp isn't needed to run the tests).
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import queue
import re
import shlex
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

from asif.infer import worker
from asif.infer.client import WorkerConfig
from asif.infer.preprocess import prepare_image
from asif.infer.schema import PROMPT_VERSION, SchemaError, caption_schema, load_prompt
from asif.infer.schema import parse_caption_json as parse
from tests.conftest import FIXTURE_EXPECT
from tests.fake_llama_server import FakeLlama, process_gone

DECODABLE = sorted(n for n, e in FIXTURE_EXPECT.items() if e == "ok" and n != "panorama.jpg")
WORKER = [sys.executable, "-I", "-m", "asif.infer.worker"]
FAKE_SCRIPT = Path(__file__).resolve().with_name("fake_llama_server.py")
OUR_LLAMA_ENV = ["LLAMA_API_KEY", "LLAMA_ARG_API_KEY", "LLAMA_ARG_OFFLINE", "LLAMA_OFFLINE"]
# llama.cpp builds that read offline mode under different names (and only
# the newer applies config.ini files): the worker must cover both.
BUILDS = ["9200", "11000"]
# Settings someone may have in ~/.config/llama.cpp/config.ini for their own
# llama-server. Applied to As If's, they'd send work to another machine,
# accept another key and log to a file.
USER_LLAMA_CONFIG = (
    "[*]\n"
    "rpc = 10.0.0.2:50052\n"
    "api-key = a-key-someone-else-knows\n"
    "log-file = /tmp/llama.log\n"
    "verbose = true\n"
)
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")


def wait_until(check: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.05)
    return check()


class Proc:
    """A worker (or anything speaking its protocol) as a raw child process."""

    def __init__(self, argv: list[str]) -> None:
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.lines: queue.Queue[bytes] = queue.Queue()
        self.raw: list[bytes] = []
        self.err = bytearray()
        self._threads = [
            threading.Thread(target=self._pump_out, daemon=True),
            threading.Thread(target=self._pump_err, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _pump_out(self) -> None:
        assert self.p.stdout is not None
        for line in self.p.stdout:
            self.raw.append(line)
            self.lines.put(line)

    def _pump_err(self) -> None:
        assert self.p.stderr is not None
        while chunk := self.p.stderr.read(65536):
            self.err += chunk

    def send(self, msg: dict[str, Any] | str) -> None:
        assert self.p.stdin is not None
        line = msg if isinstance(msg, str) else json.dumps(msg)
        self.p.stdin.write(line.encode() + b"\n")
        self.p.stdin.flush()

    def recv(self, timeout: float = 30) -> dict[str, Any]:
        msg = json.loads(self.lines.get(timeout=timeout))
        assert isinstance(msg, dict)
        return msg

    def ask(self, image: Path, img_id: str = "i_000001", attempt: int = 0) -> dict[str, Any]:
        self.send({"type": "describe", "id": img_id, "image": str(image), "attempt": attempt})
        return self.recv()

    def wait(self, timeout: float = 30) -> int:
        code = self.p.wait(timeout)
        for t in self._threads:
            t.join(5)
        return code

    def stderr(self) -> str:
        return self.err.decode("utf-8", "replace")


@pytest.fixture
def spawn() -> Iterator[Callable[..., Proc]]:
    procs: list[Proc] = []

    def make(argv: list[str] | None = None) -> Proc:
        proc = Proc(argv or WORKER)
        procs.append(proc)
        return proc

    yield make
    for proc in procs:
        if proc.p.poll() is None:
            proc.p.kill()
        proc.p.wait()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeLlama:
    if os.path.lexists(worker.SYSTEM_LLAMA_CONFIG):
        pytest.skip(f"{worker.SYSTEM_LLAMA_CONFIG} exists, so the worker refuses llama-server")
    fake = FakeLlama(tmp_path / "llama")
    # The worker's private config dir for llama-server goes here: a worker
    # killed outright (as many tests do) can't remove it.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(fake.record_path))
    monkeypatch.setenv("FAKE_LLAMA_MAX_EDGE", "512")
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    return fake


def stub_config(**kw: Any) -> dict[str, object]:
    return WorkerConfig(backend="stub", lang="en", max_edge_px=512, **kw).to_message()


def llama_config(fake: FakeLlama, mode: str = "good", **kw: Any) -> dict[str, object]:
    model, mmproj = fake.model_files(mode)
    settings: dict[str, Any] = {
        "backend": "llama",
        "lang": "en",
        "max_edge_px": 512,
        "llama_server": str(fake.binary),
        "model": str(model),
        "mmproj": str(mmproj),
        "load_timeout_s": 30,
        "request_timeout_s": 30,
    }
    settings.update(kw)
    return WorkerConfig(**settings).to_message()


def started(spawn: Callable[..., Proc], config: dict[str, object]) -> Proc:
    proc = spawn()
    proc.send(config)
    ready = proc.recv()
    assert ready["type"] == "ready", ready
    return proc


def flag_value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


# -- stub backend -----------------------------------------------------------------


def test_stub_ready_and_describe(spawn: Callable[..., Proc], fixtures_dir: Path) -> None:
    proc = spawn()
    proc.send(stub_config())
    assert proc.recv() == {"type": "ready", "backend": "stub", "prompt_version": PROMPT_VERSION}
    result = proc.ask(fixtures_dir / "plain.jpg", "i_000007")
    assert result["type"] == "result" and result["id"] == "i_000007" and result["ok"] is True
    assert isinstance(result["ms"], int) and result["ms"] >= 0
    caption = parse(result["raw"])
    assert "stub" in caption.keywords
    assert "stub" in caption.caption.lower()
    proc.send({"type": "shutdown"})
    assert proc.wait() == 0


def test_stub_is_deterministic_and_ignores_attempt(
    spawn: Callable[..., Proc], fixtures_dir: Path
) -> None:
    proc = started(spawn, stub_config())
    answers = {proc.ask(fixtures_dir / "exif.jpg", attempt=a)["raw"] for a in (0, 1, 2, 0)}
    assert len(answers) == 1


@pytest.mark.parametrize("name", DECODABLE)
def test_stub_output_is_schema_valid_for_every_photo(fixtures_dir: Path, name: str) -> None:
    raw = json.dumps(worker.stub_caption(prepare_image(fixtures_dir / name, 512)))
    caption = parse(raw)
    assert caption.keywords[0] == "stub"
    # The stricter JSON-schema limits too, not just the validator's.
    schema = caption_schema()["properties"]
    assert schema["keywords"]["minItems"] <= len(caption.keywords) <= 15
    assert len(caption.caption) <= schema["caption"]["maxLength"]


@pytest.mark.parametrize(
    ("colour", "size", "expected"),
    [
        ((250, 250, 250), (64, 48), {"white", "bright", "wide"}),
        ((5, 5, 5), (48, 64), {"black", "dark", "tall"}),
        ((128, 128, 128), (64, 64), {"grey", "mid-tone", "square"}),
        ((200, 20, 20), (64, 48), {"red"}),
        ((20, 160, 20), (64, 48), {"green"}),
        ((20, 40, 220), (64, 48), {"blue"}),
    ],
)
def test_stub_describes_colour_brightness_and_shape(
    tmp_path: Path, colour: tuple[int, int, int], size: tuple[int, int], expected: set[str]
) -> None:
    path = tmp_path / "x.png"
    Image.new("RGB", size, colour).save(path)
    answer = worker.stub_caption(prepare_image(path, 512))
    assert expected <= set(answer["keywords"])
    assert answer["keywords"][0] == "stub"


def test_stub_differs_between_photos(spawn: Callable[..., Proc], fixtures_dir: Path) -> None:
    proc = started(spawn, stub_config())
    assert (
        proc.ask(fixtures_dir / "plain.jpg")["raw"] != proc.ask(fixtures_dir / "alpha.png")["raw"]
    )


@pytest.mark.parametrize("name", ["truncated.jpg", "corrupt.jpg", "photo.heic"])
def test_undecodable_photos_are_image_decode_errors(
    spawn: Callable[..., Proc], fixtures_dir: Path, name: str
) -> None:
    proc = started(spawn, stub_config())
    result = proc.ask(fixtures_dir / name, "i_000002")
    assert result["ok"] is False and result["error"] == "image_decode"
    assert result["id"] == "i_000002"
    assert name not in result["detail"]
    assert proc.ask(fixtures_dir / "plain.jpg")["ok"] is True  # and it carries on


def test_unreadable_or_relative_paths_are_bad_requests(
    spawn: Callable[..., Proc], tmp_path: Path
) -> None:
    proc = started(spawn, stub_config())
    for image in (tmp_path / "missing.jpg", tmp_path, Path("relative.jpg")):
        result = proc.ask(image, "i_000003")
        assert (result["ok"], result["error"], result["id"]) == (False, "bad_request", "i_000003")


def test_malformed_requests_get_bad_request_results(
    spawn: Callable[..., Proc], fixtures_dir: Path
) -> None:
    proc = started(spawn, stub_config())
    image = str(fixtures_dir / "plain.jpg")
    cases: list[tuple[dict[str, Any] | str, object]] = [
        ("this is not json", None),
        ("[1, 2, 3]", None),
        ({"type": "dance", "id": "i_9"}, "i_9"),
        ({"type": "describe", "image": image}, None),
        ({"type": "describe", "id": 7, "image": image}, None),
        ({"type": "describe", "id": "i_1", "image": image, "attempt": -1}, "i_1"),
        ({"type": "describe", "id": "i_2", "image": image, "attempt": True}, "i_2"),
        ({"type": "describe", "id": "i_3", "image": 42}, "i_3"),
    ]
    for request, expected_id in cases:
        proc.send(request)
        result = proc.recv()
        assert result["type"] == "result" and result["ok"] is False, request
        assert (result["error"], result["id"]) == ("bad_request", expected_id), request
    assert proc.ask(Path(image))["ok"] is True


def test_eof_ends_the_worker_cleanly(spawn: Callable[..., Proc]) -> None:
    proc = started(spawn, stub_config())
    assert proc.p.stdin is not None
    proc.p.stdin.close()
    assert proc.wait() == 0


def test_eof_before_any_config_is_a_clean_exit(spawn: Callable[..., Proc]) -> None:
    proc = spawn()
    assert proc.p.stdin is not None
    proc.p.stdin.close()
    assert proc.wait() == 0
    assert proc.raw == []


@pytest.mark.parametrize(
    "config",
    [
        "not json at all",
        {"type": "describe"},
        {"type": "config", "backend": "gpt", "lang": "en", "max_edge_px": 512},
        {"type": "config", "backend": "stub", "lang": "en", "max_edge_px": 0},
        {"type": "config", "backend": "stub", "lang": "en", "max_edge_px": True},
        {"type": "config", "backend": "stub", "lang": "", "max_edge_px": 512},
        {"type": "config", "backend": "stub", "lang": "en", "max_edge_px": 512, "ctx": "4k"},
        {
            "type": "config",
            "backend": "stub",
            "lang": "en",
            "max_edge_px": 512,
            "server_args": "-v",
        },
        {"type": "config", "backend": "stub", "lang": "en", "max_edge_px": 512, "gpu_layers": -1},
        {"type": "config", "backend": "stub", "lang": "en", "max_edge_px": 512, "parent_pid": "1"},
        {"type": "config", "backend": "stub", "lang": "en", "max_edge_px": 512, "parent_pid": 0},
    ],
)
def test_a_bad_config_is_fatal(spawn: Callable[..., Proc], config: dict[str, Any] | str) -> None:
    proc = spawn()
    proc.send(config)
    msg = proc.recv()
    assert msg["type"] == "fatal" and msg["error"] == "bad_config"
    assert proc.wait() == 1


def test_stdout_carries_only_protocol_lines(spawn: Callable[..., Proc], fixtures_dir: Path) -> None:
    # Everything that usually ends up on stdout, from inside a request.
    code = textwrap.dedent(
        """
        import os, subprocess, sys, warnings
        from asif.infer import worker

        real = worker.prepare_image

        def noisy(path, max_edge):
            print("stray print")
            sys.stdout.flush()
            os.write(1, b"raw write to fd 1\\n")
            warnings.warn("a library warning")
            subprocess.run(["sh", "-c", "echo child process output"], check=True)
            return real(path, max_edge)

        worker.prepare_image = noisy
        raise SystemExit(worker.main())
        """
    )
    proc = spawn([sys.executable, "-c", code])
    proc.send(stub_config())
    assert proc.recv()["type"] == "ready"
    assert proc.ask(fixtures_dir / "plain.jpg")["ok"] is True
    proc.send({"type": "shutdown"})
    assert proc.wait() == 0
    assert [json.loads(line)["type"] for line in proc.raw] == ["ready", "result"]
    for noise in ("stray print", "raw write to fd 1", "a library warning", "child process output"):
        assert noise in proc.stderr()


# -- llama backend, against the fake llama-server ------------------------------------


@pytest.mark.parametrize("build", BUILDS)
def test_llama_happy_path(
    spawn: Callable[..., Proc],
    fake: FakeLlama,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    build: str,
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_BUILD", build)  # the fake insists on that build's offline mode
    proc = spawn()
    proc.send(llama_config(fake))
    assert proc.recv() == {"type": "ready", "backend": "llama", "prompt_version": PROMPT_VERSION}
    result = proc.ask(fixtures_dir / "gps.jpg")
    assert result["ok"] is True, result
    assert "fake caption" in parse(result["raw"]).caption

    [start] = fake.records("start")
    argv = start["argv"]
    model, mmproj = fake.model_files()
    assert flag_value(argv, "-m") == str(model)
    assert flag_value(argv, "--mmproj") == str(mmproj)
    assert flag_value(argv, "--host") == "127.0.0.1"
    assert flag_value(argv, "-c") == "4096"
    assert flag_value(argv, "-np") == "1"
    assert "-t" not in argv and "-ngl" not in argv  # threads 0 / gpu_layers None: llama decides
    assert not any("api-key" in arg for arg in argv)  # the key travels in the environment
    assert start["llama_env"] == OUR_LLAMA_ENV

    [request] = fake.records("request")
    assert request["problems"] == []
    assert (request["width"], request["height"]) == (512, 384)
    assert (request["seed"], request["temperature"], request["max_tokens"]) == (0, 0.0, 512)
    proc.send({"type": "shutdown"})
    assert proc.wait() == 0


def test_threads_gpu_layers_and_extra_args_reach_llama_server(
    spawn: Callable[..., Proc], fake: FakeLlama
) -> None:
    proc = started(
        spawn,
        llama_config(fake, threads=3, gpu_layers=0, ctx=8192, server_args=("--flash-attn", "on")),
    )
    proc.send({"type": "shutdown"})
    proc.wait()
    [start] = fake.records("start")
    assert flag_value(start["argv"], "-t") == "3"
    assert flag_value(start["argv"], "-ngl") == "0"
    assert flag_value(start["argv"], "-c") == "8192"
    assert start["extras"] == ["--flash-attn", "on"]


def test_retries_vary_seed_and_temperature(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path
) -> None:
    proc = started(spawn, llama_config(fake))
    raws = [proc.ask(fixtures_dir / "plain.jpg", attempt=a)["raw"] for a in (0, 1, 2)]
    assert len(set(raws)) == 3
    sampled = [(r["seed"], r["temperature"]) for r in fake.records("request")]
    assert sampled == [(0, 0.0), (1, worker.RETRY_TEMPERATURE), (2, worker.RETRY_TEMPERATURE)]


def test_the_image_sent_is_capped_and_clean(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_MAX_EDGE", "300")
    proc = started(spawn, llama_config(fake, max_edge_px=300))
    for name in ("rotated.jpg", "exif.jpg", "user_xmp.jpg", "alpha.png"):
        assert proc.ask(fixtures_dir / name)["ok"] is True, name
    sizes = [(r["width"], r["height"], r["problems"]) for r in fake.records("request")]
    assert sizes == [(300, 225, []), (300, 225, []), (300, 225, []), (300, 225, [])]


def test_the_fake_server_really_checks_image_size(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards the guard: a worker that sent too big an image would be caught.
    monkeypatch.setenv("FAKE_LLAMA_MAX_EDGE", "300")
    proc = started(spawn, llama_config(fake, max_edge_px=600))
    result = proc.ask(fixtures_dir / "plain.jpg")
    assert (result["ok"], result["error"]) == (False, "server")
    assert "HTTP 400" in result["detail"] and "larger than 300" in result["detail"]


@pytest.mark.parametrize("mode", ["invalid-json", "schema-violation"])
def test_bad_model_output_is_returned_raw_for_the_orchestrator_to_reject(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path, mode: str
) -> None:
    proc = started(spawn, llama_config(fake, mode))
    result = proc.ask(fixtures_dir / "plain.jpg")
    assert result["ok"] is True  # the worker doesn't judge; it's untrusted anyway
    with pytest.raises(SchemaError):
        parse(result["raw"])


def test_http_errors_are_server_errors_and_the_worker_carries_on(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path
) -> None:
    proc = started(spawn, llama_config(fake, "http500"))
    for _ in range(2):
        result = proc.ask(fixtures_dir / "plain.jpg")
        assert (result["ok"], result["error"]) == (False, "server")
        assert result["detail"] == "llama-server answered HTTP 500: fake internal error for tests"
    proc.send({"type": "shutdown"})
    assert proc.wait() == 0


def test_a_slow_model_times_out(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_DELAY_S", "4")
    proc = started(spawn, llama_config(fake, "slow", request_timeout_s=1))
    began = time.monotonic()
    result = proc.ask(fixtures_dir / "plain.jpg")
    assert (result["ok"], result["error"]) == (False, "timeout")
    assert time.monotonic() - began < 3.5
    assert proc.p.poll() is None


def test_a_model_that_fails_to_load_is_fatal_with_the_log_tail(
    spawn: Callable[..., Proc], fake: FakeLlama
) -> None:
    proc = spawn()
    proc.send(llama_config(fake, "crash-on-load"))
    msg = proc.recv()
    assert (msg["type"], msg["error"]) == ("fatal", "server_exited")
    assert "error loading model: fake crash for tests" in msg["detail"]
    assert proc.wait() == 1


def test_a_model_that_never_loads_times_out(spawn: Callable[..., Proc], fake: FakeLlama) -> None:
    proc = spawn()
    proc.send(llama_config(fake, "never-ready", load_timeout_s=1))
    msg = proc.recv()
    assert (msg["type"], msg["error"]) == ("fatal", "load_timeout")
    assert "server is listening" in msg["detail"]  # the log tail
    assert proc.wait() == 1
    assert all(process_gone(pid) for pid in fake.pids())


def test_a_missing_llama_server_is_fatal(spawn: Callable[..., Proc], fake: FakeLlama) -> None:
    proc = spawn()
    proc.send(llama_config(fake, llama_server=str(fake.directory / "no-such-binary")))
    msg = proc.recv()
    assert (msg["type"], msg["error"]) == ("fatal", "missing_binary")
    assert "llama-server was not found" in msg["detail"]
    assert proc.wait() == 1


@pytest.mark.parametrize("which", ["model", "mmproj"])
def test_missing_model_files_are_fatal(
    spawn: Callable[..., Proc], fake: FakeLlama, which: str
) -> None:
    proc = spawn()
    proc.send(llama_config(fake, **{which: str(fake.directory / "gone.gguf")}))
    msg = proc.recv()
    assert (msg["type"], msg["error"]) == ("fatal", "missing_model")
    assert proc.wait() == 1
    assert fake.records() == []


@pytest.mark.parametrize(
    "args",
    [
        ("--host", "0.0.0.0"),
        ("--host=0.0.0.0",),
        ("--port", "80"),
        ("--api-key", "guessable"),
        ("--api-key-file", "/tmp/k"),
        ("-hf", "someone/some-model"),
        ("-m", "/somewhere/else.gguf"),
        ("--mmproj=/somewhere/else.gguf",),
        ("-mm", "/somewhere/else.gguf"),
        ("--model-url", "https://example.com/m.gguf"),
        ("--hf_repo", "someone/some-model"),
        ("-dr", "gemma3"),
        ("--rpc", "10.0.0.2:50052"),
        ("--connect",),
        ("--tools", "all"),
    ],
)
def test_server_args_cant_undo_the_isolation(
    spawn: Callable[..., Proc], fake: FakeLlama, args: tuple[str, ...]
) -> None:
    proc = spawn()
    proc.send(llama_config(fake, server_args=args))
    msg = proc.recv()
    assert (msg["type"], msg["error"]) == ("fatal", "bad_config")
    assert proc.wait() == 1
    assert fake.records() == []  # llama-server never ran


@pytest.mark.parametrize(
    "args",
    [
        # another model, projector or adapter than the ones As If chose and hashed
        ("--model=/x.gguf",),
        ("--mmproj_url", "https://example.com/p.gguf"),
        ("-mmu", "https://example.com/p.gguf"),
        ("--no-mmproj",),
        ("-md", "/x.gguf"),
        ("--spec-draft-model", "/x.gguf"),
        ("--lora", "/x.gguf"),
        ("--control-vector", "/x.gguf"),
        ("--override-kv", "general.name=str:x"),
        ("--models-dir", "/models"),
        # downloads
        ("-hf", "o/m"),
        ("-hfr", "o/m"),
        ("-hff", "m.gguf"),
        ("-hft", "hf_token"),
        ("-hfd", "o/d"),
        ("-hfrd", "o/d"),
        ("--hf-repo-draft", "o/d"),
        ("--spec-draft-hf", "o/d"),
        ("-mu", "https://example.com/m.gguf"),
        ("--docker-repo", "gemma3"),
        ("--vision-gemma-4b-default",),
        ("--gpt-oss-20b-default",),
        # other machines, tools, files
        ("--rpc", "10.0.0.2:50052"),
        ("--connect-code", "x" * 40),
        ("-ag",),
        ("--agent",),
        ("--tools-runtime", "ssh:me@elsewhere"),
        ("--mcp-servers-json", "{}"),
        ("--webui-mcp-proxy",),
        ("--path", "/"),
        ("--media-path", "/"),
        ("--video-ffmpeg-dir", "/tmp"),
        # where it listens, and with which key
        ("--host", "0.0.0.0"),
        ("--reuse-port",),
        ("--api_key", "k"),
        ("--api-key-file", "/k"),
        ("--ssl-key-file", "/k"),
        ("--api-prefix", "/x"),
        # prompts (photos included) written to disk
        ("--log-file", "/tmp/llama.log"),
        ("--log-prompts-dir", "/tmp"),
        ("--slot-save-path", "/tmp"),
        ("-lcd", "/tmp/cache"),
        # an option As If doesn't know yet, or one hidden among allowed ones
        ("--some-future-option",),
        ("--flash-attn", "on", "--rpc", "10.0.0.2:50052"),
    ],
)
def test_server_args_allow_only_known_safe_options(args: tuple[str, ...]) -> None:
    with pytest.raises(worker._Fatal) as info:
        worker._check_server_args(args)
    assert info.value.kind == "bad_config"


@pytest.mark.parametrize(
    "args",
    [
        ("--flash-attn", "on"),
        ("-fa",),
        ("--no-mmap", "--mlock"),
        ("-lm", "mmap+mlock"),
        ("-ctk", "q8_0", "-ctv", "q8_0"),
        ("--n_gpu_layers", "99"),  # llama.cpp reads _ as - in -- options
        ("-ot", r"\.ffn_.*_exps\.=CPU"),
        ("--cpu-moe", "--n-cpu-moe", "10"),
        ("--prio", "-1", "--reasoning-budget", "-1"),  # negative numbers are values
        ("-v",),
        ("-lv", "5"),
        ("--jinja", "--chat-template", "chatml"),
        ("--temp", "1.0", "--top-k", "64", "--top-p", "0.95", "--min-p", "0.0"),
        ("--image-max-tokens", "1024", "--no-mmproj-offload"),
        ("--offline", "--no-webui"),
    ],
)
def test_server_args_take_speed_memory_and_logging_options(args: tuple[str, ...]) -> None:
    worker._check_server_args(args)


@pytest.mark.parametrize(
    "args", [("--api-key=hunter2",), ("--api-key hunter2",), ("--hf-token", "hunter2")]
)
def test_a_refused_option_is_named_but_its_value_never_shown(args: tuple[str, ...]) -> None:
    with pytest.raises(worker._Fatal) as info:
        worker._check_server_args(args)
    assert "hunter2" not in info.value.detail
    assert args[0][:8] in info.value.detail


@pytest.mark.parametrize("build", BUILDS)
def test_llama_settings_in_the_environment_are_not_passed_on(
    spawn: Callable[..., Proc],
    fake: FakeLlama,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    build: str,
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_BUILD", build)
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    monkeypatch.setenv("LLAMA_ARG_HF_REPO", "someone/some-model")
    monkeypatch.setenv("LLAMA_API_KEY", "a-key-someone-else-knows")
    monkeypatch.setenv("LLAMA_ARG_API_KEY", "a-key-someone-else-knows")
    monkeypatch.setenv("LLAMA_ARG_OFFLINE", "0")
    monkeypatch.setenv("LLAMA_OFFLINE", "0")
    # Older builds' names for --log-file and -lv: a log file of every request, photo and all.
    monkeypatch.setenv("LLAMA_LOG_FILE", str(fake.directory / "llama.log"))
    monkeypatch.setenv("LLAMA_LOG_VERBOSITY", "5")
    monkeypatch.setenv("LLAMA_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": true}')
    monkeypatch.setenv("MTMD_BACKEND_DEVICE", "RPC0")
    monkeypatch.setenv("HF_TOKEN", "hf_someone")
    monkeypatch.setenv("HF_ENDPOINT", "https://example.com")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_someone")
    proc = started(spawn, llama_config(fake))  # the fake refuses to start if not offline
    assert proc.ask(fixtures_dir / "plain.jpg")["ok"] is True  # authorised with *our* key
    assert fake.records("start")[0]["llama_env"] == OUR_LLAMA_ENV


@pytest.mark.parametrize("where", ["XDG_CONFIG_HOME", "HOME"])
def test_the_users_llama_cpp_config_file_never_reaches_llama_server(
    spawn: Callable[..., Proc],
    fake: FakeLlama,
    fixtures_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    # Current llama-server builds apply this file before their environment
    # and command line; the fake refuses to start if it sees rpc= and the like.
    home = tmp_path / "home"
    if where == "HOME":
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        user_config = home / ".config"
    else:
        user_config = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(user_config))
    monkeypatch.setenv("HOME", str(home))
    (user_config / "llama.cpp").mkdir(parents=True)
    (user_config / "llama.cpp" / "config.ini").write_text(USER_LLAMA_CONFIG)

    proc = started(spawn, llama_config(fake))
    assert proc.ask(fixtures_dir / "plain.jpg")["ok"] is True
    [start] = fake.records("start")
    assert start["config_keys"] == []
    assert start["home"] == str(home)  # GPU drivers keep their caches there
    # llama-server looked in an empty folder only the worker can use...
    private = Path(start["config_home"])
    assert private != user_config
    assert os.listdir(private) == []
    assert private.stat().st_mode & 0o777 == 0o700
    proc.send({"type": "shutdown"})
    assert proc.wait() == 0
    assert not private.exists()  # ...and that is gone with it


def test_a_system_wide_llama_cpp_config_file_is_refused(
    spawn: Callable[..., Proc], fake: FakeLlama, tmp_path: Path
) -> None:
    # No environment hides /etc/llama.cpp/config.ini from llama-server.
    system = tmp_path / "etc" / "llama.cpp" / "config.ini"
    system.parent.mkdir(parents=True)
    system.write_text("rpc = 10.0.0.2:50052\n")
    code = (
        "from asif.infer import worker\n"
        f"worker.SYSTEM_LLAMA_CONFIG = {str(system)!r}\n"
        "raise SystemExit(worker.main())\n"
    )
    proc = spawn([sys.executable, "-c", code])
    proc.send(llama_config(fake))
    msg = proc.recv()
    assert (msg["type"], msg["error"]) == ("fatal", "bad_config")
    assert str(system) in msg["detail"] and "10.0.0.2" not in msg["detail"]
    assert proc.wait() == 1
    assert fake.records() == []  # llama-server never ran


def test_a_worker_whose_orchestrator_is_already_gone_starts_nothing(
    spawn: Callable[..., Proc], fake: FakeLlama
) -> None:
    # The orchestrator died before the worker could arm its death signal:
    # the worker now has another parent than the one named in the config.
    proc = spawn()
    proc.send({**llama_config(fake), "parent_pid": os.getppid()})
    assert proc.wait(timeout=10) == 1
    assert proc.raw == []
    assert fake.records() == []


def test_llama_server_dying_mid_run_is_fatal(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path
) -> None:
    proc = started(spawn, llama_config(fake, "die-on-request"))
    msg = proc.ask(fixtures_dir / "plain.jpg")
    assert (msg["type"], msg["error"]) == ("fatal", "server_exited")
    assert "GGML_ASSERT" in msg["detail"]
    assert proc.wait() == 1


def test_a_chatty_llama_server_cant_block_the_worker(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Far more than a pipe buffer: if nobody drained it, the fake would hang.
    monkeypatch.setenv("FAKE_LLAMA_SPEW_KB", "16384")
    proc = started(spawn, llama_config(fake, "spew"))
    for _ in range(3):
        assert proc.ask(fixtures_dir / "plain.jpg")["ok"] is True
    proc.send({"type": "shutdown"})
    assert proc.wait() == 0
    assert "llama_spew" not in proc.stderr()  # its log never reaches our streams
    assert all(json.loads(line)["type"] in ("ready", "result") for line in proc.raw)


# -- llama-server never outlives the worker ------------------------------------------


@linux_only
@pytest.mark.parametrize("how", ["shutdown", "eof", "sigterm", "sighup", "sigquit", "sigkill"])
def test_llama_server_never_outlives_the_worker(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path, how: str
) -> None:
    proc = started(spawn, llama_config(fake))
    assert proc.ask(fixtures_dir / "plain.jpg")["ok"] is True
    [pid] = fake.pids()
    assert not process_gone(pid)
    if how == "shutdown":
        proc.send({"type": "shutdown"})
    elif how == "eof":
        assert proc.p.stdin is not None
        proc.p.stdin.close()
    elif how == "sigkill":  # no cleanup code runs at all: PR_SET_PDEATHSIG does the work
        proc.p.kill()
    else:  # llama-server has its own session: a closed terminal or Ctrl+\ can't reach it
        proc.p.send_signal(getattr(signal, how.upper()))
    code = proc.wait()
    # 130: the signal unwound into the cleanup code, rather than ending the worker on the spot.
    assert code == {"shutdown": 0, "eof": 0, "sigkill": -9}.get(how, 130)
    assert wait_until(lambda: process_gone(pid)), f"llama-server ({pid}) outlived the worker"


@linux_only
def test_a_wrapper_script_without_exec_is_refused_and_stopped(
    spawn: Callable[..., Proc], fake: FakeLlama, tmp_path: Path
) -> None:
    # A common way to set GPU variables. Without exec, the death signal and
    # the port check reach only the shell, never llama-server itself.
    wrapper = tmp_path / "llama-server-wrapper"
    wrapper.write_text(
        "#!/bin/sh\n"
        "export CUDA_VISIBLE_DEVICES=0\n"
        f'{shlex.quote(sys.executable)} {shlex.quote(str(FAKE_SCRIPT))} "$@"\n'
        "exit $?\n"
    )
    wrapper.chmod(0o755)
    proc = spawn()
    proc.send(llama_config(fake, llama_server=str(wrapper)))
    msg = proc.recv()
    assert (msg["type"], msg["error"]) == ("fatal", "bad_config")
    assert "without exec" in msg["detail"]
    assert proc.wait() == 1
    [pid] = fake.pids()
    assert wait_until(lambda: process_gone(pid)), f"llama-server ({pid}) outlived the worker"


def _core_limit(pid: int) -> tuple[str, str]:
    for line in Path(f"/proc/{pid}/limits").read_text().splitlines():
        if line.startswith("Max core file size"):
            soft, hard = line.split()[4:6]
            return soft, hard
    raise AssertionError(f"no core file limit for {pid}")


@linux_only
def test_neither_the_worker_nor_llama_server_can_dump_core(
    spawn: Callable[..., Proc], fake: FakeLlama
) -> None:
    # Started the way a shell that allows core dumps would start it.
    code = (
        "import resource\n"
        "_, hard = resource.getrlimit(resource.RLIMIT_CORE)\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (hard, hard))\n"
        "from asif.infer import worker\n"
        "raise SystemExit(worker.main())\n"
    )
    proc = spawn([sys.executable, "-c", code])
    proc.send(llama_config(fake))
    assert proc.recv()["type"] == "ready"
    [pid] = fake.pids()
    # 0, or 1 byte: too small for a core file, and no crash reporter either.
    assert _core_limit(proc.p.pid) in {("0", "0"), ("1", "1")}
    assert _core_limit(pid) == _core_limit(proc.p.pid)  # inherited, and it can't raise it again


# -- nobody else gets the photos ------------------------------------------------------


def test_owns_port_asks_the_kernel_who_listens(spawn: Callable[..., Proc]) -> None:
    listener = spawn(
        [
            sys.executable,
            "-c",
            "import json, socket, sys, time\n"
            "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen()\n"
            "print(json.dumps({'port': s.getsockname()[1]}), flush=True)\n"
            "time.sleep(60)\n",
        ]
    )
    port = listener.recv()["port"]
    assert worker._owns_port(listener.p.pid, port)
    assert not worker._owns_port(os.getpid(), port)
    assert not worker._owns_port(listener.p.pid, _free_port())  # nobody listens there


class _Impostor(BaseHTTPRequestHandler):
    """Poses as llama-server: healthy, and hungry for photos."""

    posts: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self) -> None:
        self.posts.append(self.path)
        self.send_response(500)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


def test_a_program_squatting_on_the_port_gets_no_photos(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path, tmp_path: Path
) -> None:
    _Impostor.posts.clear()
    impostor = ThreadingHTTPServer(("127.0.0.1", 0), _Impostor)
    threading.Thread(target=impostor.serve_forever, daemon=True).start()
    port = impostor.server_address[1]
    # A llama-server that is slow to start, so the impostor answers first.
    slow = tmp_path / "slow-llama-server"
    slow.write_text(f'#!/bin/sh\nsleep 3\nexec {fake.binary} "$@"\n')
    slow.chmod(0o755)
    code = (
        "from asif.infer import worker\n"
        f"worker._free_port = lambda: {port}\n"
        "raise SystemExit(worker.main())\n"
    )
    try:
        proc = spawn([sys.executable, "-c", code])
        proc.send(llama_config(fake, llama_server=str(slow)))
        msg = proc.recv()
        assert (msg["type"], msg["error"]) == ("fatal", "port_taken")
        assert proc.wait() == 1
        assert _Impostor.posts == []
    finally:
        impostor.shutdown()
        impostor.server_close()


def test_a_squatter_on_a_dead_servers_port_gets_no_photos(
    spawn: Callable[..., Proc], fake: FakeLlama, fixtures_dir: Path
) -> None:
    _Impostor.posts.clear()
    proc = started(spawn, llama_config(fake))
    assert proc.ask(fixtures_dir / "plain.jpg")["ok"] is True
    [start] = fake.records("start")
    os.kill(start["pid"], signal.SIGKILL)  # llama-server dies between photos...
    assert wait_until(lambda: process_gone(start["pid"]))
    port = int(flag_value(start["argv"], "--port") or 0)
    impostor = ThreadingHTTPServer(("127.0.0.1", port), _Impostor)  # ...and its port is taken
    threading.Thread(target=impostor.serve_forever, daemon=True).start()
    try:
        msg = proc.ask(fixtures_dir / "gps.jpg")
        assert (msg["type"], msg["error"]) == ("fatal", "server_exited")
        assert proc.wait() == 1
        assert _Impostor.posts == []
    finally:
        impostor.shutdown()
        impostor.server_close()


# -- the request itself --------------------------------------------------------------


def test_request_body_matches_what_llama_server_expects(fixtures_dir: Path) -> None:
    cfg = worker._parse_config(llama_config_dict(lang="pt-BR"))
    backend = worker._LlamaBackend(cfg)
    backend.prompt = load_prompt("pt-BR")
    jpeg = prepare_image(fixtures_dir / "plain.jpg", 512)
    body = backend.request_body(jpeg, attempt=0)
    system, user = body["messages"]
    assert system == {"role": "system", "content": load_prompt("pt-BR")}
    assert "Brazilian Portuguese" in system["content"]
    image, text = user["content"]
    assert image["type"] == "image_url"
    url = image["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == jpeg
    assert text == {"type": "text", "text": worker.USER_TEXT}
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "caption", "strict": True, "schema": caption_schema()},
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["temperature"], body["seed"], body["max_tokens"]) == (0.0, 0, 512)
    retry = backend.request_body(jpeg, attempt=2)
    assert (retry["temperature"], retry["seed"]) == (worker.RETRY_TEMPERATURE, 2)


def llama_config_dict(**kw: Any) -> dict[str, Any]:
    return {"type": "config", "backend": "llama", "lang": "en", "max_edge_px": 512, **kw}


def test_log_tail_is_bounded_and_hides_image_data() -> None:
    tail = worker._LogTail(lines=3)
    read, write = os.pipe()
    with os.fdopen(write, "wb") as w:
        w.write(b"one\n" + b"x" * 10_000 + b"\nrequest: data:image/jpeg;base64,/9j/4AAQ\nlast\n")
    tail.follow(os.fdopen(read, "rb"))
    tail.join(5)
    lines = tail.tail().splitlines()
    assert len(lines) == 3
    assert lines[-2] == "[line hidden: it contained image data]"
    assert lines[-1] == "last"
    assert all(len(line) <= 500 for line in lines)


def _log_tail_of(data: bytes) -> str:
    tail = worker._LogTail()
    read, write = os.pipe()
    tail.follow(os.fdopen(read, "rb"))  # reading first: data larger than a pipe would block
    with os.fdopen(write, "wb") as w:
        w.write(data)
    tail.join(5)
    return tail.tail()


_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{50,}")
_AFTER = b"".join(b"line %d\n" % i for i in range(16))


def test_log_tail_hides_all_of_a_long_line_with_image_data() -> None:
    # A verbose llama-server logs each request body, photo and all, on one line:
    # far longer than one read, so it arrives in many chunks.
    photo = base64.b64encode(os.urandom(240_000))
    body = b'{"messages": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'
    shown = _log_tail_of(b"srv  log_server_r: request:  " + body + photo + b'"}}]}\n' + _AFTER)
    assert _BASE64_RUN.search(shown) is None
    lines = shown.splitlines()
    assert lines == ["[line hidden: it contained image data]"] + [f"line {i}" for i in range(16)]


def test_log_tail_hides_image_data_whose_marker_straddles_two_reads() -> None:
    head = (b"srv  log_server_r: request:  " + b'{"a": 1, ' * 500)[: 4096 - len(b"data:ima")]
    photo = base64.b64encode(os.urandom(30_000))
    shown = _log_tail_of(head + b"data:image/jpeg;base64," + photo + b'"}\n' + _AFTER)
    assert _BASE64_RUN.search(shown) is None
    assert shown.splitlines()[-16:] == [f"line {i}" for i in range(16)]


# -- the fake itself must be strict, or the tests above prove little ------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def _fake_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """The fake's environment as the worker gives it, plus ``extra``."""
    xdg = tmp_path / "fake-xdg"  # never the developer's own llama.cpp settings
    xdg.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "LLAMA_API_KEY": "k",
        "LLAMA_ARG_API_KEY": "k",
        "XDG_CONFIG_HOME": str(xdg),
    }
    env.update(LLAMA_OFFLINE="1", LLAMA_ARG_OFFLINE="1")
    env.update(extra)
    return env


def _start_fake(fake: FakeLlama, env: dict[str, str]) -> tuple[subprocess.Popen[bytes], int]:
    port = _free_port()
    model, mmproj = fake.model_files()
    argv = [str(fake.binary), "-m", str(model), "--mmproj", str(mmproj), "--port", str(port)]
    server = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_until(lambda: _get(port, "/health") is not None):
        server.kill()
        server.wait()
        raise AssertionError("the fake llama-server didn't start")
    return server, port


@pytest.fixture
def fake_server(fake: FakeLlama, tmp_path: Path) -> Iterator[tuple[int, str]]:
    key = "test-key-123"
    server, port = _start_fake(fake, _fake_env(tmp_path, LLAMA_API_KEY=key, LLAMA_ARG_API_KEY=key))
    try:
        yield port, key
    finally:
        server.kill()
        server.wait()


def _get(port: int, path: str) -> int | None:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        return conn.getresponse().status
    except OSError:
        return None
    finally:
        conn.close()


def _post(port: int, key: str, body: dict[str, Any]) -> tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read().decode()
    finally:
        conn.close()


def _body_for(jpeg: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    backend = worker._LlamaBackend(worker._parse_config(llama_config_dict()))
    backend.prompt = load_prompt("en")
    body = backend.request_body(jpeg, 0)
    url = f"data:{mime};base64," + base64.b64encode(jpeg).decode()
    body["messages"][1]["content"][0]["image_url"]["url"] = url
    return body


def test_fake_health_says_loading_first(fake_server: tuple[int, str]) -> None:
    port, _key = fake_server
    statuses = [_get(port, "/health") for _ in range(4)]
    assert 503 in statuses and statuses[-1] == 200


def test_fake_checks_key_image_and_schema(
    fake_server: tuple[int, str], fixtures_dir: Path, tmp_path: Path
) -> None:
    port, key = fake_server
    good = prepare_image(fixtures_dir / "plain.jpg", 512)
    assert _post(port, key, _body_for(good))[0] == 200
    assert _post(port, "wrong-key", _body_for(good))[0] == 401

    status, text = _post(port, key, _body_for(good, mime="image/png"))
    assert status == 400 and "data:image/jpeg" in text

    with_exif = (fixtures_dir / "exif.jpg").read_bytes()  # the original, metadata and all
    status, text = _post(port, key, _body_for(with_exif))
    assert status == 400 and "metadata" in text

    status, text = _post(port, key, _body_for(good[: len(good) // 2]))
    assert status == 400 and "valid JPEG" in text

    big = tmp_path / "big.png"
    Image.new("RGB", (1000, 800), "red").save(big)
    status, text = _post(port, key, _body_for(prepare_image(big, 1000)))
    assert status == 400 and "larger than 512" in text

    body = _body_for(good)
    body["response_format"]["json_schema"]["schema"] = {"type": "object"}
    del body["chat_template_kwargs"]
    body["messages"][0]["content"] = " "
    status, text = _post(port, key, body)
    assert status == 400
    for problem in ("JSON schema", "thinking", "system prompt"):
        assert problem in text


@pytest.mark.parametrize(
    ("extra", "env"),
    [
        (["--host", "0.0.0.0"], {}),
        (["--host", "192.168.1.10"], {}),
        (["--api-key", "k"], {}),
        ([], {"LLAMA_ARG_OFFLINE": "0"}),
        ([], {"FAKE_LLAMA_BUILD": "11000", "LLAMA_ARG_OFFLINE": ""}),  # LLAMA_OFFLINE is ignored
        ([], {"FAKE_LLAMA_BUILD": "9200", "LLAMA_OFFLINE": ""}),  # LLAMA_ARG_OFFLINE is ignored
    ],
)
def test_fake_refuses_exposed_hosts_argv_keys_and_online_mode(
    fake: FakeLlama, tmp_path: Path, extra: list[str], env: dict[str, str]
) -> None:
    model, mmproj = fake.model_files()
    argv = [str(fake.binary), "-m", str(model), "--mmproj", str(mmproj), *extra]
    proc = subprocess.run(
        argv, env=_fake_env(tmp_path, **env), capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 2
    assert "error" in proc.stdout


@pytest.mark.parametrize("where", ["system", "XDG_CONFIG_HOME", "HOME"])
def test_fake_applies_llama_cpp_config_files_like_current_builds(
    fake: FakeLlama, tmp_path: Path, where: str
) -> None:
    files = {
        "system": tmp_path / "etc" / "config.ini",
        "XDG_CONFIG_HOME": tmp_path / "fake-xdg" / "llama.cpp" / "config.ini",
        "HOME": tmp_path / "home" / ".config" / "llama.cpp" / "config.ini",
    }
    env = _fake_env(tmp_path, FAKE_LLAMA_SYSTEM_CONFIG=str(files["system"]))
    env["HOME"] = str(tmp_path / "home")
    if where == "HOME":
        del env["XDG_CONFIG_HOME"]
    files[where].parent.mkdir(parents=True, exist_ok=True)
    files[where].write_text(USER_LLAMA_CONFIG)
    model, mmproj = fake.model_files()
    argv = [str(fake.binary), "-m", str(model), "--mmproj", str(mmproj)]
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
    assert "config.ini sets log-file, rpc, verbose" in proc.stdout
    # A build from before config.ini files ignores them.
    env["FAKE_LLAMA_BUILD"] = "9200"
    server, _port = _start_fake(fake, env)
    server.kill()
    server.wait()


def test_fake_takes_a_config_files_api_key_as_well(
    fake: FakeLlama, fixtures_dir: Path, tmp_path: Path
) -> None:
    # As llama.cpp does: each source adds a key, so another program could use it.
    env = _fake_env(tmp_path)
    (tmp_path / "fake-xdg" / "llama.cpp").mkdir(parents=True)
    (tmp_path / "fake-xdg" / "llama.cpp" / "config.ini").write_text("api-key = theirs\n")
    server, port = _start_fake(fake, env)
    try:
        body = _body_for(prepare_image(fixtures_dir / "plain.jpg", 512))
        assert _post(port, "k", body)[0] == 200
        assert _post(port, "theirs", body)[0] == 200
        assert _post(port, "nobody's", body)[0] == 401
    finally:
        server.kill()
        server.wait()
