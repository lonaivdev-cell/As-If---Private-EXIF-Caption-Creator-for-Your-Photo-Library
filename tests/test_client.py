"""The orchestrator's side of the inference pipe.

Most tests run the real worker (stub backend, or llama against the fake
llama-server). Misbehaving workers are small scripts that speak the protocol
wrongly on purpose: the client must never hang, never attribute an answer to
the wrong photo, and never let a worker's chatter fill a pipe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from asif.infer import client as client_mod
from asif.infer import worker
from asif.infer.client import InferenceClient, InferenceError, WorkerConfig
from asif.infer.schema import PROMPT_VERSION, parse_caption_json
from tests.fake_llama_server import FakeLlama, process_gone
from tests.test_netguard import forbidden_imports

SCRIPTED_WORKER = r"""
import json, os, signal, sys, time

behaviour = sys.argv[1]
out = sys.stdout


def send(obj):
    out.write(json.dumps(obj) + "\n")
    out.flush()


def spew(mib):
    line = "noise " * 100 + "\n"
    for _ in range(mib * 1024 * 1024 // len(line)):
        sys.stderr.write(line)
    sys.stderr.flush()


config = json.loads(sys.stdin.readline())
if behaviour == "exit_before_ready":
    sys.stderr.write("Traceback (most recent call last):\n")
    sys.stderr.write("ModuleNotFoundError: No module named 'asif'\n")
    sys.exit(3)
if behaviour == "garbage_instead_of_ready":
    out.write("hello there\n")
    out.flush()
    time.sleep(60)
if behaviour == "never_ready":
    time.sleep(60)
if behaviour == "fatal_on_start":
    send({"type": "fatal", "error": "missing_binary", "detail": "llama-server was not found"})
    sys.exit(1)
if behaviour == "ignores_shutdown":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if behaviour == "spew":
    spew(8)
send({"type": "ready", "backend": config["backend"], "prompt_version": "caption_v1"})

for line in sys.stdin:
    msg = json.loads(line)
    if msg["type"] == "shutdown":
        if behaviour == "ignores_shutdown":
            time.sleep(60)
        sys.exit(0)
    rid = msg["id"]
    if behaviour == "silent":
        time.sleep(60)
    elif behaviour == "wrong_id":
        send({"type": "result", "id": "i_999999", "ok": True, "raw": "{}", "ms": 1})
    elif behaviour == "not_json":
        out.write("hello\n")
        out.flush()
    elif behaviour == "no_ok":
        send({"type": "result", "id": rid})
    elif behaviour == "weird_error":
        send({"type": "result", "id": rid, "ok": False, "error": "weird", "detail": "x"})
    elif behaviour == "ok_without_raw":
        send({"type": "result", "id": rid, "ok": True})
    elif behaviour == "ready_again":
        send({"type": "ready", "backend": "stub", "prompt_version": "caption_v1"})
    elif behaviour == "huge_line":
        out.write("x" * (5 << 20))
        out.flush()
        time.sleep(60)
    elif behaviour == "exit_on_describe":
        sys.stderr.write("boom: the worker fell over\n")
        sys.stderr.flush()
        os._exit(3)
    elif behaviour == "fatal_mid_run":
        send({"type": "fatal", "error": "server_exited", "detail": "llama-server stopped"})
        sys.exit(1)
    elif behaviour == "close_stdout":
        os.close(1)
        time.sleep(60)
    else:
        if behaviour == "spew":
            spew(4)
        send({"type": "result", "id": rid, "ok": True, "raw": "echo:" + msg["image"], "ms": 1})
"""


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the client's safety margins so failure tests finish quickly."""
    monkeypatch.setattr(client_mod, "SLACK_S", 0.5)
    monkeypatch.setattr(client_mod, "SHUTDOWN_GRACE_S", 1.0)
    monkeypatch.setattr(client_mod, "TERM_GRACE_S", 1.0)


@pytest.fixture
def clients() -> Iterator[Callable[[WorkerConfig], InferenceClient]]:
    made: list[InferenceClient] = []

    def make(config: WorkerConfig) -> InferenceClient:
        made.append(InferenceClient(config))
        return made[-1]

    yield make
    for c in made:
        c.close()


@pytest.fixture
def scripted(tmp_path: Path) -> Callable[..., WorkerConfig]:
    script = tmp_path / "scripted_worker.py"
    script.write_text(SCRIPTED_WORKER)

    def config(behaviour: str, **kw: Any) -> WorkerConfig:
        settings: dict[str, Any] = {"load_timeout_s": 10, "request_timeout_s": 10}
        settings.update(kw)
        return WorkerConfig(
            backend="stub",
            lang="en",
            max_edge_px=512,
            command=(sys.executable, str(script), behaviour),
            **settings,
        )

    return config


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeLlama:
    fake = FakeLlama(tmp_path / "llama")
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(fake.record_path))
    monkeypatch.setenv("FAKE_LLAMA_MAX_EDGE", "512")
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    return fake


def stub(**kw: Any) -> WorkerConfig:
    return WorkerConfig(backend="stub", lang="en", max_edge_px=512, **kw)


def llama(fake: FakeLlama, mode: str = "good", **kw: Any) -> WorkerConfig:
    model, mmproj = fake.model_files(mode)
    settings: dict[str, Any] = {
        "llama_server": str(fake.binary),
        "model": str(model),
        "mmproj": str(mmproj),
        "load_timeout_s": 30,
        "request_timeout_s": 30,
    }
    settings.update(kw)
    return WorkerConfig(backend="llama", lang="en", max_edge_px=512, **settings)


# -- the real worker ------------------------------------------------------------------


def test_start_describe_close_with_the_stub(
    clients: Callable[[WorkerConfig], InferenceClient], fixtures_dir: Path
) -> None:
    c = clients(stub())
    assert not c.alive
    ready = c.start()
    assert ready == {"type": "ready", "backend": "stub", "prompt_version": PROMPT_VERSION}
    assert c.alive
    caption = parse_caption_json(c.describe(fixtures_dir / "plain.jpg", "i_000001"))
    assert "stub" in caption.keywords
    retry = c.describe(fixtures_dir / "plain.jpg", "i_000001", attempt=1)
    assert parse_caption_json(retry) == caption
    c.close()
    assert not c.alive
    c.close()  # idempotent
    with pytest.raises(InferenceError) as info:
        c.describe(fixtures_dir / "plain.jpg", "i_000002")
    assert info.value.kind == "dead"


def test_context_manager_starts_and_closes(fixtures_dir: Path) -> None:
    with InferenceClient(stub()) as c:
        assert c.alive
        assert c.describe(fixtures_dir / "exif.jpg", "i_000001")
    assert not c.alive


def test_a_client_starts_once(clients: Callable[[WorkerConfig], InferenceClient]) -> None:
    c = clients(stub())
    c.start()
    with pytest.raises(RuntimeError):
        c.start()
    closed = clients(stub())
    closed.close()  # closing a client that never started is fine
    with pytest.raises(RuntimeError):
        closed.start()


def test_describe_before_start_is_dead(
    clients: Callable[[WorkerConfig], InferenceClient], fixtures_dir: Path
) -> None:
    with pytest.raises(InferenceError) as info:
        clients(stub()).describe(fixtures_dir / "plain.jpg", "i_000001")
    assert info.value.kind == "dead"


def test_worker_errors_keep_the_worker_alive(
    clients: Callable[[WorkerConfig], InferenceClient], fixtures_dir: Path, tmp_path: Path
) -> None:
    c = clients(stub())
    c.start()
    with pytest.raises(InferenceError) as info:
        c.describe(fixtures_dir / "truncated.jpg", "i_000001")
    assert info.value.kind == "image_decode"
    assert "truncated" in info.value.detail
    with pytest.raises(InferenceError) as info:
        c.describe(tmp_path / "missing.jpg", "i_000002")
    assert info.value.kind == "bad_request"
    assert c.alive
    assert c.describe(fixtures_dir / "plain.jpg", "i_000003")


def test_relative_paths_are_made_absolute(
    clients: Callable[[WorkerConfig], InferenceClient],
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(fixtures_dir)
    c = clients(stub())
    c.start()
    assert "stub" in parse_caption_json(c.describe(Path("plain.jpg"), "i_000001")).keywords


def test_a_killed_worker_is_dead(
    clients: Callable[[WorkerConfig], InferenceClient], fixtures_dir: Path
) -> None:
    c = clients(stub())
    c.start()
    assert c._proc is not None
    c._proc.kill()
    c._proc.wait()
    assert not c.alive
    with pytest.raises(InferenceError) as info:
        c.describe(fixtures_dir / "plain.jpg", "i_000001")
    assert info.value.kind == "dead"
    c.close()


def test_the_worker_doesnt_import_from_the_current_directory(
    clients: Callable[[WorkerConfig], InferenceClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A folder of downloads with an "asif" package in it must not hijack the worker.
    planted = tmp_path / "downloads" / "asif"
    (planted / "infer").mkdir(parents=True)
    marker = tmp_path / "pwned"
    payload = f"open({str(marker)!r}, 'w').close()\n"
    (planted / "__init__.py").write_text(payload)
    (planted / "infer" / "__init__.py").write_text(payload)
    (planted / "infer" / "worker.py").write_text(payload)
    monkeypatch.chdir(planted.parent)
    c = clients(stub())
    assert c.start()["type"] == "ready"
    assert not marker.exists()
    assert client_mod.DEFAULT_COMMAND[1] == "-I"


def test_the_config_message_is_what_the_worker_parses(fake: FakeLlama) -> None:
    config = llama(fake, threads=2, gpu_layers=10, server_args=("--flash-attn", "on"))
    parsed = worker._parse_config(json.loads(json.dumps(config.to_message())))
    assert parsed.backend == "llama"
    assert (parsed.threads, parsed.gpu_layers) == (2, 10)
    assert parsed.server_args == ("--flash-attn", "on")
    assert parsed.model == config.model and parsed.mmproj == config.mmproj
    assert worker._parse_config(stub().to_message()).backend == "stub"


# -- the real worker with llama, against the fake llama-server ------------------------


def test_llama_start_describe_close(
    clients: Callable[[WorkerConfig], InferenceClient], fake: FakeLlama, fixtures_dir: Path
) -> None:
    c = clients(llama(fake))
    assert c.start()["backend"] == "llama"
    raws = [c.describe(fixtures_dir / "gps.jpg", "i_000001", attempt=a) for a in (0, 1)]
    assert [parse_caption_json(r).caption for r in raws] == [
        "A fake caption for a 512x384 image (seed 0, temperature 0.0).",
        "A fake caption for a 512x384 image (seed 1, temperature 0.3).",
    ]
    [pid] = fake.pids()
    c.close()
    assert process_gone(pid)


def test_llama_start_failures_explain_themselves(
    clients: Callable[[WorkerConfig], InferenceClient], fake: FakeLlama
) -> None:
    c = clients(llama(fake, llama_server=str(fake.directory / "missing")))
    with pytest.raises(InferenceError) as info:
        c.start()
    assert info.value.kind == "start"
    assert info.value.detail.startswith("missing_binary: llama-server was not found")
    assert not c.alive

    c = clients(llama(fake, "crash-on-load"))
    with pytest.raises(InferenceError) as info:
        c.start()
    assert info.value.kind == "start"
    assert "server_exited" in info.value.detail
    assert "error loading model: fake crash for tests" in info.value.detail


def test_a_model_timeout_is_reported_by_the_worker_which_stays_alive(
    clients: Callable[[WorkerConfig], InferenceClient],
    fake: FakeLlama,
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_LLAMA_DELAY_S", "4")
    c = clients(llama(fake, "slow", request_timeout_s=1))
    c.start()
    with pytest.raises(InferenceError) as info:
        c.describe(fixtures_dir / "plain.jpg", "i_000001")
    assert info.value.kind == "timeout"
    assert c.alive


def test_llama_server_dying_makes_the_client_dead(
    clients: Callable[[WorkerConfig], InferenceClient], fake: FakeLlama, fixtures_dir: Path
) -> None:
    c = clients(llama(fake, "die-on-request"))
    c.start()
    with pytest.raises(InferenceError) as info:
        c.describe(fixtures_dir / "plain.jpg", "i_000001")
    assert info.value.kind == "dead"
    assert "server_exited" in info.value.detail and "GGML_ASSERT" in info.value.detail
    assert not c.alive


# -- start failures -------------------------------------------------------------------


@pytest.mark.usefixtures("fast")
@pytest.mark.parametrize(
    ("behaviour", "expected"),
    [
        ("exit_before_ready", "No module named 'asif'"),
        ("garbage_instead_of_ready", "isn't JSON"),
        ("fatal_on_start", "missing_binary: llama-server was not found"),
    ],
)
def test_start_failures_are_start_errors(
    clients: Callable[[WorkerConfig], InferenceClient],
    scripted: Callable[..., WorkerConfig],
    behaviour: str,
    expected: str,
) -> None:
    c = clients(scripted(behaviour))
    began = time.monotonic()
    with pytest.raises(InferenceError) as info:
        c.start()
    assert time.monotonic() - began < 5
    assert info.value.kind == "start"
    assert expected in info.value.detail
    assert not c.alive


@pytest.mark.usefixtures("fast")
def test_start_times_out(
    clients: Callable[[WorkerConfig], InferenceClient], scripted: Callable[..., WorkerConfig]
) -> None:
    c = clients(scripted("never_ready", load_timeout_s=1))
    began = time.monotonic()
    with pytest.raises(InferenceError) as info:
        c.start()
    assert 1.4 <= time.monotonic() - began < 5  # load_timeout_s + SLACK_S
    assert info.value.kind == "start"
    assert c._proc is not None and c._proc.poll() is not None


def test_a_worker_that_cant_be_run_is_a_start_error(
    clients: Callable[[WorkerConfig], InferenceClient], tmp_path: Path
) -> None:
    c = clients(stub(command=(str(tmp_path / "no-such-python"),)))
    with pytest.raises(InferenceError) as info:
        c.start()
    assert info.value.kind == "start"


# -- misbehaving workers --------------------------------------------------------------


@pytest.mark.usefixtures("fast")
def test_no_answer_in_time_kills_the_worker(
    clients: Callable[[WorkerConfig], InferenceClient], scripted: Callable[..., WorkerConfig]
) -> None:
    c = clients(scripted("silent", request_timeout_s=1))
    c.start()
    began = time.monotonic()
    with pytest.raises(InferenceError) as info:
        c.describe(Path("/x.jpg"), "i_000001")
    assert 1.4 <= time.monotonic() - began < 5
    assert info.value.kind == "timeout"
    assert not c.alive
    assert c._proc is not None and c._proc.poll() is not None
    with pytest.raises(InferenceError) as info:
        c.describe(Path("/x.jpg"), "i_000002")
    assert info.value.kind == "dead"


@pytest.mark.usefixtures("fast")
@pytest.mark.parametrize(
    "behaviour",
    ["wrong_id", "not_json", "no_ok", "weird_error", "ok_without_raw", "ready_again", "huge_line"],
)
def test_protocol_violations_kill_the_worker(
    clients: Callable[[WorkerConfig], InferenceClient],
    scripted: Callable[..., WorkerConfig],
    behaviour: str,
) -> None:
    c = clients(scripted(behaviour))
    c.start()
    with pytest.raises(InferenceError) as info:
        c.describe(Path("/x.jpg"), "i_000001")
    assert info.value.kind == "protocol"
    assert not c.alive  # its next line can't be trusted to be about the next photo


@pytest.mark.usefixtures("fast")
def test_a_worker_that_exits_mid_request_is_dead(
    clients: Callable[[WorkerConfig], InferenceClient], scripted: Callable[..., WorkerConfig]
) -> None:
    c = clients(scripted("exit_on_describe"))
    c.start()
    with pytest.raises(InferenceError) as info:
        c.describe(Path("/x.jpg"), "i_000001")
    assert info.value.kind == "dead"
    assert "code 3" in info.value.detail
    assert "boom: the worker fell over" in info.value.detail
    assert not c.alive


@pytest.mark.usefixtures("fast")
@pytest.mark.parametrize("behaviour", ["fatal_mid_run", "close_stdout"])
def test_a_worker_that_gives_up_is_dead(
    clients: Callable[[WorkerConfig], InferenceClient],
    scripted: Callable[..., WorkerConfig],
    behaviour: str,
) -> None:
    c = clients(scripted(behaviour))
    c.start()
    with pytest.raises(InferenceError) as info:
        c.describe(Path("/x.jpg"), "i_000001")
    assert info.value.kind == "dead"
    assert not c.alive


def test_paths_reach_the_worker_absolute(
    clients: Callable[[WorkerConfig], InferenceClient],
    scripted: Callable[..., WorkerConfig],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    c = clients(scripted("echo"))
    c.start()
    assert c.describe(Path("photo.jpg"), "i_000001") == f"echo:{tmp_path / 'photo.jpg'}"


@pytest.mark.usefixtures("fast")
def test_close_escalates_when_the_worker_ignores_shutdown(
    clients: Callable[[WorkerConfig], InferenceClient], scripted: Callable[..., WorkerConfig]
) -> None:
    c = clients(scripted("ignores_shutdown"))
    c.start()
    proc = c._proc
    assert proc is not None
    began = time.monotonic()
    c.close()
    assert time.monotonic() - began < 5
    assert proc.returncode == -9  # it ignored SIGTERM too, so it got SIGKILL


def test_a_spewing_worker_cant_deadlock_the_client(
    clients: Callable[[WorkerConfig], InferenceClient], scripted: Callable[..., WorkerConfig]
) -> None:
    # 8 MiB of stderr before "ready" and 4 MiB per answer: hundreds of pipe buffers.
    c = clients(scripted("spew"))
    c.start()
    for i in range(3):
        assert c.describe(Path("/x.jpg"), f"i_{i:06d}") == "echo:/x.jpg"
    tail = c.stderr_tail()
    assert 0 < len(tail) <= client_mod.STDERR_LINES
    assert tail[-1].startswith("noise")


@pytest.mark.usefixtures("fast")
def test_error_details_carry_a_bounded_stderr_tail(
    clients: Callable[[WorkerConfig], InferenceClient], scripted: Callable[..., WorkerConfig]
) -> None:
    c = clients(scripted("exit_on_describe"))
    c.start()
    with pytest.raises(InferenceError) as info:
        c.describe(Path("/x.jpg"), "i_000001")
    assert len(info.value.detail.splitlines()) <= client_mod.DETAIL_LINES + 2


# -- no networking on the orchestrator side -------------------------------------------


@pytest.mark.parametrize("module", ["asif.infer.client", "asif.infer.preprocess"])
def test_importing_does_not_load_networking_modules(module: str) -> None:
    imported, bad = forbidden_imports(module)
    assert imported == [module]
    assert bad == []


def test_the_client_runs_under_the_netguard_tripwire(fixtures_dir: Path) -> None:
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        from asif import netguard
        from asif.infer.client import InferenceClient, WorkerConfig
        netguard.install()
        with InferenceClient(WorkerConfig(backend="stub", lang="en", max_edge_px=256)) as c:
            print(c.describe(Path({str(fixtures_dir / "plain.jpg")!r}), "i_000001"))
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_LLAMA")}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "stub" in parse_caption_json(proc.stdout.strip()).keywords
