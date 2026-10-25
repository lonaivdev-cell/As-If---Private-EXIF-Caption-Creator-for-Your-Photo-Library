from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest

from asif.pipeline.timings import TimingsLog, new_run_id, prune


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_events_have_the_documented_shape(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    log = TimingsLog(path, run_id="r_1234", model="stub")
    with log.timed("infer", "i_000001") as ev:
        ev["ok"] = True
    (event,) = _events(path)
    assert event["run"] == "r_1234"
    assert event["img"] == "i_000001"
    assert event["stage"] == "infer"
    assert event["model"] == "stub"
    assert isinstance(event["ms"], int)
    assert event["ok"] is True
    assert path.stat().st_mode & 0o777 == 0o600


def test_unknown_fields_are_dropped_so_paths_cannot_leak(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    log = TimingsLog(path, run_id="r_1", model="m")
    log.event("open", img="i_1", path="/home/me/Pictures/secret.jpg", caption="my kid")
    text = path.read_text()
    assert "secret" not in text
    assert "my kid" not in text


def test_failures_are_recorded_with_the_error_class_only(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    log = TimingsLog(path, run_id="r_1", model="m")
    with pytest.raises(ValueError), log.timed("write", "i_2"):
        raise ValueError("/home/me/secret.jpg is broken")
    (event,) = _events(path)
    assert event["ok"] is False
    assert event["error"] == "ValueError"
    assert "secret" not in path.read_text()


def test_disabled_log_writes_nothing(tmp_path: Path) -> None:
    log = TimingsLog(None, run_id="r_1", model="m")
    log.event("open")
    with log.timed("infer"):
        pass
    assert list(tmp_path.iterdir()) == []


def test_a_write_that_fails_mid_run_stops_the_log_not_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "t.jsonl"
    log = TimingsLog(path, run_id="r_1", model="m")
    log.event("run_start")
    real_open = Path.open

    def full_disk(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == path:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", full_disk)
    with pytest.raises(ValueError, match="the stage's own"), log.timed("write", "i_1"):
        raise ValueError("the stage's own error")  # not replaced by the log's
    with log.timed("commit", "i_1"):
        pass
    assert log.problem == "No space left on device"
    monkeypatch.setattr(Path, "open", real_open)  # space again: the log stays off for this run
    log.event("run_end")
    assert [e["stage"] for e in _events(path)] == ["run_start"]
    assert log.problem == "No space left on device"


def test_prune_drops_old_and_garbled_lines(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text('{"ts":"2001-01-01T00:00:00Z","stage":"old"}\nnot json\n')
    TimingsLog(path, run_id="r", model="m").event("new")
    assert prune(path, retention_days=30) == 2
    assert [e["stage"] for e in _events(path)] == ["new"]
    assert prune(tmp_path / "missing.jsonl", 30) == 0


def test_run_ids_are_random() -> None:
    assert new_run_id() != new_run_id()
    assert new_run_id().startswith("r_")
