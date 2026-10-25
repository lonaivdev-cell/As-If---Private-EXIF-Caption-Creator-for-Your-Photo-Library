"""The minimal timings log (M1). Replaced by the encrypted telemetry store in M9.

One JSON object per line, in the shape of §14 of docs/DESIGN.md. It never
contains file names, paths, sizes, captions or keywords: an image is only
ever a per-run counter (``i_000123``) over a randomly shuffled processing
order, and nothing records which file got which number. But the captioned
copies' own file times show the order they were written in, so anyone who
can see those copies could line them up with the log's timestamps: treat
the log as no less private than the destination folder. The encrypted
telemetry store (M9) replaces this log.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Keys an event may carry. Anything else is dropped, so a careless caller
# can't leak a path into the log by passing it as an extra field.
_ALLOWED = frozenset({"ts", "run", "img", "stage", "ms", "ok", "model", "error", "state", "n"})


def new_run_id() -> str:
    return "r_" + secrets.token_hex(4)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class TimingsLog:
    """Append-only JSONL writer. ``path=None`` disables logging entirely."""

    def __init__(self, path: Path | None, run_id: str, model: str) -> None:
        self.path = path
        self.run_id = run_id
        self.model = model
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.close(fd)

    def event(self, stage: str, **fields: Any) -> None:
        if self.path is None:
            return
        record: dict[str, Any] = {"ts": _now(), "run": self.run_id, "stage": stage}
        record.update({k: v for k, v in fields.items() if k in _ALLOWED})
        record.setdefault("model", self.model)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    @contextmanager
    def timed(self, stage: str, img: str | None = None) -> Iterator[dict[str, Any]]:
        """Time a block. The yielded dict can carry extra fields (e.g. ``ok``)."""
        extra: dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield extra
        except BaseException as exc:
            extra.setdefault("ok", False)
            extra.setdefault("error", type(exc).__name__)
            raise
        finally:
            ms = round((time.perf_counter() - start) * 1000)
            extra.setdefault("ok", True)
            if img is not None:
                extra["img"] = img
            self.event(stage, ms=ms, **extra)


def prune(path: Path, retention_days: int) -> int:
    """Drop events older than ``retention_days``. Returns how many were dropped."""
    if not path.exists():
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    kept: list[str] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ts = datetime.strptime(json.loads(line)["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except (ValueError, KeyError, TypeError):
            dropped += 1
            continue
        if ts >= cutoff:
            kept.append(line)
        else:
            dropped += 1
    if dropped:
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("".join(f"{line}\n" for line in kept))
        os.replace(tmp, path)
    return dropped
