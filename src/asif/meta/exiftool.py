"""A small, strict wrapper around one long-lived ``exiftool -stay_open`` process.

Spawning exiftool per file costs ~100-200 ms of Perl start-up; one
``-stay_open`` process makes that a one-off. Commands go in over stdin, one
argument per line, so any argument containing a newline is refused: it
would let text (say, a model's caption) smuggle in extra options.

Not thread-safe: one command at a time.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

NS_CONFIG = Path(__file__).with_name("asif_ns.config")

COMMON_ARGS = ("-charset", "filename=utf8", "-api", "largefilesupport=1")

_FORBIDDEN_IN_ARGS = ("\n", "\r", "\0")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ExifToolError(RuntimeError):
    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    warnings: list[str] = field(default_factory=list)


def check_arg(arg: str) -> str:
    """Refuse arguments that could break exiftool's one-per-line protocol."""
    if any(ch in arg for ch in _FORBIDDEN_IN_ARGS):
        raise ValueError("exiftool argument contains a line break or NUL; refusing to send it")
    return arg


class ExifTool:
    def __init__(
        self,
        executable: str = "exiftool",
        *,
        config: Path | None = NS_CONFIG,
        timeout_s: float = 120.0,
    ) -> None:
        self.executable = executable
        self.config = config
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._counter = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.alive:
            return
        argv = [self.executable]
        if self.config is not None:
            argv += ["-config", str(self.config)]
        argv += ["-stay_open", "True", "-@", "-"]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ExifToolError(
                f"exiftool not found ({self.executable!r}); run `asif doctor`"
            ) from exc

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(b"-stay_open\nFalse\n")
                proc.stdin.flush()
                proc.stdin.close()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
        finally:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()

    def __enter__(self) -> ExifTool:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- commands ----------------------------------------------------------

    def execute(self, *args: str) -> ExecResult:
        for arg in args:
            check_arg(arg)
        if not self.alive:
            self.start()
        proc = self._proc
        assert proc is not None and proc.stdin and proc.stdout and proc.stderr

        self._counter += 1
        marker = f"{{ready{self._counter}}}"
        lines = [*args, *COMMON_ARGS, "-echo4", marker, f"-execute{self._counter}"]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except OSError as exc:
            self._kill()
            raise ExifToolError(f"exiftool died: {exc}") from exc

        out, err = self._read_until(marker.encode())
        stderr = err.decode("utf-8", "replace")
        stdout = out.decode("utf-8", "replace")
        errors = [line for line in stderr.splitlines() if line.startswith("Error")]
        if errors:
            raise ExifToolError(errors[0], stderr)
        warnings = [line for line in stderr.splitlines() if line.startswith("Warning")]
        return ExecResult(stdout=stdout, stderr=stderr, warnings=warnings)

    def _read_until(self, marker: bytes) -> tuple[bytes, bytes]:
        """Read stdout and stderr until each ends with ``marker`` (+ newline)."""
        proc = self._proc
        assert proc is not None and proc.stdout and proc.stderr
        bufs = {"out": bytearray(), "err": bytearray()}
        done = {"out": False, "err": False}
        deadline = time.monotonic() + self.timeout_s
        with selectors.DefaultSelector() as sel:
            sel.register(proc.stdout, selectors.EVENT_READ, "out")
            sel.register(proc.stderr, selectors.EVENT_READ, "err")
            while not all(done.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill()
                    raise ExifToolError(f"exiftool timed out after {self.timeout_s:.0f} s")
                for key, _ in sel.select(timeout=remaining):
                    name = key.data
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        self._kill()
                        raise ExifToolError(
                            "exiftool exited unexpectedly", bufs["err"].decode("utf-8", "replace")
                        )
                    buf = bufs[name]
                    buf += chunk
                    stripped = buf.rstrip(b"\r\n")
                    if stripped.endswith(marker):
                        done[name] = True
                        del buf[len(stripped) - len(marker) :]
                        sel.unregister(key.fileobj)
        return bytes(bufs["out"]), bytes(bufs["err"])

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            proc.wait()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()

    # -- helpers -----------------------------------------------------------

    def execute_json(self, *args: str) -> list[dict[str, Any]]:
        result = self.execute("-json", *args)
        text = result.stdout.strip()
        if not text:
            return []
        data = json.loads(text)
        if not isinstance(data, list):
            raise ExifToolError("unexpected JSON from exiftool")
        return data

    def version(self) -> str:
        return self.execute("-ver").stdout.strip()

    def image_data_hash(self, path: Path) -> str:
        result = self.execute(
            "-api",
            "requesttags=imagedatahash",
            "-api",
            "imagehashtype=SHA256",
            "-s3",
            "-ImageDataHash",
            str(path),
        )
        value = result.stdout.strip()
        if not _HEX64.match(value):
            raise ExifToolError("exiftool couldn't compute the image-data hash", result.stderr)
        return value

    def read_tags(self, path: Path, *tags: str, binary: bool = False) -> dict[str, Any]:
        """``-json -G1 -a`` for the given tag selectors, without ``SourceFile``.

        Convenient, but not for integrity checks: with ``-G1`` exiftool's JSON
        keeps only the first of two same-named tags in one group, and numbers
        are print-converted. ``asif.meta.gates`` has its own lossless dump.
        """
        args = ["-G1", "-a"]
        if binary:
            args.append("-b")
        rows = self.execute_json(*args, *tags, str(path))
        if not rows:
            return {}
        row = dict(rows[0])
        row.pop("SourceFile", None)
        return row
