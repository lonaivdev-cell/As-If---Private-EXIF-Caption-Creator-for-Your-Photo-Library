"""The RAM work dir: the only place a plaintext photo ever lives.

On Linux this is a private directory on a tmpfs (``$XDG_RUNTIME_DIR`` first,
it's per-user; then ``/dev/shm``). Files there never touch a disk, so they
can't end up in snapshots or backups, and they vanish on power loss. If no
tmpfs is available (macOS before the RAM-disk support in M10), As If falls
back to an on-disk temp dir and says so.
"""

from __future__ import annotations

import os
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

_RAM_FSTYPES = frozenset({"tmpfs", "ramfs"})


@dataclass(frozen=True)
class RamDirChoice:
    path: Path
    in_ram: bool
    reason: str


def _mount_fstype(path: Path, mounts_file: Path = Path("/proc/mounts")) -> str | None:
    """Filesystem type of the mount that contains ``path`` (Linux only)."""
    try:
        lines = mounts_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    try:
        target = path.resolve()
    except OSError:
        return None
    best: tuple[int, str] | None = None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = Path(parts[1].replace("\\040", " "))
        if target == mount_point or mount_point in target.parents:
            depth = len(mount_point.parts)
            if best is None or depth > best[0]:
                best = (depth, parts[2])
    return best[1] if best else None


def find_ram_dir(override: str = "", mounts_file: Path = Path("/proc/mounts")) -> RamDirChoice:
    """Pick the parent directory for the RAM work dir."""
    if override:
        path = Path(override).expanduser()
        in_ram = _mount_fstype(path, mounts_file) in _RAM_FSTYPES
        why = "configured work_dir" + ("" if in_ram else " (not a tmpfs: files touch disk)")
        return RamDirChoice(path, in_ram, why)
    if sys.platform.startswith("linux"):
        candidates = []
        if runtime := os.environ.get("XDG_RUNTIME_DIR"):
            candidates.append(Path(runtime))
        # Shared, but we only ever create a 0700 directory with a random name in it.
        candidates.append(Path("/dev/shm"))  # noqa: S108
        for cand in candidates:
            if (
                cand.is_dir()
                and os.access(cand, os.W_OK | os.X_OK)
                and _mount_fstype(cand, mounts_file) in _RAM_FSTYPES
            ):
                return RamDirChoice(cand, True, f"tmpfs at {cand}")
    fallback = Path(tempfile.gettempdir())
    in_ram = _mount_fstype(fallback, mounts_file) in _RAM_FSTYPES
    reason = "temp dir is a tmpfs" if in_ram else "no tmpfs found: work files touch disk"
    return RamDirChoice(fallback, in_ram, reason)


def wipe_file(path: Path) -> None:
    """Overwrite a file with zeros, then unlink it. Missing files are fine.

    On tmpfs the overwrite really replaces the pages; on a disk it is best
    effort (SSDs and copy-on-write filesystems may keep old blocks), which is
    exactly why plaintext belongs in RAM.
    """
    try:
        size = path.stat().st_size
        with path.open("r+b", buffering=0) as fh:
            chunk = b"\0" * min(size, 1 << 20) if size else b""
            remaining = size
            while remaining > 0:
                n = min(remaining, len(chunk))
                fh.write(chunk[:n])
                remaining -= n
            os.fsync(fh.fileno())
    except FileNotFoundError:
        return
    except OSError:
        pass
    path.unlink(missing_ok=True)


class RamWorkDir:
    """A private (0700) directory in RAM, wiped on exit."""

    def __init__(self, override: str = "") -> None:
        self.choice = find_ram_dir(override)
        self.path: Path | None = None

    def __enter__(self) -> RamWorkDir:
        parent = self.choice.path
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"asif-{secrets.token_hex(6)}"
        path.mkdir(mode=0o700)
        self.path = path
        return self

    def new_path(self, suffix: str) -> Path:
        """A fresh, unpredictable file name inside the work dir."""
        assert self.path is not None, "use RamWorkDir as a context manager"
        return self.path / f"{secrets.token_hex(8)}{suffix.lower()}"

    def free_bytes(self) -> int:
        assert self.path is not None
        return shutil.disk_usage(self.path).free

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.path is None:
            return
        for child in self.path.iterdir():
            if child.is_file():
                wipe_file(child)
        shutil.rmtree(self.path, ignore_errors=True)
        self.path = None
