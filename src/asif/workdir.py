"""The RAM work dir: the only place a plaintext photo ever lives.

On Linux this is a private directory on a tmpfs (``$XDG_RUNTIME_DIR`` first,
it's per-user; then ``/dev/shm``). Files there never touch a disk, so they
can't end up in snapshots or backups, and they vanish on power loss. If no
tmpfs is available (macOS before the RAM-disk support in M10), As If refuses
to work unless you explicitly allow an on-disk work dir.

Each run holds a lock on its own work dir. A run that was killed outright
(SIGKILL, power loss on an on-disk dir) can't wipe its leftovers itself, so
the next run wipes every work dir whose lock nobody holds any more, in every
place a run of this user could have put one: the usual RAM folders, plus
every folder a run has used before (remembered in the state dir, so a
changed ``work_dir`` setting doesn't strand a killed run's leftovers).
"""

from __future__ import annotations

import fcntl
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from asif import paths

_RAM_FSTYPES = frozenset({"tmpfs", "ramfs"})
_DIR_NAME = re.compile(r"asif-[0-9a-f]{12}")
_LOCK = ".lock"
# A work dir without a lock file is either being created right now or was
# left by a run that died between mkdir and taking the lock.
_UNLOCKED_GRACE_S = 600


class NotInRamError(RuntimeError):
    """The work dir would be on disk and that wasn't explicitly allowed."""


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
            # ">=": when several filesystems are mounted on the same path, the
            # one listed last is on top and is the one files actually land on.
            if best is None or depth >= best[0]:
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


def _wipe_tree(path: Path) -> None:
    for root, _dirs, files in os.walk(path):
        for name in files:
            wipe_file(Path(root) / name)
    shutil.rmtree(path, ignore_errors=True)


def sweep_stale(parent: Path) -> int:
    """Wipe this user's work dirs left by runs that are gone. Returns how many.

    A live run holds an exclusive lock on its dir's lock file, so taking the
    lock proves nobody is using the dir; another ``asif`` running at the
    same time is never touched.
    """
    try:
        entries = list(parent.iterdir())
    except OSError:
        return 0
    removed = 0
    for entry in entries:
        if not _DIR_NAME.fullmatch(entry.name):
            continue
        try:
            st = entry.lstat()
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            continue
        try:
            fd: int | None = os.open(entry / _LOCK, os.O_RDWR | os.O_NOFOLLOW)
        except FileNotFoundError:
            if time.time() - st.st_mtime < _UNLOCKED_GRACE_S:
                continue
            fd = None
        except OSError:
            continue
        try:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue  # a live run
            _wipe_tree(entry)
            removed += 1
        finally:
            if fd is not None:
                os.close(fd)
    return removed


def _registry() -> Path:
    return paths.state_dir() / "workdir-parents"


def _remembered_parents() -> list[Path]:
    try:
        lines = _registry().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [Path(line) for line in lines if line.startswith("/")]


def _remember_parent(parent: Path) -> None:
    """Record ``parent`` before a work dir is made in it, so a later run can sweep it.

    Raises OSError if the state dir can't be written.
    """
    entry = str(parent.resolve())
    if "\n" in entry or Path(entry) in _remembered_parents():
        return
    registry = _registry()
    paths.ensure_private_dir(registry.parent)
    # Append-only, never rewritten: concurrent runs can't drop each other's
    # entries (a duplicate is harmless), and a folder that is missing today
    # (an unmounted drive) is still swept once it's back.
    fd = os.open(registry, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(entry + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# Filesystems that can stop answering (a server gone, a stalled FUSE daemon),
# hanging anything that touches them in a way Ctrl+C can't interrupt.
_REMOTE_FSTYPES = (
    "nfs",
    "cifs",
    "smb",
    "fuse",
    "9p",
    "ceph",
    "gluster",
    "afs",
    "davfs",
    "autofs",
    "lustre",
    "afpfs",  # macOS, from here on
    "webdav",
    "macfuse",
    "osxfuse",
)


def _unescape_mount(field: str) -> str:
    """A path from /proc/mounts, where spaces and the like are octal escapes."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def _mount_table(mounts_file: Path) -> list[tuple[str, str, list[str]]] | None:
    """Every mount as (mount point, type, options), in mount order; None if unknown.

    Linux has /proc/mounts. macOS has none, so its ``mount`` command is
    asked: it lists what the kernel already knows, without touching a
    share that might not answer.
    """
    try:
        lines = mounts_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        if sys.platform != "darwin":
            return None
        try:
            out = subprocess.run(
                ["/sbin/mount"], capture_output=True, text=True, timeout=10, check=True
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_macos_mounts(out)
    table = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            table.append((_unescape_mount(parts[1]), parts[2], parts[3].split(",")))
    return table


def parse_macos_mounts(out: str) -> list[tuple[str, str, list[str]]]:
    """``mount`` output on macOS: "//me@nas/photos on /Volumes/photos (smbfs, nodev, ...)"."""
    table = []
    for line in out.splitlines():
        head, sep, tail = line.rpartition(" (")
        _, on, mount_point = head.partition(" on ")
        if not sep or not on or not tail.endswith(")"):
            continue
        fstype, *options = [part.strip() for part in tail[:-1].split(",")]
        table.append((mount_point, fstype, options))
    return table


def mount_of(
    entry: str, mounts_file: Path = Path("/proc/mounts")
) -> tuple[str, str, list[str]] | None:
    """The mount point, filesystem type and options of the mount holding ``entry``.

    ``entry`` must be an already-resolved path. Decided from the mount table
    alone, without touching the path, which could hang if the mount has
    stopped answering. None if the table can't be read.
    """
    table = _mount_table(mounts_file)
    if table is None:
        return None
    best: tuple[int, str, str, list[str]] | None = None
    for mount_point, fstype, options in table:
        prefix = mount_point.rstrip("/") + "/"
        if entry == mount_point or entry.startswith(prefix):
            depth = len(Path(mount_point).parts)
            # ">=": the mount listed last on the same path is the one on top.
            if best is None or depth >= best[0]:
                best = (depth, mount_point, fstype, options)
    return None if best is None else best[1:]


def _is_remote(entry: str, mounts_file: Path = Path("/proc/mounts")) -> bool:
    """Is the already-resolved path ``entry`` on a network or FUSE mount?"""
    mount = mount_of(entry, mounts_file)
    return mount is not None and mount[1].startswith(_REMOTE_FSTYPES)


def _sweep_parents(override: str) -> list[Path]:
    """Every folder a run of this user could have put its work dir in.

    Folders remembered from earlier runs are skipped while they are on a
    network or FUSE mount (a dead one would hang every run); a run that uses
    such a folder again sweeps it then.
    """
    candidates = [p for p in _remembered_parents() if not _is_remote(str(p))]
    if override:
        candidates.append(Path(override).expanduser())
    if runtime := os.environ.get("XDG_RUNTIME_DIR"):
        candidates.append(Path(runtime))
    candidates += [
        Path(f"/run/user/{os.getuid()}"),
        Path("/dev/shm"),  # noqa: S108  # only our own asif-<hex> dirs are touched
        Path(tempfile.gettempdir()),
    ]
    seen: list[Path] = []
    for cand in candidates:
        try:
            real = cand.resolve()
        except OSError:
            continue
        if real.is_dir() and real not in seen:
            seen.append(real)
    return seen


class RamWorkDir:
    """A private (0700) directory in RAM, locked while in use and wiped on exit."""

    def __init__(self, override: str = "", *, allow_disk: bool = False) -> None:
        self.choice = find_ram_dir(override)
        self.allow_disk = allow_disk
        self.path: Path | None = None
        self.swept = 0
        """How many dead runs' leftovers were wiped on entry."""
        self.unremembered: str | None = None
        """Why the work dir's folder couldn't be noted for later sweeps, if it couldn't."""
        self._lock_fd: int | None = None

    def __enter__(self) -> RamWorkDir:
        if not self.choice.in_ram and not self.allow_disk:
            raise NotInRamError(
                f"no RAM-backed folder for the work dir ({self.choice.reason}), so photos "
                "would be written to disk while they're processed"
            )
        parent = self.choice.path
        parent.mkdir(parents=True, exist_ok=True)
        # Remembered before the work dir exists: a run killed right after
        # creating it must still be findable. Best effort: a state dir that
        # can't be written (read-only, someone else's) mustn't stop the run.
        try:
            _remember_parent(parent)
        except OSError as exc:
            self.unremembered = exc.strerror or str(exc)
        self.swept = sum(sweep_stale(p) for p in _sweep_parents(str(parent)))
        path = parent / f"asif-{secrets.token_hex(6)}"
        path.mkdir(mode=0o700)
        # A ".lock" file must never exist unlocked, or a sweep by another run
        # could take it in between and wipe this live dir. So lock it under a
        # temporary name and only then give it its real name.
        tmp = path / f".lock-{secrets.token_hex(4)}"
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.rename(tmp, path / _LOCK)
        except BaseException:
            os.close(fd)
            _wipe_tree(path)
            raise
        self._lock_fd = fd
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
        try:
            _wipe_tree(self.path)
        finally:
            self.path = None
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
