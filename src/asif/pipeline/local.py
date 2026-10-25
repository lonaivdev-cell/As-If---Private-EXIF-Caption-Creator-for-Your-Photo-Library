"""M1: caption a local folder into a separate folder of copies.

Originals are only ever read. Each photo is copied into the RAM work dir,
described by the model, tagged there by exiftool, checked by every gate, and
only then written to the destination folder, under the same relative path.
A destination file that already exists is never overwritten.

Per photo (states from §5 of docs/DESIGN.md)::

    copy into RAM (hashing as we go)  -> OPENED
    snapshot: ImageDataHash, tags, bytes
    model -> JSON -> schema            -> DESCRIBED   (retry <= 2, then QUARANTINE)
    additive-only write on the copy    -> WRITTEN
    gates (equality, protected fields,
      trailer, structure, size)        -> VERIFIED    (else FAILED_GATE)
    original unchanged since copy?                    (else SKIPPED_CONFLICT)
    no-clobber write to destination    -> UPLOADED
    re-read destination, compare hash  -> CONFIRMED
    wipe the RAM copy                  -> CLEAN

What goes wrong with one photo (unreadable, vanished, mislabelled, refused by
a gate) ends up in the summary and the run goes on. What would go wrong with
every photo (the destination is full or read-only, the model worker can't be
restarted, too many gate failures, a stop signal) ends the run early, and
the summary of what was done is still returned.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import grp
import hashlib
import os
import pwd
import secrets
import stat
import struct
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from asif import __version__, proc
from asif.config import Config
from asif.infer.client import InferenceClient, InferenceError, WorkerConfig
from asif.infer.schema import PROMPT_VERSION, CaptionResult, SchemaError, parse_caption_json
from asif.meta.exiftool import ExifTool, ExifToolError
from asif.meta.gates import (
    Fingerprint,
    all_ok,
    check_source_unchanged,
    check_written,
    fingerprint,
    snapshot,
)
from asif.meta.writer import Provenance, WritePlan, apply_write, plan_write, read_existing
from asif.models.manifest import load_manifest
from asif.pipeline.states import State
from asif.pipeline.timings import TimingsLog
from asif.workdir import NotInRamError, RamWorkDir, mount_of, wipe_file

WRITABLE = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
_SUFFIX_FORMAT = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".tif": "tiff",
    ".tiff": "tiff",
}
NOT_YET = {
    **dict.fromkeys(
        (".heic", ".heif", ".avif"),
        "HEIC/AVIF: not written until the fixture pass proves it safe on your exiftool",
    ),
    **dict.fromkeys(
        (
            ".dng",
            ".cr2",
            ".cr3",
            ".nef",
            ".nrw",
            ".arw",
            ".srf",
            ".sr2",
            ".raf",
            ".orf",
            ".rw2",
            ".pef",
            ".srw",
            ".x3f",
            ".3fr",
            ".erf",
            ".kdc",
            ".mos",
            ".iiq",
        ),
        "RAW: gets a sidecar .xmp in a later milestone, never written into",
    ),
    **dict.fromkeys(
        (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp", ".webm", ".mts", ".m2ts"),
        "video is out of scope for v1",
    ),
}

SCHEMA_RETRIES = 2
FAILURE_WINDOW = 200
MAX_CONSECUTIVE_WORKER_FAILURES = 3
_COPY_CHUNK = 1 << 20
# Destination errors that concern one photo's path, not the whole destination.
_PER_PATH_ERRNOS = frozenset(
    {errno.EEXIST, errno.ENOTDIR, errno.EISDIR, errno.ENAMETOOLONG, errno.EINVAL, errno.EILSEQ}
)

Backend = Literal["llama", "stub"]


class CaptionError(RuntimeError):
    """The run can't start (bad folders, missing model, missing tools)."""


class RunAborted(RuntimeError):
    """The run stopped early because something looks systematically wrong."""


class _DestinationError(RuntimeError):
    """Writing to the destination failed in a way that would hit every photo."""


@dataclass
class ItemOutcome:
    rel: str
    """Path relative to the source folder. Shown in the terminal, never logged."""
    state: State
    reason: str = ""
    result: CaptionResult | None = None
    """The model's answer, kept only for --dry-run previews."""
    plan: WritePlan | None = None
    """What would be written, kept only for --dry-run previews."""
    note: str = ""
    """Something worth knowing about a photo that was captioned anyway."""
    systematic: bool = False
    """Counts against the failure budget: a sign of a bug, not of a bad input file."""


@dataclass
class RunSummary:
    run_id: str
    dry_run: bool
    outcomes: list[ItemOutcome] = field(default_factory=list)
    ignored: int = 0
    """Files that aren't photos at all (or are hidden, links or special files)."""
    unreadable_dirs: list[str] = field(default_factory=list)
    """Folders under the source that couldn't be listed, relative to it."""
    aborted: str | None = None
    interrupted: bool = False
    """Stopped by Ctrl+C or a stop signal."""

    def count(self, *states: State) -> int:
        return sum(1 for o in self.outcomes if o.state in states)

    @property
    def failed(self) -> list[ItemOutcome]:
        bad = {State.FAILED_GATE, State.FAILED_UPLOAD, State.QUARANTINE}
        return [o for o in self.outcomes if o.state in bad]


@dataclass(frozen=True)
class ModelChoice:
    backend: Backend
    model_id: str
    model_sha256: str
    worker: WorkerConfig


ProgressFn = Callable[[ItemOutcome, int, int], None]
NoticeFn = Callable[[str], None]


# -- planning --------------------------------------------------------------


def check_folders(src: Path, dest: Path) -> tuple[Path, Path]:
    """Resolve and sanity-check the source and destination folders."""
    src = src.expanduser().resolve()
    dest = dest.expanduser().resolve()
    if not src.is_dir():
        raise CaptionError(f"source folder not found: {src}")
    if dest == src:
        raise CaptionError(
            "the destination is the source folder. M1 only writes copies into a separate "
            "folder; in-place writing comes with the cloud milestones."
        )
    if src in dest.parents:
        raise CaptionError("the destination can't be inside the source folder")
    if dest in src.parents:
        raise CaptionError("the source can't be inside the destination folder")
    if dest.exists() and not dest.is_dir():
        raise CaptionError(f"the destination exists and isn't a folder: {dest}")
    return src, dest


@dataclass(frozen=True)
class Discovery:
    files: list[Path]
    ignored: int
    unreadable_dirs: list[str]


def discover(src: Path) -> Discovery:
    """Photo-ish files under ``src``, sorted.

    Hidden files and folders are skipped, and symlinks are never followed, so
    a link can't pull in files from outside the folder you chose. Only
    regular files count: a FIFO named ``x.jpg`` would block forever on read.
    Folders that can't be listed are reported, not silently skipped.
    """
    found: list[Path] = []
    ignored = 0
    unreadable: list[str] = []

    def on_error(exc: OSError) -> None:
        where = Path(exc.filename) if exc.filename else src
        unreadable.append(where.relative_to(src).as_posix() if where != src else ".")

    for root, dirs, files in os.walk(src, followlinks=False, onerror=on_error):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            path = Path(root) / name
            suffix = path.suffix.lower()
            try:
                st = path.lstat()
            except OSError:
                ignored += 1
                continue
            if name.startswith(".") or not stat.S_ISREG(st.st_mode):
                ignored += 1
            elif suffix in WRITABLE or suffix in NOT_YET:
                found.append(path)
            else:
                ignored += 1
    return Discovery(found, ignored, sorted(unreadable))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def choose_model(cfg: Config, *, notice: NoticeFn | None = None) -> ModelChoice:
    """Work out what will produce the captions, or explain what's missing."""
    m = cfg.model
    lang = cfg.general.caption_lang
    max_edge = cfg.resources.max_edge_px
    catalogue = {entry.id: entry for entry in load_manifest()}
    if m.id not in ("stub", "auto", "local") and m.id not in catalogue:
        raise CaptionError(
            f"unknown model id {m.id!r}. Use stub, local, or an id from `asif models list`."
        )
    if m.id == "stub":
        worker = WorkerConfig(
            backend="stub",
            lang=lang,
            max_edge_px=max_edge,
            load_timeout_s=m.load_timeout_s,
            request_timeout_s=m.request_timeout_s,
        )
        return ModelChoice("stub", "stub", "", worker)
    if not (m.gguf and m.mmproj):
        raise CaptionError(
            "no model configured. Download a vision model's .gguf and its mmproj .gguf "
            "(see the README), then pass --gguf and --mmproj (or set them under [model] "
            "in your config). To try the pipeline without a model, use --model stub."
        )
    gguf = Path(m.gguf).expanduser()
    mmproj = Path(m.mmproj).expanduser()
    for label, path in (("model (--gguf)", gguf), ("projector (--mmproj)", mmproj)):
        if not path.is_file():
            raise CaptionError(f"{label} not found: {path}")
    if notice:
        notice("Hashing the model file for the provenance record (once per run)...")
    model_sha = sha256_file(gguf)

    # A catalogue id is only recorded when the file is provably that model.
    entry = catalogue.get(m.id)
    weights = {f.sha256 for f in entry.files if f.role == "weights"} if entry else set()
    if entry and entry.pinned and model_sha in weights:
        model_id = entry.id
    else:
        model_id = f"local:{gguf.name}"
        if entry and notice:
            notice(
                f"{entry.id} has no pinned checksum to verify this file against yet, so it's "
                f"recorded as {model_id}."
            )

    # Placement (§9.3): gpu = every layer on the GPU; hybrid = the same, but
    # MoE experts stay in RAM; cpu = nothing on the GPU; auto = llama.cpp decides.
    gpu_layers = m.gpu_layers
    if gpu_layers is None:
        gpu_layers = {"cpu": 0, "gpu": 999, "hybrid": 999}.get(m.device)
    placement = ("--cpu-moe",) if m.device == "hybrid" else ()
    worker = WorkerConfig(
        backend="llama",
        lang=lang,
        max_edge_px=max_edge,
        load_timeout_s=m.load_timeout_s,
        request_timeout_s=m.request_timeout_s,
        llama_server=m.llama_server,
        model=str(gguf),
        mmproj=str(mmproj),
        ctx=m.ctx,
        threads=m.threads,
        gpu_layers=gpu_layers,
        server_args=(*placement, *m.server_args),
    )
    return ModelChoice("llama", model_id, model_sha, worker)


# -- the run ---------------------------------------------------------------


class _NotAFile(OSError):
    """The source is a FIFO, device or directory, not a photo file."""


def _copy_and_fingerprint(src: Path, dest: Path) -> Fingerprint:
    """Copy ``src`` to ``dest``, hashing exactly the bytes copied.

    The source is opened first (no-follow, non-blocking, and checked to be a
    regular file) so a failure leaves nothing behind in the work dir.
    """
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    src_fd = os.open(src, flags)
    with os.fdopen(src_fd, "rb") as fin:
        st = os.fstat(fin.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise _NotAFile(errno.EINVAL, "not a regular file")
        digest = hashlib.sha256()
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fout:
            while chunk := fin.read(_COPY_CHUNK):
                digest.update(chunk)
                fout.write(chunk)
    os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns))
    return Fingerprint(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest.hexdigest())


def _has_acl(path: Path | int) -> bool:
    """Does ``path`` (or an open fd) carry a POSIX access ACL? (Linux; else False.)"""
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:  # macOS: no POSIX ACLs, group bits mean what they say
        return False
    try:
        getxattr(path, "system.posix_acl_access")
    except OSError as exc:
        # No ACL, or a filesystem without them. Anything else: assume one.
        return exc.errno not in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP)
    return True


def _reach_mask(path: Path) -> int:
    """Permission bits ``path`` and its folders actually let group/others use.

    A 0644 photo inside a 0700 folder is private; its copy must not become
    readable just because the destination folder is open. The file's group
    gets through a folder by that folder's group bits only if it is the
    folder's group too; otherwise its members are "others" there. With a
    POSIX ACL the group bits are the ACL's mask, not the owning group's
    access, and the copy gets no ACL; so wherever an ACL is involved, group
    access is dropped. This can only narrow access, never widen it.
    """
    mask = 0o777
    try:
        gid = path.resolve().stat().st_gid
    except OSError:
        return 0o700
    if _private_mount(mount_of(str(path.resolve()))):
        return 0o700  # only we can enter that mount
    if _has_acl(path):
        mask &= ~0o070
    for parent in path.parents:
        try:
            pst = parent.stat()
        except OSError:
            return 0o700
        mode = pst.st_mode
        if _has_acl(parent):
            mask &= ~0o070
        if not mode & stat.S_IXOTH:
            mask &= ~0o007
        if not mode & (stat.S_IXGRP if pst.st_gid == gid else stat.S_IXOTH):
            mask &= ~0o070
    return mask


# Who, besides a folder's owner (and root), can list or search it: nobody,
# the members of one group (its gid), or anyone. Coarse on purpose: As If
# only needs to know that a destination folder's audience, over-estimated,
# fits inside its original's, under-estimated.
_ANYONE = -1
_NOBODY = -2


def _both(a: int, b: int, *, wide: bool) -> int:
    """The audience in both ``a`` and ``b``; when unsure, more if ``wide``, else fewer."""
    if a in (_ANYONE, b):
        return b
    if b == _ANYONE:
        return a
    if _NOBODY in (a, b):
        return _NOBODY
    return a if wide else _NOBODY  # two groups: their common members, if any


@functools.cache
def _nobody_elses(gid: int) -> bool:
    """Is ``gid`` a group with no one else in it: root's, or our own private
    one (named after us, as most Linux systems make one per user, and no
    one else's main group)?"""
    if gid == 0:
        return True
    if gid != os.getgid():
        return False
    try:
        me = pwd.getpwuid(os.getuid()).pw_name
        group = grp.getgrgid(gid)
    except KeyError:
        return False
    if group.gr_name != me or not set(group.gr_mem) <= {me}:
        return False
    return not any(user.pw_gid == gid and user.pw_name != me for user in pwd.getpwall())


def _group(gid: int) -> int:
    return _NOBODY if _nobody_elses(gid) else gid


def _someone_else(uid: int) -> bool:
    return uid not in (os.getuid(), 0)


Mount = tuple[str, str, list[str]] | None  # mount point, type, options


def _private_mount(mount: Mount) -> bool:
    """Is it a FUSE mount only the user who mounted it can enter?

    sshfs, rclone and the like mount that way unless told ``allow_other``,
    and since As If can get in, that user is us. (Only on this computer:
    the share's other users still see what the share lets them.)
    """
    return mount is not None and mount[1].startswith("fuse") and "allow_other" not in mount[2]


def _modes_unenforced(mount: Mount) -> bool:
    """Is it a FUSE mount every local user can enter, whatever the modes say?

    With ``allow_other`` but not ``default_permissions`` the kernel doesn't
    check permissions at all: everyone gets in as the mount's own account.
    """
    return (
        mount is not None
        and mount[1].startswith("fuse")
        and "allow_other" in mount[2]
        and "default_permissions" not in mount[2]
    )


# Filesystems that other computers' users get to as well: network shares,
# and FUSE ones, which mostly are network or cloud storage. For a folder on
# one, the folders above its mount point keep none of them out.
_SHARED_FSTYPES = ("nfs", "cifs", "smb", "ceph", "gluster", "afs", "davfs", "lustre", "9p", "fuse")
_LOCAL_FUSE = (
    "fuseblk",
    "fuse.gocryptfs",
    "fuse.encfs",
    "fuse.cryfs",
    "fuse.securefs",
    "fuse.mergerfs",
)


def _shared_mount(mount: Mount) -> bool:
    return (
        mount is not None
        and mount[1].startswith(_SHARED_FSTYPES)
        and not mount[1].startswith(_LOCAL_FUSE)
    )


def _chmod_ignored(fd: int) -> bool:
    """Does the filesystem ignore permissions As If sets (rclone and other cloud mounts)?

    Two owner-only modes are tried: no fixed mode can show both.
    """
    for mode in (0o700, 0o600):
        try:
            os.fchmod(fd, mode)
        except PermissionError:
            return True
        if stat.S_IMODE(os.fstat(fd).st_mode) != mode:
            return True
    return False


def _within(path: Path, mount_point: str) -> bool:
    entry = str(path)
    return entry == mount_point or entry.startswith(mount_point.rstrip("/") + "/")


# POSIX ACL entry tags, as in system.posix_acl_access.
_ACL_USER, _ACL_GROUP_OBJ, _ACL_GROUP, _ACL_MASK, _ACL_OTHER = 0x02, 0x04, 0x08, 0x10, 0x20


def _acl_entries(path: Path) -> list[tuple[int, int, int]] | None:
    """``path``'s POSIX access ACL as (tag, permissions, id); None without one."""
    if not _has_acl(path):
        return None
    try:
        raw = os.getxattr(path, "system.posix_acl_access")
        return [struct.unpack_from("<HHI", raw, at) for at in range(4, len(raw) - 7, 8)]
    except (OSError, struct.error):
        return []  # there, but unreadable


def _acl_audience(entries: list[tuple[int, int, int]], gid: int, perm: int, *, wide: bool) -> int:
    """Who a POSIX ACL gives ``perm`` (4 read, 1 search): over-estimated when
    ``wide``, under-estimated otherwise."""
    if not entries:
        return _ANYONE if wide else _NOBODY
    mask = next((p for tag, p, _ in entries if tag == _ACL_MASK), 7)

    def grants(tag: int, granted: int) -> bool:
        return bool(granted & perm & (7 if tag == _ACL_OTHER else mask))

    if not wide:
        # A named user's entry decides for that user alone, even against
        # what the group or others get; anyone in a group gets in if any of
        # their groups' entries says so, and only then.
        if any(t == _ACL_USER and _someone_else(i) and not grants(t, g) for t, g, i in entries):
            return _NOBODY
        allowed = [grants(t, g) for t, g, _ in entries if t in (_ACL_GROUP_OBJ, _ACL_GROUP)]
        if all(allowed) and any(grants(t, g) for t, g, _ in entries if t == _ACL_OTHER):
            return _ANYONE
        if any(grants(t, g) for t, g, _ in entries if t == _ACL_GROUP_OBJ):
            return _group(gid)
        return _NOBODY
    groups: set[int] = set()
    for tag, granted, ident in entries:
        if not grants(tag, granted):
            continue
        if tag == _ACL_OTHER or (tag == _ACL_USER and _someone_else(ident)):
            return _ANYONE  # everyone, or another person by name
        if tag in (_ACL_GROUP_OBJ, _ACL_GROUP):
            groups.add(_group(gid if tag == _ACL_GROUP_OBJ else ident))
    groups.discard(_NOBODY)
    return _NOBODY if not groups else groups.pop() if len(groups) == 1 else _ANYONE


# NFSv4 ACLs, as the Linux client shows them (system.nfs4_acl, XDR).
_NFS4_ALLOW, _NFS4_DENY = 0, 1
_NFS4_READ = 0x1  # read data; list, for a folder
_NFS4_EXECUTE = 0x20  # search, for a folder
_NFS4_SPECIAL_WHO = frozenset({b"OWNER@", b"GROUP@", b"EVERYONE@"})


def _nfs4_aces(target: Path | int) -> list[tuple[int, int, bytes]] | None:
    """The NFSv4 ACL of ``target`` (a path or an open fd) as (type, access,
    who); None where there is none (not NFSv4).

    An NFS server can give files an ACL (inherited from their folder, say)
    that the Linux client shows only as ``system.nfs4_acl``, never as a
    POSIX ACL. One that can't be read comes back as granting everything to
    someone unknown, so it's never taken as harmless.
    """
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:
        return None
    try:
        raw = getxattr(target, "system.nfs4_acl")
    except OSError as exc:
        if exc.errno in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
            return None
        raise
    unknown = [(_NFS4_ALLOW, 0xFFFFFFFF, b"?")]
    aces = []
    try:
        (count,) = struct.unpack_from(">I", raw, 0)
        pos = 4
        for _ in range(count):
            kind, _flags, access, length = struct.unpack_from(">IIII", raw, pos)
            who = raw[pos + 16 : pos + 16 + length]
            pos += 16 + (length + 3) // 4 * 4
            if len(who) != length:
                return unknown
            aces.append((kind, access, who))
    except struct.error:
        return unknown
    return aces


def _nfs4_named(aces: list[tuple[int, int, bytes]], access: int, kind: int = _NFS4_ALLOW) -> bool:
    """Does an entry for someone by name (not owner, group, everyone) allow (or deny) ``access``?"""
    return any(k == kind and a & access and w not in _NFS4_SPECIAL_WHO for k, a, w in aces)


def _nfs4_named_readers(fd: int) -> bool:
    """Does an NFSv4 ACL on ``fd`` let someone read it by name?

    Entries for the owner, group and everyone follow the mode As If sets;
    any other that grants reading means someone else can.
    """
    aces = _nfs4_aces(fd)
    return aces is not None and _nfs4_named(aces, _NFS4_READ)


def _audience(path: Path, bits: int, *, wide: bool) -> int:
    """Who gets the ``bits`` access (read 0o444 or search 0o111) to ``path``."""
    st = path.stat()
    mode = st.st_mode
    aces = _nfs4_aces(path)
    if aces is not None:
        access = _NFS4_READ if bits == 0o444 else _NFS4_EXECUTE
        if wide and _nfs4_named(aces, access):
            return _ANYONE  # someone by name; the mode doesn't show them
        if not wide and _nfs4_named(aces, access, _NFS4_DENY):
            return _NOBODY  # someone by name is kept out, whatever the mode says
    entries = _acl_entries(path)
    if entries is not None:
        return _acl_audience(entries, st.st_gid, bits & 0o7, wide=wide)
    other, group = mode & bits & 0o007, mode & bits & 0o070
    if other and (group or wide):
        return _ANYONE
    if group:
        return _group(st.st_gid)
    return _NOBODY  # (others but not the group: rare, so counted as nobody)


def _folder_audience(folder: Path, *, wide: bool, bits: int = 0o444) -> int:
    """Who besides the owner can list ``folder`` (or, with ``bits`` 0o111, get
    to what's in it): over-estimated when ``wide`` (a destination folder),
    under-estimated otherwise (an original's folder).

    Symlinks are followed: what counts is the folder the names end up in.
    """
    real = folder.resolve()
    mount = mount_of(str(real))
    if not wide and _private_mount(mount):
        return _NOBODY  # no one else on this computer gets in
    if wide and _modes_unenforced(mount):
        return _ANYONE
    if wide and _shared_mount(mount):
        # What a share shows as owners, groups and modes may not be what its
        # other users get (sshfs or cifs uid=, gid=, umask=; all_squash), and
        # they never pass through this computer's folders: As If can't tell.
        return _ANYONE
    st = real.stat()
    audience = _audience(real, bits, wide=wide)
    if wide and _someone_else(st.st_uid):
        audience = _ANYONE  # someone else's folder: they can list it too
    for parent in real.parents:
        search = _audience(parent, 0o111, wide=wide)
        if wide and _someone_else(parent.stat().st_uid):
            search = _ANYONE  # its owner gets through, whatever the bits say
        audience = _both(audience, search, wide=wide)
    return audience


def _more_people_can_list(dest_folder: Path, src_folder: Path) -> bool:
    """Can people list ``dest_folder`` who can't list ``src_folder``?

    Names alone can say a lot, so a copy never goes where its name would be
    visible to people who can't see the original's. Errors reading the
    original's folder raise ``_SourceUnreadable``.
    """
    theirs = _folder_audience(dest_folder, wide=True)
    if theirs == _NOBODY:
        return False
    try:
        ours = _folder_audience(src_folder, wide=False)
    except OSError as exc:
        raise _SourceUnreadable(exc) from exc
    return ours not in (_ANYONE, theirs)


class _SourceUnreadable(Exception):
    """The original's folder couldn't be read while checking the destination."""

    def __init__(self, exc: OSError) -> None:
        super().__init__(str(exc))
        self.exc = exc


class _InheritedAcl(Exception):
    """The destination gives new files an ACL that As If can't take away."""


class _ModeNotKept(Exception):
    """The destination gave a copy more read access than As If set (a FAT or
    exFAT drive, or a network share, whose permissions are fixed when mounted)."""

    def __init__(self, forced: int) -> None:
        super().__init__(oct(forced))
        self.forced = forced


_MODE_NOT_KEPT = (
    "the destination would let other users read the copy (it gives every file the "
    "same permissions), but not the original"
)


class _FolderRefused(Exception):
    """The destination can't hold a folder by this name (or a file inside it)."""

    def __init__(self, folder: Path, exc: OSError) -> None:
        super().__init__(str(exc))
        self.folder, self.exc = folder, exc


def _kept_private(fd: int, mode: int, folder: Path) -> None:
    """Check a new file in ``folder`` really has no more access than ``mode``.

    An ACL that couldn't be removed gets a mask that grants nothing (on an
    NFS server with POSIX ACLs, that is what its NFSv4 ACL follows too).
    Raises ``_InheritedAcl`` or ``_ModeNotKept`` if that's still not so.
    """
    if _has_acl(fd) or _nfs4_named_readers(fd):
        os.fchmod(fd, mode & 0o707)
        if _nfs4_named_readers(fd):
            raise _InheritedAcl
    forced = stat.S_IMODE(os.fstat(fd).st_mode) & 0o044 & ~mode
    if forced and _folder_audience(folder, wide=True, bits=0o111) != _NOBODY:
        raise _ModeNotKept(forced)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _make_private_dirs(path: Path, root: Path | None = None) -> list[Path]:
    """Create ``path`` and any missing parents as 0700, one level at a time.

    Returns the folders it created, top first. A name the destination can't
    hold raises ``_FolderRefused`` for that folder. ``root`` itself is never
    created: if it's gone (a drive unplugged, say), copies must not end up
    on whatever disk is left in its place.
    """
    missing = []
    while not path.exists():
        if path == root:
            raise FileNotFoundError(errno.ENOENT, "the destination folder is gone", str(root))
        missing.append(path)
        path = path.parent
    created: list[Path] = []
    try:
        for folder in reversed(missing):
            try:
                os.mkdir(folder, 0o700)
            except FileExistsError:
                continue
            except OSError as exc:
                if exc.errno in _PER_PATH_ERRNOS:
                    raise _FolderRefused(folder, exc) from exc
                raise
            created.append(folder)
    except BaseException:
        _remove_created(created)
        raise
    return created


def _remove_created(created: list[Path]) -> None:
    """Remove folders As If just created, if still empty (their names stay private)."""
    for folder in reversed(created):
        with contextlib.suppress(OSError):
            os.rmdir(folder)


class _ReadBackMismatch(Exception):
    """The copy on the destination isn't byte-identical to what was checked."""


def _no_wider_than_the_original(fd: int, mode: int, gid: int | None) -> int:
    """Make a new copy no more open than its original; returns the mode to set.

    A file created in the destination folder can pick up that folder's
    group (setgid folders, or a different primary group) and its default
    ACL. Either would hand the copy to people who couldn't read the
    original. So the inherited ACL goes, and the copy keeps the original's
    group, or gets no group access at all. (An NFSv4 ACL can't be removed
    this way; ``_commit`` deals with it once the mode is set.)
    """
    removexattr = getattr(os, "removexattr", None)  # Linux
    if removexattr is not None:
        try:
            removexattr(fd, "system.posix_acl_access")
        except OSError as exc:
            # EPERM: the file shows as someone else's (an ID-mapped share).
            # An ACL left on it is neutralised by ``_kept_private``.
            if exc.errno not in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM):
                raise
    if gid is not None and os.fstat(fd).st_gid != gid:
        try:
            os.fchown(fd, -1, gid)
        except OSError:
            # Its group stays someone else's: give it only what everyone (the
            # original's group and all others) could already do.
            mode = _everyones_group(mode)
    return mode


def _everyones_group(mode: int) -> int:
    """``mode`` with the group given only what everyone else gets too."""
    return (mode & ~0o070) | (mode & (mode & 0o007) << 3)


def _check_root(root: Path | None, root_id: tuple[int, int] | None) -> None:
    """Raise if ``root`` is no longer the folder the run started with.

    A drive unmounted from it leaves an empty folder of the same name on
    the disk underneath, which mustn't get the copies.
    """
    if root is None or root_id is None:
        return
    try:
        st = os.stat(root)
    except OSError as exc:
        raise FileNotFoundError(errno.ENOENT, "the destination folder is gone", str(root)) from exc
    if (st.st_dev, st.st_ino) != root_id:
        raise FileNotFoundError(errno.ENOENT, "the destination folder is gone", str(root))


def _commit(
    ram_copy: Path,
    dest: Path,
    mode: int,
    *,
    gid: int | None = None,
    expect_sha: str | None = None,
    root: Path | None = None,
    root_id: tuple[int, int] | None = None,
) -> bool:
    """Write ``ram_copy`` to ``dest`` without ever replacing an existing file.

    Returns False if ``dest`` appeared in the meantime. The bytes go to a
    temp file next to ``dest`` first and are hard-linked into place (atomic
    and no-clobber); on filesystems without hard links (FAT, exFAT, many
    FUSE mounts such as rclone) it falls back to an exclusive create.
    The temp name is short and fixed-length, so a long photo name can't push
    it past the filesystem's name limit. The copy gets ``mode``, the
    original's group ``gid`` (or no group access) and no inherited ACL,
    all before it holds a byte; ``_InheritedAcl`` is raised if an NFSv4 ACL
    would still let others read it, and ``_ModeNotKept`` if the destination
    gives it more read access than ``mode`` anyway. Folders created for a
    copy that isn't placed are removed again, and nothing is written once
    ``root`` isn't the folder with ``root_id`` any more.

    With ``expect_sha``, the placed copy is read back and must hash to it,
    or ``_ReadBackMismatch`` is raised. A copy that was placed but not
    confirmed (mismatch, error, Ctrl+C, even mid-placement) is removed again,
    so nothing unverified is ever left for a later run to take as done.
    """
    _check_root(root, root_id)
    created = _make_private_dirs(dest.parent, root)
    tmp = dest.parent / f".asif-{secrets.token_hex(8)}.tmp"
    done = False
    try:
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            if exc.errno in _PER_PATH_ERRNOS - {errno.EEXIST}:
                raise _FolderRefused(dest.parent, exc) from exc  # the temp name is short and plain
            raise
        with ram_copy.open("rb") as fin, os.fdopen(fd, "wb") as fout:
            mode = _no_wider_than_the_original(fout.fileno(), mode, gid)
            os.fchmod(fout.fileno(), mode & 0o777)
            _kept_private(fout.fileno(), mode, dest.parent)
            while chunk := fin.read(_COPY_CHUNK):
                fout.write(chunk)
            fout.flush()
            os.fsync(fout.fileno())
        st = ram_copy.stat()
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        ours = os.stat(tmp)
        _check_root(root, root_id)  # once more, just before it's placed
        placed = False
        try:
            try:
                os.link(tmp, dest)
                placed = True
            except FileExistsError:
                return False
            except OSError as exc:
                no_links = (errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EXDEV)
                if exc.errno not in no_links:
                    raise
                # No hard links here. An exclusive rename isn't portable, so
                # do the next best thing: refuse if anything is there, then move.
                if dest.exists():
                    return False
                os.replace(tmp, dest)
                placed = True
            _fsync_dir(dest.parent)
            if expect_sha is not None and sha256_file(dest) != expect_sha:
                raise _ReadBackMismatch
        except BaseException:
            # Remove what we placed, never someone else's file in the way.
            # samestat covers a stop inside os.link; ``placed`` covers mounts
            # that give each name its own inode number (sshfs, CIFS).
            with contextlib.suppress(OSError):
                if placed or os.path.samestat(os.stat(dest, follow_symlinks=False), ours):
                    dest.unlink()
            raise
        done = True
        return True
    finally:
        tmp.unlink(missing_ok=True)
        if not done:
            _remove_created(created)


Done = Literal["done", "stale", "no", "unknown"]


def _already_done(et: ExifTool, src: Path, dest: Path) -> Done:
    """Is ``dest`` As If's own finished copy of the current ``src``?"""
    try:
        tags = et.read_tags(dest, "-XMP-asif:RunId", "-XMP-asif:SourceSHA256")
        if "XMP-asif:RunId" not in tags:
            return "no"
        recorded = tags.get("XMP-asif:SourceSHA256")
        if recorded and str(recorded) == fingerprint(src).sha256:
            return "done"
        # Not this exact original. The gates proved a copy's image data equals
        # its original's, so the same pixels mean "same photo, since edited";
        # different pixels mean another photo is in the way (e.g. two names
        # that collide on a case-insensitive destination).
        same_photo = et.image_data_hash(dest) == et.image_data_hash(src)
        if recorded:
            return "stale" if same_photo else "no"
        return "done" if same_photo else "no"  # copies from before the hash was recorded
    except (ExifToolError, ValueError, OSError):
        # ValueError: a name exiftool can't be given (line break). Don't guess.
        return "unknown"


class _FailureBudget:
    """Abort when more than ``rate`` of the last ``FAILURE_WINDOW`` items failed (§11)."""

    def __init__(self, rate: float) -> None:
        self.limit = rate * FAILURE_WINDOW
        self.window: deque[bool] = deque(maxlen=FAILURE_WINDOW)

    def record(self, failed: bool) -> None:
        self.window.append(failed)
        if sum(self.window) > self.limit:
            raise RunAborted(
                f"{sum(self.window)} of the last {len(self.window)} photos failed, which is "
                "more than the failure budget allows. Something systematic is wrong; "
                "nothing more will be written."
            )


@dataclass
class _Ctx:
    cfg: Config
    model: ModelChoice
    et: ExifTool
    client: InferenceClient
    work: RamWorkDir
    log: TimingsLog
    run_id: str
    dry_run: bool
    dest_root: Path
    blocked: dict[Path, str] = field(default_factory=dict)
    """Destination folders As If found it can't use, and why."""
    usable: set[Path] = field(default_factory=set)
    """Destination folders a test file could be written to (and kept private) in."""
    forced: dict[Path, int] = field(default_factory=dict)
    """Read bits destination folders give every new file, where others can get to it."""
    dest_id: tuple[int, int] | None = None
    """The destination folder's (device, inode) when the run started."""


def _describe(ctx: _Ctx, ram: Path, img: str) -> CaptionResult:
    """Ask the model, validate, retry schema failures. Raises SchemaError/InferenceError."""
    last: SchemaError | None = None
    for attempt in range(SCHEMA_RETRIES + 1):
        with ctx.log.timed("infer", img) as ev:
            raw = ctx.client.describe(ram, img, attempt=attempt)
            try:
                return parse_caption_json(raw)
            except SchemaError as exc:
                ev["ok"] = False
                ev["error"] = "SchemaError"
                last = exc
    assert last is not None
    raise last


def _only_minor_warnings(stderr: str) -> bool:
    """True if exiftool only objected to this file's own metadata.

    exiftool marks these "[minor]" (it would repair something on the way)
    or "[Minor]" (it would drop part of the XMP it can't carry over).
    """
    lines = [line for line in stderr.splitlines() if line.strip()]
    return bool(lines) and all(
        line.startswith(("Warning: [minor]", "Warning: [Minor]")) for line in lines
    )


def _nearest_existing(folder: Path, root: Path) -> Path:
    """``folder``, or the closest folder above it (up to ``root``) that exists.

    That is the folder a copy is refused by: its own folder, or the one a
    missing folder would have to be created in (or can't be searched past).
    """
    while folder != root and root in folder.parents:
        try:
            os.stat(folder)
        except OSError:
            folder = folder.parent
            continue
        return folder
    return root


def _shown(ctx: _Ctx, folder: Path) -> str:
    """A destination folder as the summary names it."""
    if folder == ctx.dest_root:
        return "the destination folder"
    return f"the destination folder {folder.relative_to(ctx.dest_root).as_posix()}/"


def _blocked_folder(ctx: _Ctx, rel: str, dest: Path) -> ItemOutcome | None:
    """The outcome for a photo whose destination folder already refused a copy.

    That is the nearest existing folder, or a missing one on the way to it
    that couldn't be created (nothing below that can exist either).
    """
    if not ctx.blocked:
        return None
    nearest = _nearest_existing(dest.parent, ctx.dest_root)
    for folder in (dest.parent, *dest.parent.parents):
        if (why := ctx.blocked.get(folder)) is not None:
            return ItemOutcome(rel, State.FAILED_UPLOAD, why)
        if folder == nearest:
            return None
    return None


def _block(ctx: _Ctx, rel: str, folder: Path, why: str) -> ItemOutcome:
    """Skip every later photo bound for ``folder`` before the model runs."""
    ctx.blocked[folder] = why
    return ItemOutcome(rel, State.FAILED_UPLOAD, why)


def _destination_problem(ctx: _Ctx, rel: str, dest: Path, exc: OSError) -> ItemOutcome:
    """One photo's destination path is unusable; or raise if the destination is."""
    if exc.errno in _PER_PATH_ERRNOS:
        return ItemOutcome(rel, State.FAILED_UPLOAD, f"couldn't write the copy ({exc.strerror})")
    # Permission denied is one folder's answer (someone else's in a shared
    # destination, or a 0700 one from another user's run): skip that folder,
    # not the run. Folders below it that the user can write to still work.
    if _dest_gone(ctx):
        raise _DestinationError(_gone(ctx)) from exc
    if exc.errno == errno.EACCES:
        folder = _nearest_existing(dest.parent, ctx.dest_root)
        return _block(ctx, rel, folder, f"can't write into {_shown(ctx, folder)}")
    raise _DestinationError(f"can't write to the destination ({exc.strerror or exc})") from exc


def _permissions_not_kept(dest: Path, src: Path) -> bool:
    """Does the destination give new files and folders more access than As If
    sets, with people who can't see the originals able to get to them?

    Tested for real with a file and a folder, then removed again. A root
    that takes neither (a share where only some subfolders are yours) can't
    be tested here; each copy is still checked as it's written.
    """
    forced = 0
    ignored = False
    name = f".asif-{secrets.token_hex(8)}"
    try:
        fd = os.open(dest / f"{name}.tmp", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            if _shared_mount(mount_of(str(dest.resolve()))) and _chmod_ignored(fd):
                ignored = True  # nothing As If sets holds there
            _no_wider_than_the_original(fd, 0o600, None)
            os.fchmod(fd, 0o600)
            forced |= stat.S_IMODE(os.fstat(fd).st_mode) & 0o077
        finally:
            os.close(fd)
            (dest / f"{name}.tmp").unlink(missing_ok=True)
        os.mkdir(dest / name, 0o700)
        try:
            forced |= stat.S_IMODE(os.stat(dest / name).st_mode) & 0o077
        finally:
            os.rmdir(dest / name)
    except PermissionError:
        return False
    if not ignored and (not forced or _folder_audience(dest, wide=True, bits=0o111) == _NOBODY):
        return False
    return _source_is_private(src)


def _dest_gone(ctx: _Ctx) -> bool:
    """Is the destination folder gone, or no longer the one the run started with?"""
    try:
        st = os.stat(ctx.dest_root)
    except OSError:
        return True
    return ctx.dest_id is not None and (st.st_dev, st.st_ino) != ctx.dest_id


def _gone(ctx: _Ctx) -> str:
    return f"the destination folder {ctx.dest_root} is gone (unplugged or unmounted?)"


def _try_folder(ctx: _Ctx, rel: str, dest: Path) -> ItemOutcome | None:
    """Make sure a copy can go where ``dest`` is, before the model runs.

    A test file is written once in each folder a copy (or a folder for it)
    would be created in: a folder that refuses it, or gives it access rules
    As If can't remove, is skipped without spending a model run per photo.
    """
    if _dest_gone(ctx):
        raise _DestinationError(_gone(ctx))
    folder = _nearest_existing(dest.parent, ctx.dest_root)
    if folder in ctx.usable:
        return None
    probe = folder / f".asif-{secrets.token_hex(8)}.tmp"
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            if _shared_mount(mount_of(str(folder.resolve()))) and _chmod_ignored(fd):
                ctx.forced[folder] = 0o044  # nothing As If sets holds there
            _no_wider_than_the_original(fd, 0o600, None)
            os.fchmod(fd, 0o600)
            _kept_private(fd, 0o600, folder)
        finally:
            os.close(fd)
            probe.unlink(missing_ok=True)
    except _InheritedAcl:
        return _block(ctx, rel, folder, _inherited_acl_reason(ctx, folder))
    except _ModeNotKept as kept:
        ctx.forced[folder] = kept.forced  # fine for photos others may read anyway
    except OSError as exc:
        return _destination_problem(ctx, rel, dest, exc)
    ctx.usable.add(folder)
    return None


def _inherited_acl_reason(ctx: _Ctx, folder: Path) -> str:
    return (
        f"{_shown(ctx, folder)} gives new files extra access rules (an inherited ACL) "
        "that As If can't remove"
    )


def _listable_folder(ctx: _Ctx, src: Path, rel: str) -> Path | None:
    """The first existing destination subfolder more people can list than its original.

    Names alone can say a lot, so a copy never goes where its own name, or
    a folder name above it, would be visible to people who can't see the
    original's. Folders As If creates are 0700, so only existing ones count.
    """
    folders = Path(rel).parents[:-1]  # "a/b/x.jpg": a/b, a (the root is checked up front)
    src_root = src.parents[len(folders)]
    for folder in reversed(folders):
        mine = ctx.dest_root / folder
        if not mine.is_dir():
            return None
        if _more_people_can_list(mine, src_root / folder):
            return mine
    return None


def _source_problem(rel: str, exc: OSError, when: str) -> ItemOutcome:
    """A per-photo outcome for an original that can't be read (never systematic)."""
    if isinstance(exc, FileNotFoundError):
        return ItemOutcome(rel, State.SKIPPED_CONFLICT, f"the original disappeared {when}")
    if isinstance(exc, _NotAFile):
        return ItemOutcome(rel, State.QUARANTINE, "not a regular file")
    return ItemOutcome(rel, State.QUARANTINE, f"can't read the original ({exc.strerror or exc})")


def _copy_mode(ctx: _Ctx, src: Path, src_st: os.stat_result, dest: Path) -> int:
    """The mode a copy gets: no more than its original effectively had, and on
    a share, no more for the group than for everyone (its groups are the
    server's, whatever it shows)."""
    mode = src_st.st_mode & _reach_mask(src)
    nearest = _nearest_existing(dest.parent, ctx.dest_root).resolve()
    if _shared_mount(mount_of(str(nearest))):
        mode = _everyones_group(mode)
    return mode


def _process(ctx: _Ctx, src: Path, dest: Path, rel: str, img: str) -> ItemOutcome:
    suffix = src.suffix.lower()
    if suffix in NOT_YET:
        return ItemOutcome(rel, State.SKIPPED_FORMAT, NOT_YET[suffix])

    occupied = False
    if not ctx.dry_run:
        try:
            occupied = dest.exists() or dest.is_symlink()
        except OSError as exc:
            return _destination_problem(ctx, rel, dest, exc)
    if occupied:
        done = _already_done(ctx.et, src, dest)
        if done == "done":
            return ItemOutcome(rel, State.SKIPPED_DONE, "already captioned")
        reason = {
            "stale": "the original changed after its copy was made; delete the copy to redo it",
            "unknown": "a file is already at the destination and can't be checked",
        }.get(done, "a different file is already at the destination (As If never overwrites)")
        return ItemOutcome(rel, State.SKIPPED_CONFLICT, reason)

    if (blocked := _blocked_folder(ctx, rel, dest)) is not None:
        return blocked
    if not ctx.dry_run:
        try:
            if (folder := _listable_folder(ctx, src, rel)) is not None:
                why = f"other users can list {_shown(ctx, folder)}, but not the original's folder"
                return _block(ctx, rel, folder, why)
        except _SourceUnreadable as gone:
            return _source_problem(rel, gone.exc, "before As If got to it")
        except OSError as exc:
            return _destination_problem(ctx, rel, dest, exc)
        if (refused := _try_folder(ctx, rel, dest)) is not None:
            return refused

    try:
        src_st = src.stat()
    except OSError as exc:
        return _source_problem(rel, exc, "before As If got to it")
    size = src_st.st_size
    if not ctx.dry_run and ctx.forced:
        forced = ctx.forced.get(_nearest_existing(dest.parent, ctx.dest_root), 0)
        if forced & ~_copy_mode(ctx, src, src_st, dest):
            return ItemOutcome(rel, State.FAILED_UPLOAD, _MODE_NOT_KEPT)
    if size * 3 > ctx.work.free_bytes():
        return ItemOutcome(rel, State.QUARANTINE, "too big for the RAM work dir")

    ram = ctx.work.new_path(suffix)
    try:
        try:
            with ctx.log.timed("open", img):
                fp_then = _copy_and_fingerprint(src, ram)
        except OSError as exc:
            return _source_problem(rel, exc, "before As If got to it")
        try:
            with ctx.log.timed("snapshot", img):
                before = snapshot(ctx.et, ram)
        except ExifToolError:
            return ItemOutcome(rel, State.QUARANTINE, "exiftool can't read it (corrupt?)")

        # Checked before the (slow) model: a HEIC or PNG named .jpg is a
        # problem with this file, not a sign that something is broken.
        expected = _SUFFIX_FORMAT[suffix]
        if before.fmt != expected:
            actual = before.fmt.upper() if before.fmt != "other" else "another format"
            return ItemOutcome(
                rel,
                State.QUARANTINE,
                f"the name says {expected.upper()} but the file is {actual}; "
                "fix the extension to caption it",
            )
        # A file the gates couldn't prove a write safe for would fail them
        # after the model's work; say why now, as a limit of this file.
        if before.unprovable is not None:
            if before.unprovable.startswith("its structure can't be checked"):
                return ItemOutcome(rel, State.QUARANTINE, before.unprovable)
            return ItemOutcome(
                rel, State.SKIPPED_FORMAT, f"can't be written safely: {before.unprovable}"
            )

        try:
            result = _describe(ctx, ram, img)
        except SchemaError:
            return ItemOutcome(
                rel,
                State.QUARANTINE,
                "the model's answer never matched the schema",
                systematic=True,
            )
        except InferenceError as exc:
            if exc.kind == "image_decode":
                return ItemOutcome(rel, State.QUARANTINE, "the image can't be decoded (corrupt?)")
            raise

        lang = ctx.cfg.general.caption_lang
        compat = ctx.cfg.general.compat
        try:
            existing = read_existing(ctx.et, ram)
            plan = plan_write(
                existing, result, lang=lang, compat=compat, image_hash=before.image_hash
            )
        except (ExifToolError, ValueError):
            return ItemOutcome(
                rel, State.FAILED_GATE, "couldn't read the existing metadata", systematic=True
            )

        if ctx.dry_run:
            return ItemOutcome(rel, State.DESCRIBED, plan.caption_reason, result=result, plan=plan)

        prov = Provenance(
            tool_version=__version__,
            model_id=ctx.model.model_id,
            model_sha256=ctx.model.model_sha256,
            prompt_version=PROMPT_VERSION,
            run_id=ctx.run_id,
            timestamp=datetime.now(UTC).strftime("%Y:%m:%d %H:%M:%SZ"),
            source_sha256=fp_then.sha256,
        )
        try:
            with ctx.log.timed("write", img):
                record = apply_write(ctx.et, ram, plan, prov, lang=lang)
        except ExifToolError as exc:
            if _only_minor_warnings(exc.stderr):
                # A property of this photo's metadata, not a bug.
                return ItemOutcome(
                    rel,
                    State.SKIPPED_FORMAT,
                    "exiftool can't rewrite this file's metadata without repairing or "
                    "dropping part of it",
                )
            return ItemOutcome(
                rel, State.FAILED_GATE, "exiftool refused the write", systematic=True
            )
        except ValueError:
            # The writer refused an inconsistent plan: can't happen with
            # validated input, so it's a bug.
            return ItemOutcome(
                rel, State.FAILED_GATE, "exiftool refused the write", systematic=True
            )

        with ctx.log.timed("gates", img) as ev:
            gates = check_written(ctx.et, before, ram, compat=compat)
            ev["ok"] = all_ok(gates)
        if not all_ok(gates):
            names = ", ".join(
                f"{g.name} ({g.detail})" if g.detail else g.name for g in gates if not g.ok
            )
            return ItemOutcome(rel, State.FAILED_GATE, f"gate failed: {names}", systematic=True)

        try:
            unchanged = check_source_unchanged(fp_then, fingerprint(src))
            src_st = src.stat()
            mode = _copy_mode(ctx, src, src_st, dest)
        except OSError as exc:
            return _source_problem(rel, exc, "while As If was working on it")
        if not unchanged.ok:
            return ItemOutcome(
                rel, State.SKIPPED_CONFLICT, "the original changed while As If was working on it"
            )

        processed_sha = sha256_file(ram)
        try:
            with ctx.log.timed("commit", img):
                committed = _commit(
                    ram,
                    dest,
                    mode,
                    gid=src_st.st_gid,
                    expect_sha=processed_sha,
                    root=ctx.dest_root,
                    root_id=ctx.dest_id,
                )
        except _InheritedAcl:
            # The folders made for it are gone again: the rule comes from above.
            folder = _nearest_existing(dest.parent, ctx.dest_root)
            return _block(ctx, rel, folder, _inherited_acl_reason(ctx, folder))
        except _ModeNotKept:
            return ItemOutcome(rel, State.FAILED_UPLOAD, _MODE_NOT_KEPT)
        except _FolderRefused as refused:
            folder = refused.folder
            why = (
                f"the destination can't hold a folder named {folder.name!r} "
                f"({refused.exc.strerror or refused.exc})"
            )
            return _block(ctx, rel, folder, why)
        except _ReadBackMismatch:
            return ItemOutcome(
                rel,
                State.FAILED_UPLOAD,
                "the written copy didn't read back identically; removed it",
                systematic=True,
            )
        except OSError as exc:
            return _destination_problem(ctx, rel, dest, exc)
        if not committed:
            return ItemOutcome(
                rel, State.SKIPPED_CONFLICT, "a file appeared at the destination meanwhile"
            )
        note = "; ".join(n for n in (record.iptc_note, record.subject_note) if n)
        return ItemOutcome(rel, State.CLEAN, plan.caption_reason, note=note)
    finally:
        wipe_file(ram)


def _destination_ok(dest: Path, src: Path) -> None:
    """Raise ``CaptionError`` if copies of ``src`` mustn't go into ``dest``."""
    if _shared_mount(mount_of(str(dest.resolve()))) and _source_is_private(src):
        raise CaptionError(
            f"{dest} is on a network drive, and As If can't tell who can see files there "
            "(the share decides, not this computer), so it doesn't put copies of private "
            "photos on one yet. Caption into a folder on this computer; syncing to a NAS or "
            "cloud, encrypted, comes in a later version."
        )
    if _permissions_not_kept(dest, src):
        raise CaptionError(
            f"other users on this computer could see and open the copies in {dest}: "
            "its drive gives every file the same permissions (FAT and exFAT drives "
            "often do), so As If can't keep them private. Use a folder only you can "
            "open, such as one in your home folder, or mount the drive just for you "
            "(for example with the options uid=$(id -u),umask=077)."
        )
    if _more_people_can_list(dest, src):
        raise CaptionError(
            f"other users can list the destination {dest} but not {src}, so they "
            f"would see your photos' names. Run `chmod 700 {dest}` if it's yours, or "
            "name a new folder (As If creates new ones private)."
        )


def _source_is_private(src: Path) -> bool:
    """Can anyone besides us not list the source folder?"""
    try:
        return _folder_audience(src, wide=False) != _ANYONE
    except OSError as exc:
        raise _SourceUnreadable(exc) from exc


def caption_folder(
    src: Path,
    dest: Path,
    cfg: Config,
    *,
    model: ModelChoice,
    run_id: str,
    log: TimingsLog,
    dry_run: bool = False,
    limit: int | None = None,
    exiftool: str = "exiftool",
    on_progress: ProgressFn | None = None,
    on_start: Callable[[int], None] | None = None,
    notice: NoticeFn | None = None,
) -> RunSummary:
    """Caption every photo under ``src`` into copies under ``dest``.

    Raises ``CaptionError`` if the run can't start and ``InferenceError`` if
    the model worker can't start. Everything after that ends up in the
    returned summary, including why the run stopped early, if it did.
    """
    src, dest = check_folders(src, dest)
    found = discover(src)
    files = found.files[:limit] if limit is not None else list(found.files)
    summary = RunSummary(
        run_id=run_id, dry_run=dry_run, ignored=found.ignored, unreadable_dirs=found.unreadable_dirs
    )
    if on_start:
        on_start(len(files))
    if not files:
        return summary
    # A random order, so the log's per-run ids (i_000001, ...) don't follow
    # the folder listing. (The copies' own file times still show the order
    # they were written in, so the log is only as private as that folder.)
    secrets.SystemRandom().shuffle(files)

    if not dry_run:
        created: list[Path] = []
        try:
            try:
                created = _make_private_dirs(dest)
                _destination_ok(dest, src)
            except BaseException:
                # Nothing it made is left behind (its name would say something).
                _remove_created(created)
                raise
        except _FolderRefused as exc:
            raise CaptionError(
                f"can't create the destination folder {exc.folder}: {exc.exc.strerror}"
            ) from exc
        except _SourceUnreadable as exc:
            raise CaptionError(f"can't read the source folder {src}: {exc.exc.strerror}") from exc
        except OSError as exc:
            raise CaptionError(
                f"can't create the destination folder {dest}: {exc.strerror}"
            ) from exc

    completed = False
    try:
        with proc.ShieldedExitStack() as stack:
            # Unwinds in reverse: the model worker stops, the RAM copies are
            # wiped, then exiftool closes. Also on Ctrl+C and stop signals.
            et = ExifTool(exiftool)
            try:
                et.start()
            except ExifToolError as exc:
                raise CaptionError(str(exc)) from exc
            stack.callback(et.close)
            work = RamWorkDir(cfg.resources.work_dir, allow_disk=cfg.resources.allow_disk_workdir)
            try:
                stack.enter_context(work)
            except NotInRamError as exc:
                raise CaptionError(str(exc)) from exc
            except OSError as exc:
                raise CaptionError(f"can't create the work dir: {exc.strerror}") from exc
            if notice and not work.choice.in_ram:
                notice(f"The work dir is on disk ({work.choice.reason}); you allowed that.")
            if notice and work.swept:
                notice(
                    f"Wiped photo copies left in RAM by {work.swept} run(s) that were cut short."
                )
            if notice and work.unremembered and cfg.resources.work_dir:
                notice(
                    f"Couldn't note the work dir's folder in As If's state folder "
                    f"({work.unremembered}). If this run is cut short, what it leaves there is "
                    "only wiped by a later run with the same work_dir."
                )

            ctx = _Ctx(
                cfg, model, et, InferenceClient(model.worker), work, log, run_id, dry_run, dest
            )
            if not dry_run:
                # Held open for the run: the folder can't be renumbered (sshfs,
                # FAT), so a different one at that path means it was unmounted.
                flags = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0)
                try:
                    dest_fd = os.open(dest, flags)
                except OSError as exc:
                    raise CaptionError(f"can't open the destination folder {dest}: {exc}") from exc
                stack.callback(os.close, dest_fd)
                dest_st = os.fstat(dest_fd)
                ctx.dest_id = (dest_st.st_dev, dest_st.st_ino)
            stack.callback(lambda: ctx.client.close())
            log.event("run_start", n=len(files))
            try:
                with log.timed("model_load"):
                    ctx.client.start()
                _run(ctx, src, dest, files, summary, on_progress)
                completed = True
            finally:
                log.event(
                    "run_end", n=len(summary.outcomes), ok=completed and summary.aborted is None
                )
    except KeyboardInterrupt:
        # Ctrl+C or a stop signal, before or during the run (the cleanup above
        # can't be cut short; one arriving during it is acted on after it).
        # After every photo is done, a late signal changes nothing.
        if not completed:
            summary.interrupted = True
            summary.aborted = summary.aborted or _STOPPED
    return summary


_STOPPED = "stopped (Ctrl+C or a stop signal); nothing was left half-done"


def _run(
    ctx: _Ctx,
    src: Path,
    dest: Path,
    files: list[Path],
    summary: RunSummary,
    on_progress: ProgressFn | None,
) -> None:
    """The loop over photos. Sets ``summary.aborted`` when it stops early."""
    budget = _FailureBudget(ctx.cfg.guards.max_failure_rate)
    worker_failures = 0
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(src).as_posix()
        img = f"i_{index:06d}"
        stop: str | None = None
        try:
            outcome = _process(ctx, path, dest / rel, rel, img)
            worker_failures = 0
        except InferenceError as exc:
            worker_failures += 1
            outcome = ItemOutcome(
                rel, State.QUARANTINE, f"the model worker failed ({exc.kind})", systematic=True
            )
            if worker_failures >= MAX_CONSECUTIVE_WORKER_FAILURES:
                stop = (
                    f"the model worker failed on {worker_failures} photos in a row "
                    f"({exc.kind}). {exc.detail}"
                ).strip()
            elif not ctx.client.alive:
                ctx.client.close()
                ctx.client = InferenceClient(ctx.model.worker)
                try:
                    ctx.client.start()
                except InferenceError as again:
                    stop = (
                        f"the model worker stopped ({exc.kind}) and couldn't be restarted "
                        f"({again.kind}). {again.detail or exc.detail}"
                    ).strip()
        except _DestinationError as exc:
            outcome = ItemOutcome(rel, State.FAILED_UPLOAD, str(exc), systematic=True)
            stop = f"{exc}; nothing more will be written"
        summary.outcomes.append(outcome)
        ctx.log.event("item", img=img, state=outcome.state.value, ok=not outcome.systematic)
        if on_progress:
            on_progress(outcome, index, len(files))
        if stop is None:
            try:
                budget.record(outcome.systematic)
            except RunAborted as exc:
                stop = str(exc)
        if stop is not None:
            summary.aborted = stop
            return


def iter_problems(summary: RunSummary) -> Iterator[ItemOutcome]:
    """Outcomes worth showing the user: anything that didn't simply work."""
    quiet = {State.CLEAN, State.DESCRIBED, State.SKIPPED_DONE}
    return (o for o in summary.outcomes if o.state not in quiet)
