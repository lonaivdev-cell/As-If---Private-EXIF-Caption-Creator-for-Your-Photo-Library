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
import hashlib
import os
import secrets
import stat
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from asif import __version__
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
from asif.workdir import NotInRamError, RamWorkDir, wipe_file

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


def _reach_mask(path: Path) -> int:
    """Permission bits ``path``'s folders actually let group/others use.

    A 0644 photo inside a 0700 folder is private; its copy must not become
    readable just because the destination folder is open. ACLs are ignored,
    so this can only narrow access.
    """
    mask = 0o777
    for parent in path.parents:
        try:
            mode = parent.stat().st_mode
        except OSError:
            return 0o700
        if not mode & stat.S_IXOTH:
            mask &= ~0o007
        if not mode & stat.S_IXGRP:
            mask &= ~0o070
    return mask


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


def _make_private_dirs(path: Path) -> None:
    """Create ``path`` and any missing parents as 0700, one level at a time."""
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    for folder in reversed(missing):
        with contextlib.suppress(FileExistsError):
            os.mkdir(folder, 0o700)


def _commit(ram_copy: Path, dest: Path, mode: int) -> bool:
    """Write ``ram_copy`` to ``dest`` without ever replacing an existing file.

    Returns False if ``dest`` appeared in the meantime. The bytes go to a
    temp file next to ``dest`` first and are hard-linked into place (atomic
    and no-clobber); on filesystems without hard links (FAT, exFAT) it falls
    back to an exclusive create. The temp name is short and fixed-length, so
    a long photo name can't push it past the filesystem's name limit.
    """
    _make_private_dirs(dest.parent)
    tmp = dest.parent / f".asif-{secrets.token_hex(8)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with ram_copy.open("rb") as fin, os.fdopen(fd, "wb") as fout:
            while chunk := fin.read(_COPY_CHUNK):
                fout.write(chunk)
            fout.flush()
            os.fsync(fout.fileno())
        st = ram_copy.stat()
        os.chmod(tmp, mode & 0o777)
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        try:
            os.link(tmp, dest)
        except FileExistsError:
            return False
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EXDEV):
                raise
            # No hard links here. An exclusive rename isn't portable, so do
            # the next best thing: refuse if anything is there, then move.
            if dest.exists():
                return False
            os.replace(tmp, dest)
        _fsync_dir(dest.parent)
        return True
    finally:
        tmp.unlink(missing_ok=True)


Done = Literal["done", "stale", "no", "unknown"]


def _already_done(et: ExifTool, src: Path, dest: Path) -> Done:
    """Is ``dest`` As If's own finished copy of the current ``src``?"""
    try:
        tags = et.read_tags(dest, "-XMP-asif:RunId", "-XMP-asif:SourceSHA256")
        if "XMP-asif:RunId" not in tags:
            return "no"
        recorded = tags.get("XMP-asif:SourceSHA256")
        if recorded:
            return "done" if str(recorded) == fingerprint(src).sha256 else "stale"
        # Copies made before the source hash was recorded.
        return "done" if et.image_data_hash(dest) == et.image_data_hash(src) else "stale"
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


def _source_problem(rel: str, exc: OSError, when: str) -> ItemOutcome:
    """A per-photo outcome for an original that can't be read (never systematic)."""
    if isinstance(exc, FileNotFoundError):
        return ItemOutcome(rel, State.SKIPPED_CONFLICT, f"the original disappeared {when}")
    if isinstance(exc, _NotAFile):
        return ItemOutcome(rel, State.QUARANTINE, "not a regular file")
    return ItemOutcome(rel, State.QUARANTINE, f"can't read the original ({exc.strerror or exc})")


def _process(ctx: _Ctx, src: Path, dest: Path, rel: str, img: str) -> ItemOutcome:
    suffix = src.suffix.lower()
    if suffix in NOT_YET:
        return ItemOutcome(rel, State.SKIPPED_FORMAT, NOT_YET[suffix])

    if not ctx.dry_run and (dest.exists() or dest.is_symlink()):
        done = _already_done(ctx.et, src, dest)
        if done == "done":
            return ItemOutcome(rel, State.SKIPPED_DONE, "already captioned")
        reason = {
            "stale": "the original changed after its copy was made; delete the copy to redo it",
            "unknown": "a file is already at the destination and can't be checked",
        }.get(done, "a different file is already at the destination (As If never overwrites)")
        return ItemOutcome(rel, State.SKIPPED_CONFLICT, reason)

    try:
        size = src.stat().st_size
    except OSError as exc:
        return _source_problem(rel, exc, "before As If got to it")
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
        )
        try:
            with ctx.log.timed("write", img):
                record = apply_write(ctx.et, ram, plan, prov, lang=lang)
        except (ExifToolError, ValueError):
            # ValueError: the writer refused an inconsistent plan. Neither can
            # happen with validated input, so both count as systematic.
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
            mode = src.stat().st_mode & _reach_mask(src)
        except OSError as exc:
            return _source_problem(rel, exc, "while As If was working on it")
        if not unchanged.ok:
            return ItemOutcome(
                rel, State.SKIPPED_CONFLICT, "the original changed while As If was working on it"
            )

        processed_sha = sha256_file(ram)
        committed = False
        try:
            with ctx.log.timed("commit", img):
                committed = _commit(ram, dest, mode)
            if committed and sha256_file(dest) != processed_sha:
                dest.unlink(missing_ok=True)
                return ItemOutcome(
                    rel,
                    State.FAILED_UPLOAD,
                    "the written copy didn't read back identically; removed it",
                    systematic=True,
                )
        except OSError as exc:
            if committed:
                dest.unlink(missing_ok=True)  # never leave an unverified copy behind
            if exc.errno in _PER_PATH_ERRNOS:
                return ItemOutcome(
                    rel, State.FAILED_UPLOAD, f"couldn't write the copy ({exc.strerror})"
                )
            raise _DestinationError(
                f"can't write to the destination ({exc.strerror or exc})"
            ) from exc
        if not committed:
            return ItemOutcome(
                rel, State.SKIPPED_CONFLICT, "a file appeared at the destination meanwhile"
            )
        return ItemOutcome(rel, State.CLEAN, plan.caption_reason, note=record.iptc_note or "")
    finally:
        wipe_file(ram)


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
    # A random order, so the log's per-run ids (i_000001, ...) say nothing
    # about which file is which, even to someone who can list the folder.
    secrets.SystemRandom().shuffle(files)

    if not dry_run:
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CaptionError(
                f"can't create the destination folder {dest}: {exc.strerror}"
            ) from exc

    with contextlib.ExitStack() as stack:
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
            notice(f"Wiped photo copies left in RAM by {work.swept} run(s) that were cut short.")

        ctx = _Ctx(cfg, model, et, InferenceClient(model.worker), work, log, run_id, dry_run)
        stack.callback(lambda: ctx.client.close())
        log.event("run_start", n=len(files))
        completed = False
        try:
            with log.timed("model_load"):
                ctx.client.start()
            _run(ctx, src, dest, files, summary, on_progress)
            completed = True
        except KeyboardInterrupt:
            summary.interrupted = True
            summary.aborted = _STOPPED
        finally:
            log.event("run_end", n=len(summary.outcomes), ok=completed and summary.aborted is None)
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
