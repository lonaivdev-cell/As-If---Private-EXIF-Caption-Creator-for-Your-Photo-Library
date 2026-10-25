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
"""

from __future__ import annotations

import hashlib
import os
import secrets
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
from asif.meta.writer import Provenance, apply_write, plan_write, read_existing
from asif.pipeline.states import State
from asif.pipeline.timings import TimingsLog
from asif.workdir import RamWorkDir, wipe_file

WRITABLE = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
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

Backend = Literal["llama", "stub"]


class CaptionError(RuntimeError):
    """The run can't start (bad folders, missing model, missing tools)."""


class RunAborted(RuntimeError):
    """The run stopped early because something looks systematically wrong."""


@dataclass
class ItemOutcome:
    rel: str
    """Path relative to the source folder. Shown in the terminal, never logged."""
    state: State
    reason: str = ""
    result: CaptionResult | None = None
    """The model's answer, kept only for --dry-run previews."""
    systematic: bool = False
    """Counts against the failure budget: a sign of a bug, not of a bad input file."""


@dataclass
class RunSummary:
    run_id: str
    dry_run: bool
    outcomes: list[ItemOutcome] = field(default_factory=list)
    ignored: int = 0
    """Files that aren't photos at all (or are hidden, or symlinks)."""
    aborted: str | None = None

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


def discover(src: Path) -> tuple[list[Path], int]:
    """Photo-ish files under ``src`` (sorted), and how many other files were ignored.

    Hidden files and folders are skipped, and symlinks are never followed, so
    a link can't pull in files from outside the folder you chose.
    """
    found: list[Path] = []
    ignored = 0
    for root, dirs, files in os.walk(src, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            path = Path(root) / name
            suffix = path.suffix.lower()
            if name.startswith(".") or path.is_symlink():
                ignored += 1
            elif suffix in WRITABLE or suffix in NOT_YET:
                found.append(path)
            else:
                ignored += 1
    return found, ignored


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
    model_id = m.id if m.id not in ("auto", "local") else f"local:{gguf.name}"
    gpu_layers = m.gpu_layers
    if gpu_layers is None and m.device == "cpu":
        gpu_layers = 0
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
        server_args=tuple(m.server_args),
    )
    return ModelChoice("llama", model_id, model_sha, worker)


# -- the run ---------------------------------------------------------------


def _copy_and_fingerprint(src: Path, dest: Path) -> Fingerprint:
    """Copy ``src`` to ``dest``, hashing exactly the bytes copied."""
    digest = hashlib.sha256()
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with src.open("rb") as fin, os.fdopen(fd, "wb") as fout:
        st = os.fstat(fin.fileno())
        while chunk := fin.read(_COPY_CHUNK):
            digest.update(chunk)
            fout.write(chunk)
    os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns))
    return Fingerprint(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest.hexdigest())


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


def _commit(ram_copy: Path, dest: Path, mode: int) -> bool:
    """Write ``ram_copy`` to ``dest`` without ever replacing an existing file.

    Returns False if ``dest`` appeared in the meantime. The bytes go to a
    temp file next to ``dest`` first and are hard-linked into place (atomic
    and no-clobber); on filesystems without hard links (FAT, exFAT) it falls
    back to an exclusive create.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.asif-{secrets.token_hex(4)}.tmp"
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
        except OSError:
            # No hard links here. An exclusive rename isn't portable, so do
            # the next best thing: refuse if anything is there, then move.
            if dest.exists():
                return False
            os.replace(tmp, dest)
        _fsync_dir(dest.parent)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def _already_done(et: ExifTool, src: Path, dest: Path) -> bool:
    """Is ``dest`` As If's own finished copy of ``src``?"""
    try:
        tags = et.read_tags(dest, "-XMP-asif:RunId")
        if "XMP-asif:RunId" not in tags:
            return False
        return et.image_data_hash(dest) == et.image_data_hash(src)
    except ExifToolError:
        return False


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


def _process(ctx: _Ctx, src: Path, dest: Path, rel: str, img: str) -> ItemOutcome:
    suffix = src.suffix.lower()
    if suffix in NOT_YET:
        return ItemOutcome(rel, State.SKIPPED_FORMAT, NOT_YET[suffix])

    if not ctx.dry_run and (dest.exists() or dest.is_symlink()):
        if _already_done(ctx.et, src, dest):
            return ItemOutcome(rel, State.SKIPPED_DONE, "already captioned")
        return ItemOutcome(
            rel,
            State.SKIPPED_CONFLICT,
            "a different file is already at the destination (As If never overwrites)",
        )

    size = src.stat().st_size
    if size * 3 > ctx.work.free_bytes():
        return ItemOutcome(rel, State.QUARANTINE, "too big for the RAM work dir")

    ram = ctx.work.new_path(suffix)
    try:
        with ctx.log.timed("open", img) as ev:
            fp_then = _copy_and_fingerprint(src, ram)
            ev["bytes"] = fp_then.size
        try:
            with ctx.log.timed("snapshot", img):
                before = snapshot(ctx.et, ram)
        except ExifToolError:
            return ItemOutcome(rel, State.QUARANTINE, "exiftool can't read it (corrupt?)")

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

        if ctx.dry_run:
            return ItemOutcome(rel, State.DESCRIBED, "dry run: nothing written", result=result)

        prov = Provenance(
            tool_version=__version__,
            model_id=ctx.model.model_id,
            model_sha256=ctx.model.model_sha256,
            prompt_version=PROMPT_VERSION,
            run_id=ctx.run_id,
            timestamp=datetime.now(UTC).strftime("%Y:%m:%d %H:%M:%SZ"),
        )
        lang = ctx.cfg.general.caption_lang
        compat = ctx.cfg.general.compat
        try:
            with ctx.log.timed("write", img):
                existing = read_existing(ctx.et, ram)
                plan = plan_write(
                    existing, result, lang=lang, compat=compat, image_hash=before.image_hash
                )
                apply_write(ctx.et, ram, plan, prov, lang=lang)
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

        unchanged = check_source_unchanged(fp_then, fingerprint(src))
        if not unchanged.ok:
            return ItemOutcome(
                rel, State.SKIPPED_CONFLICT, "the original changed while As If was working on it"
            )

        processed_sha = sha256_file(ram)
        with ctx.log.timed("commit", img):
            committed = _commit(ram, dest, src.stat().st_mode)
        if not committed:
            return ItemOutcome(
                rel, State.SKIPPED_CONFLICT, "a file appeared at the destination meanwhile"
            )
        if sha256_file(dest) != processed_sha:
            dest.unlink(missing_ok=True)
            return ItemOutcome(
                rel,
                State.FAILED_UPLOAD,
                "the written copy didn't read back identically; removed it",
                systematic=True,
            )
        return ItemOutcome(rel, State.CLEAN, plan.caption_reason)
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
) -> RunSummary:
    """Caption every photo under ``src`` into copies under ``dest``.

    Raises ``CaptionError`` if the run can't start and ``InferenceError`` if
    the model worker can't start. Per-photo problems never raise: they end up
    in the summary.
    """
    src, dest = check_folders(src, dest)
    files, ignored = discover(src)
    if limit is not None:
        files = files[:limit]
    summary = RunSummary(run_id=run_id, dry_run=dry_run, ignored=ignored)
    if on_start:
        on_start(len(files))
    if not files:
        return summary

    budget = _FailureBudget(cfg.guards.max_failure_rate)
    log.event("run_start", n=len(files))
    client = InferenceClient(model.worker)
    try:
        with ExifTool(exiftool) as et, RamWorkDir(cfg.resources.work_dir) as work:
            with log.timed("model_load"):
                client.start()
            ctx = _Ctx(cfg, model, et, client, work, log, run_id, dry_run)
            worker_failures = 0
            for index, path in enumerate(files, start=1):
                rel = path.relative_to(src).as_posix()
                img = f"i_{index:06d}"
                try:
                    outcome = _process(ctx, path, dest / rel, rel, img)
                    worker_failures = 0
                except InferenceError as exc:
                    worker_failures += 1
                    outcome = ItemOutcome(
                        rel,
                        State.QUARANTINE,
                        f"the model worker failed ({exc.kind})",
                        systematic=True,
                    )
                    if worker_failures >= MAX_CONSECUTIVE_WORKER_FAILURES:
                        summary.outcomes.append(outcome)
                        summary.aborted = (
                            f"the model worker failed on {worker_failures} photos in a row "
                            f"({exc.kind}). {exc.detail}"
                        ).strip()
                        break
                    if not ctx.client.alive:
                        ctx.client.close()
                        ctx.client = client = InferenceClient(model.worker)
                        client.start()
                summary.outcomes.append(outcome)
                log.event("item", img=img, state=outcome.state.value, ok=not outcome.systematic)
                if on_progress:
                    on_progress(outcome, index, len(files))
                try:
                    budget.record(outcome.systematic)
                except RunAborted as exc:
                    summary.aborted = str(exc)
                    break
    finally:
        client.close()
        log.event("run_end", n=len(summary.outcomes), ok=summary.aborted is None)
    return summary


def iter_problems(summary: RunSummary) -> Iterator[ItemOutcome]:
    """Outcomes worth showing the user: anything that didn't simply work."""
    quiet = {State.CLEAN, State.DESCRIBED, State.SKIPPED_DONE}
    return (o for o in summary.outcomes if o.state not in quiet)
