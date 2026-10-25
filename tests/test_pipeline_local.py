"""End-to-end M1: folder in -> captioned copies out, with the stub model.

This is the M1 checkpoint from §19 of docs/DESIGN.md, on synthetic photos:
every captioned copy has the same ImageDataHash as its original, dates and
GPS are byte-identical, and the originals are never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from asif.config import Config
from asif.meta.exiftool import ExifTool
from asif.meta.gates import Fingerprint, GateResult
from asif.pipeline import local
from asif.pipeline.local import (
    CaptionError,
    RunSummary,
    caption_folder,
    check_folders,
    choose_model,
    discover,
)
from asif.pipeline.states import State
from asif.pipeline.timings import TimingsLog
from tests.conftest import FIXTURE_EXPECT, requires_exiftool

pytestmark = requires_exiftool

EXPECTED_STATE = {
    "ok": State.CLEAN,
    "quarantine": State.QUARANTINE,
    "skip-format": State.SKIPPED_FORMAT,
}

PROTECTED_TAGS = (
    "-EXIF:DateTimeOriginal",
    "-EXIF:CreateDate",
    "-EXIF:ModifyDate",
    "-GPS:all",
    "-IFD0:Orientation",
    "-ICC_Profile:all",
    "-IFD1:ThumbnailImage",
    "-MPF:all",
    # exiftool reports MPImageStart as an absolute file offset, which moves when the
    # XMP segment is inserted ahead of MPF. The stored (relative) value is checked
    # byte-for-byte by the structure gate.
    "--MPImageStart",
    "-XMP-GCamera:all",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def library(fixtures_dir: Path, tmp_path: Path) -> Path:
    """A messy little photo library: nested folders, plus things to ignore."""
    src = tmp_path / "Pictures"
    for i, fixture in enumerate(sorted(FIXTURE_EXPECT)):
        if fixture == "panorama.jpg":
            continue
        sub = src / ("2019" if i % 2 else "2020/summer")
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixtures_dir / fixture, sub / fixture)
    (src / "notes.txt").write_text("not a photo")
    (src / ".hidden.jpg").write_bytes((fixtures_dir / "plain.jpg").read_bytes())
    (src / ".thumbnails").mkdir()
    (src / ".thumbnails" / "t.jpg").write_bytes(b"x")
    os.symlink(fixtures_dir / "plain.jpg", src / "link.jpg")
    return src


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    config = Config()
    config.model.id = "stub"
    config.resources.work_dir = str(tmp_path / "ram")
    config.resources.allow_disk_workdir = True  # tmp_path is on disk; tested on purpose
    return config


def _run(src: Path, dest: Path, cfg: Config, tmp_path: Path, **kw: Any) -> RunSummary:
    log = TimingsLog(tmp_path / "timings.jsonl", run_id="r_test", model="stub")
    return caption_folder(src, dest, cfg, model=choose_model(cfg), run_id="r_test", log=log, **kw)


def _by_name(summary: RunSummary) -> dict[str, State]:
    return {Path(o.rel).name: o.state for o in summary.outcomes}


def test_m1_checkpoint(library: Path, cfg: Config, tmp_path: Path) -> None:
    originals = {p: (_sha(p), p.stat().st_mtime_ns) for p in library.rglob("*") if p.is_file()}
    dest = tmp_path / "Captioned"

    summary = _run(library, dest, cfg, tmp_path)

    assert summary.aborted is None
    states = _by_name(summary)
    for name, expect in FIXTURE_EXPECT.items():
        if name != "panorama.jpg":
            assert states[name] == EXPECTED_STATE[expect], (name, summary.outcomes)
    assert summary.ignored == 3  # notes.txt, .hidden.jpg, link.jpg
    assert "link.jpg" not in states and ".hidden.jpg" not in states

    # Originals: byte-identical, same mtime.
    for path, (sha, mtime) in originals.items():
        assert _sha(path) == sha, path
        assert path.stat().st_mtime_ns == mtime, path

    with ExifTool() as et:
        for outcome in summary.outcomes:
            src_path, out_path = library / outcome.rel, dest / outcome.rel
            if outcome.state != State.CLEAN:
                assert not out_path.exists(), outcome
                continue
            # The checkpoint: same image data, protected tags byte-identical.
            assert et.image_data_hash(out_path) == et.image_data_hash(src_path)
            before = et.read_tags(src_path, *PROTECTED_TAGS, binary=True)
            after = et.read_tags(out_path, *PROTECTED_TAGS, binary=True)
            assert before == after, outcome.rel
            # Captioned, attributed, and the copy keeps the original's mtime.
            tags = et.read_tags(out_path, "-XMP-dc:all", "-XMP-asif:all", "-XMP-et:all")
            subjects = tags.get("XMP-dc:Subject")
            subjects = subjects if isinstance(subjects, list) else [subjects]
            if outcome.rel.endswith("user_iptc.jpg"):
                # The owner's keywords are Latin-1 IPTC that can't be copied into
                # XMP safely; XMP keywords would hide them in most apps, so none.
                assert "XMP-dc:Subject" not in tags
                assert "keyword" in outcome.note
            else:
                assert "stub" in subjects
            assert tags["XMP-asif:RunId"] == "r_test"
            assert tags["XMP-asif:ModelId"] == "stub"
            assert tags["XMP-et:OriginalImageHashType"] == "SHA256"
            assert out_path.stat().st_mtime_ns == src_path.stat().st_mtime_ns
            if outcome.rel.endswith("user_xmp.jpg"):
                assert tags["XMP-dc:Description"] == "Grandma's garden, summer 2003"
                assert {"Garden", "family"} <= set(subjects)
            else:
                assert tags["XMP-dc:Description"]

    # No plaintext left behind in the work dir, no temp files in the output.
    assert not any(p.is_file() for p in (tmp_path / "ram").rglob("*"))
    assert not [p for p in dest.rglob(".*") if p.is_file()]


def test_rerun_skips_what_is_done_and_never_overwrites(
    library: Path, cfg: Config, tmp_path: Path
) -> None:
    dest = tmp_path / "out"
    first = _run(library, dest, cfg, tmp_path)
    done = {o.rel for o in first.outcomes if o.state == State.CLEAN}
    hashes = {rel: _sha(dest / rel) for rel in done}

    second = _run(library, dest, cfg, tmp_path)
    for outcome in second.outcomes:
        if outcome.rel in done:
            assert outcome.state == State.SKIPPED_DONE, outcome
    assert {rel: _sha(dest / rel) for rel in done} == hashes


def test_foreign_file_at_destination_is_a_conflict(
    library: Path, cfg: Config, tmp_path: Path
) -> None:
    dest = tmp_path / "out"
    rel = next(p.relative_to(library) for p in library.rglob("plain.png"))
    (dest / rel).parent.mkdir(parents=True)
    (dest / rel).write_bytes(b"someone else's file")
    summary = _run(library, dest, cfg, tmp_path)
    (outcome,) = [o for o in summary.outcomes if o.rel == rel.as_posix()]
    assert outcome.state == State.SKIPPED_CONFLICT
    assert (dest / rel).read_bytes() == b"someone else's file"


def test_dry_run_writes_nothing(library: Path, cfg: Config, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    summary = _run(library, dest, cfg, tmp_path, dry_run=True, limit=3)
    assert len(summary.outcomes) == 3
    assert not dest.exists()
    for outcome in summary.outcomes:
        if outcome.state == State.DESCRIBED:
            assert outcome.result is not None and "stub" in outcome.result.keywords


def test_original_changing_mid_run_is_a_conflict(
    library: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local, "fingerprint", lambda p: Fingerprint(0, 0, "changed"))
    dest = tmp_path / "out"
    summary = _run(library, dest, cfg, tmp_path)
    clean = [o for o in summary.outcomes if o.state == State.CLEAN]
    assert clean == []
    assert summary.count(State.SKIPPED_CONFLICT) >= 10
    assert not dest.exists() or not any(p.is_file() for p in dest.rglob("*"))


def test_failed_gates_never_write_and_trip_the_failure_budget(
    library: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*_a: Any, **_kw: Any) -> list[GateResult]:
        return [GateResult("equality", False, "hash differs")]

    monkeypatch.setattr(local, "check_written", failing)
    dest = tmp_path / "out"
    summary = _run(library, dest, cfg, tmp_path)
    assert summary.aborted is not None and "failure budget" in summary.aborted
    # 2% of 200 is 4: the fifth systematic failure stops the run.
    assert summary.count(State.FAILED_GATE) == 5
    assert not dest.exists() or not any(p.is_file() for p in dest.rglob("*"))


def test_schema_failures_are_retried_then_quarantined(
    library: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def bad_describe(self: Any, image: Path, img_id: str, attempt: int = 0) -> str:
        calls.append(attempt)
        return '{"caption": "no keywords"}'

    monkeypatch.setattr(local.InferenceClient, "describe", bad_describe)
    src = tmp_path / "one"
    src.mkdir()
    shutil.copy2(next(library.rglob("plain.jpg")), src / "plain.jpg")
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    (outcome,) = summary.outcomes
    assert outcome.state == State.QUARANTINE
    assert "schema" in outcome.reason
    assert calls == [0, 1, 2]


def test_timings_log_never_mentions_files_or_captions(
    library: Path, cfg: Config, tmp_path: Path
) -> None:
    summary = _run(library, tmp_path / "out", cfg, tmp_path)
    text = (tmp_path / "timings.jsonl").read_text()
    for outcome in summary.outcomes:
        assert Path(outcome.rel).stem not in text
    assert "Pictures" not in text and "summer" not in text
    with ExifTool() as et:
        for outcome in summary.outcomes:
            if outcome.state != State.CLEAN:
                continue
            tags = et.read_tags(tmp_path / "out" / outcome.rel, "-XMP-dc:Description")
            caption = str(tags["XMP-dc:Description"])
            assert caption and caption not in text
            assert caption.split(":")[0] not in text  # not even a fragment ("Stub caption")
    stages = {json.loads(line)["stage"] for line in text.splitlines()}
    assert {"run_start", "model_load", "open", "infer", "write", "gates", "commit"} <= stages


def test_discover_ignores_hidden_files_links_and_special_files(library: Path) -> None:
    os.mkfifo(library / "pipe.jpg")  # reading it would block forever
    found = discover(library)
    names = {p.name for p in found.files}
    assert "link.jpg" not in names and ".hidden.jpg" not in names and "pipe.jpg" not in names
    assert "photo.heic" in names and "raw.dng" in names
    # notes.txt, .hidden.jpg, link.jpg, pipe.jpg (.thumbnails is never entered)
    assert found.ignored == 4
    assert found.unreadable_dirs == []


def test_folder_checks(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(CaptionError, match="not found"):
        check_folders(tmp_path / "missing", tmp_path / "out")
    with pytest.raises(CaptionError, match="separate"):
        check_folders(src, src)
    with pytest.raises(CaptionError, match="inside the source"):
        check_folders(src, src / "out")
    with pytest.raises(CaptionError, match="inside the destination"):
        check_folders(src, tmp_path)
    (tmp_path / "file").write_text("x")
    with pytest.raises(CaptionError, match="isn't a folder"):
        check_folders(src, tmp_path / "file")


def test_choose_model_explains_what_is_missing(tmp_path: Path) -> None:
    cfg = Config()
    with pytest.raises(CaptionError, match="--model stub"):
        choose_model(cfg)
    cfg.model.gguf = str(tmp_path / "m.gguf")
    cfg.model.mmproj = str(tmp_path / "p.gguf")
    with pytest.raises(CaptionError, match="not found"):
        choose_model(cfg)
    (tmp_path / "m.gguf").write_bytes(b"weights")
    (tmp_path / "p.gguf").write_bytes(b"projector")
    choice = choose_model(cfg)
    assert choice.backend == "llama"
    assert choice.model_id == "local:m.gguf"
    assert choice.model_sha256 == hashlib.sha256(b"weights").hexdigest()


def test_device_cpu_means_no_gpu_layers(tmp_path: Path) -> None:
    cfg = Config()
    (tmp_path / "m.gguf").write_bytes(b"w")
    (tmp_path / "p.gguf").write_bytes(b"p")
    cfg.model.gguf, cfg.model.mmproj, cfg.model.device = (
        str(tmp_path / "m.gguf"),
        str(tmp_path / "p.gguf"),
        "cpu",
    )
    assert choose_model(cfg).worker.gpu_layers == 0
