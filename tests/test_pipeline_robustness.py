"""What happens when one photo, the destination or the model misbehaves.

A problem with one photo becomes one line in the summary and the run goes
on. A problem that would hit every photo stops the run early, and the
summary of what was already done is still returned. Either way, nothing is
left behind in the RAM work dir.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from asif.config import Config
from asif.infer.client import InferenceError
from asif.pipeline import local
from asif.pipeline.local import CaptionError, RunSummary, caption_folder, choose_model
from asif.pipeline.states import State
from asif.pipeline.timings import TimingsLog
from tests.conftest import requires_exiftool

pytestmark = requires_exiftool


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    config = Config()
    config.model.id = "stub"
    config.resources.work_dir = str(tmp_path / "ram")
    config.resources.allow_disk_workdir = True
    return config


@pytest.fixture
def photos(fixtures_dir: Path, tmp_path: Path) -> Path:
    """Five ordinary photos."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        shutil.copy2(fixtures_dir / "plain.jpg", src / f"p{i}.jpg")
    return src


def _run(src: Path, dest: Path, cfg: Config, tmp_path: Path, **kw: Any) -> RunSummary:
    log = TimingsLog(tmp_path / "timings.jsonl", run_id="r_test", model="stub")
    return caption_folder(src, dest, cfg, model=choose_model(cfg), run_id="r_test", log=log, **kw)


def _states(summary: RunSummary) -> dict[str, State]:
    return {o.rel: o.state for o in summary.outcomes}


def _ram_is_empty(tmp_path: Path) -> bool:
    return not any(p.is_file() for p in (tmp_path / "ram").rglob("*"))


# -- the source --------------------------------------------------------------


def test_a_photo_deleted_before_its_turn_is_skipped_not_fatal(
    photos: Path, cfg: Config, tmp_path: Path
) -> None:
    victim: list[Path] = []

    def delete_one_unprocessed(outcome: Any, index: int, total: int) -> None:
        if index == 1:
            done = {photos / outcome.rel}
            victim.append(next(p for p in sorted(photos.iterdir()) if p not in done))
            victim[0].unlink()

    summary = _run(photos, tmp_path / "out", cfg, tmp_path, on_progress=delete_one_unprocessed)
    states = _states(summary)
    assert summary.aborted is None
    assert states.pop(victim[0].name) == State.SKIPPED_CONFLICT
    assert set(states.values()) == {State.CLEAN}
    assert len(states) == 4
    assert _ram_is_empty(tmp_path)


def test_a_photo_that_vanishes_mid_way_is_skipped(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = local._describe

    def describe_then_move(ctx: Any, ram: Path, img: str) -> Any:
        result = real(ctx, ram, img)
        target = photos / "p2.jpg"
        if target.exists() and img == _img_of(ctx, "p2.jpg"):
            target.rename(photos.parent / "moved.jpg")
        return result

    order: dict[str, str] = {}

    def _img_of(ctx: Any, name: str) -> str:
        return order.get(name, "")

    real_process = local._process

    def remember(ctx: Any, src: Path, dest: Path, rel: str, img: str) -> Any:
        order[rel] = img
        return real_process(ctx, src, dest, rel, img)

    monkeypatch.setattr(local, "_describe", describe_then_move)
    monkeypatch.setattr(local, "_process", remember)
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    outcome = next(o for o in summary.outcomes if o.rel == "p2.jpg")
    assert outcome.state == State.SKIPPED_CONFLICT
    assert "disappeared" in outcome.reason
    assert not (tmp_path / "out" / "p2.jpg").exists()
    assert summary.count(State.CLEAN) == 4


def test_an_unreadable_photo_is_quarantined_and_leaks_nothing(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Root can read mode-000 files, so simulate the open failing.
    real_open = os.open

    def deny(path: Any, flags: int, *args: Any, **kw: Any) -> int:
        if str(path).endswith("p1.jpg"):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(path, flags, *args, **kw)

    monkeypatch.setattr(local.os, "open", deny)
    fds_before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else 0
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    outcome = next(o for o in summary.outcomes if o.rel == "p1.jpg")
    assert outcome.state == State.QUARANTINE
    assert "can't read the original" in outcome.reason
    assert not outcome.systematic
    assert summary.count(State.CLEAN) == 4
    if fds_before:
        assert len(os.listdir("/proc/self/fd")) <= fds_before + 1
    assert _ram_is_empty(tmp_path)


@pytest.mark.parametrize(("fixture", "named", "says"), [("plain.png", "x.jpg", "PNG")])
def test_a_wrong_extension_is_a_bad_input_not_a_bug(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, fixture: str, named: str, says: str
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(8):  # more than the failure budget would tolerate if systematic
        shutil.copy2(fixtures_dir / fixture, src / f"{i}{named}")
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    assert summary.aborted is None
    assert summary.count(State.QUARANTINE) == 8
    assert all(not o.systematic and says in o.reason for o in summary.outcomes)


def test_a_folder_that_cant_be_listed_is_reported(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (photos / "locked").mkdir()
    real_walk = os.walk

    def walk(top: Any, *args: Any, onerror: Any = None, **kw: Any) -> Any:
        for root, dirs, files in real_walk(top, *args, onerror=onerror, **kw):
            if "locked" in dirs:
                dirs.remove("locked")
                onerror(PermissionError(errno.EACCES, "Permission denied", str(photos / "locked")))
            yield root, dirs, files

    monkeypatch.setattr(local.os, "walk", walk)
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    assert summary.unreadable_dirs == ["locked"]
    assert summary.count(State.CLEAN) == 5


def test_a_very_long_file_name_works(fixtures_dir: Path, cfg: Config, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    name = "a" * 250 + ".jpg"  # legal (255 bytes max), but no room for a prefix/suffix
    shutil.copy2(fixtures_dir / "plain.jpg", src / name)
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    assert _states(summary) == {name: State.CLEAN}


def test_non_utf8_names_survive_a_rerun(fixtures_dir: Path, cfg: Config, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    name = os.fsdecode(b"caf\xe9.jpg")  # Latin-1 bytes, not valid UTF-8
    shutil.copy2(fixtures_dir / "plain.jpg", src / name)
    assert _states(_run(src, tmp_path / "out", cfg, tmp_path)) == {name: State.CLEAN}
    assert _states(_run(src, tmp_path / "out", cfg, tmp_path)) == {name: State.SKIPPED_DONE}


def test_a_line_break_in_a_name_doesnt_break_reruns(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    name = "two\nlines.jpg"
    shutil.copy2(fixtures_dir / "plain.jpg", src / name)
    assert _states(_run(src, tmp_path / "out", cfg, tmp_path)) == {name: State.CLEAN}
    # exiftool can't be handed this name to check the copy: say so, don't crash.
    rerun = _run(src, tmp_path / "out", cfg, tmp_path)
    assert _states(rerun) == {name: State.SKIPPED_CONFLICT}


# -- the destination ---------------------------------------------------------


def test_a_full_destination_stops_the_run_and_keeps_the_summary(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_commit = local._commit
    calls = {"n": 0}

    def fill_up(ram: Path, dest: Path, mode: int, **kw: Any) -> bool:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_commit(ram, dest, mode, **kw)

    monkeypatch.setattr(local, "_commit", fill_up)
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    assert summary.aborted is not None and "No space left" in summary.aborted
    assert summary.count(State.CLEAN) == 2
    assert summary.outcomes[-1].state == State.FAILED_UPLOAD
    assert len(summary.outcomes) == 3  # nothing after the failure was attempted
    assert _ram_is_empty(tmp_path)


def test_a_per_path_destination_problem_only_affects_that_photo(
    photos: Path, cfg: Config, tmp_path: Path
) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (photos / "sub").mkdir()
    shutil.copy2(photos / "p0.jpg", photos / "sub" / "x.jpg")
    (dest / "sub").write_text("a file where a folder should be")
    summary = _run(photos, dest, cfg, tmp_path)
    outcome = next(o for o in summary.outcomes if o.rel == "sub/x.jpg")
    assert outcome.state == State.FAILED_UPLOAD
    assert summary.aborted is None
    assert summary.count(State.CLEAN) == 5


def test_an_unusable_destination_is_refused_before_the_model_loads(
    photos: Path, cfg: Config, tmp_path: Path
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(CaptionError, match="destination"):
        _run(photos, blocker / "out", cfg, tmp_path)


def test_copies_of_private_photos_stay_private(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    private = src / "medical"
    private.mkdir(parents=True)
    photo = private / "scan.jpg"
    shutil.copy2(fixtures_dir / "plain.jpg", photo)
    photo.chmod(0o644)
    private.chmod(0o700)
    dest = tmp_path / "out"
    _run(src, dest, cfg, tmp_path)
    copy = dest / "medical" / "scan.jpg"
    assert stat.S_IMODE(copy.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE((dest / "medical").stat().st_mode) == 0o700
    assert stat.S_IMODE(dest.stat().st_mode) == 0o700  # the new destination root too


# -- the model ---------------------------------------------------------------


def test_a_worker_that_crashes_and_cant_restart_stops_the_run_cleanly(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"describe": 0, "start": 0}
    real_start = local.InferenceClient.start

    def describe(self: Any, image: Path, img_id: str, attempt: int = 0) -> str:
        calls["describe"] += 1
        if calls["describe"] == 2:
            self.close()
            raise InferenceError("dead", "worker crashed")
        return (
            '{"caption": "A test.", "keywords": ["a", "b", "c"], "has_text": false, '
            '"people_count": 0, "setting": "unclear"}'
        )

    def start(self: Any) -> Any:
        calls["start"] += 1
        if calls["start"] > 1:
            raise InferenceError("start", "model file vanished")
        return real_start(self)

    monkeypatch.setattr(local.InferenceClient, "describe", describe)
    monkeypatch.setattr(local.InferenceClient, "start", start)
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    assert summary.aborted is not None and "couldn't be restarted" in summary.aborted
    assert [o.state for o in summary.outcomes] == [State.CLEAN, State.QUARANTINE]
    assert _ram_is_empty(tmp_path)


def test_ctrl_c_mid_run_wipes_the_ram_copy_and_returns_a_summary(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    real = local._describe

    def interrupt_second(ctx: Any, ram: Path, img: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            assert ram.exists()  # the plaintext copy is there right now...
            raise KeyboardInterrupt
        return real(ctx, ram, img)

    monkeypatch.setattr(local, "_describe", interrupt_second)
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    assert summary.interrupted
    assert [o.state for o in summary.outcomes] == [State.CLEAN]
    assert _ram_is_empty(tmp_path)  # ...and gone afterwards
    assert not [p for p in (tmp_path / "out").rglob(".asif-*")]


def test_a_missing_exiftool_is_a_clear_start_error(
    photos: Path, cfg: Config, tmp_path: Path
) -> None:
    log = TimingsLog(None, run_id="r", model="stub")
    with pytest.raises(CaptionError, match="exiftool not found"):
        caption_folder(
            photos,
            tmp_path / "out",
            cfg,
            model=choose_model(cfg),
            run_id="r",
            log=log,
            exiftool=str(tmp_path / "no-such-exiftool"),
        )


# -- what the run reports ---------------------------------------------------


def test_dry_run_shows_what_would_really_be_written(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "user_xmp.jpg", src / "owner.jpg")
    summary = _run(src, tmp_path / "out", cfg, tmp_path, dry_run=True)
    (outcome,) = summary.outcomes
    assert outcome.plan is not None
    assert outcome.plan.caption is None  # the owner's caption is kept
    assert outcome.reason == "kept-user"


def test_compat_wide_says_when_iptc_was_skipped(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    cfg.general.compat = "wide"
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.png", src / "a.png")
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    (outcome,) = summary.outcomes
    assert outcome.state == State.CLEAN
    assert outcome.note  # e.g. "IPTC isn't written to PNG"


def test_the_timings_log_holds_no_sizes_and_no_order(
    photos: Path, cfg: Config, tmp_path: Path
) -> None:
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    text = (tmp_path / "timings.jsonl").read_text()
    assert '"bytes"' not in text
    for outcome in summary.outcomes:
        assert outcome.rel not in text


def test_unknown_model_ids_are_rejected(cfg: Config) -> None:
    cfg.model.id = "stbu"
    with pytest.raises(CaptionError, match="unknown model id"):
        choose_model(cfg)


@pytest.mark.parametrize(
    ("device", "layers", "extra"),
    [("cpu", 0, ()), ("gpu", 999, ()), ("hybrid", 999, ("--cpu-moe",)), ("auto", None, ())],
)
def test_device_decides_placement(
    tmp_path: Path, device: str, layers: int | None, extra: tuple[str, ...]
) -> None:
    cfg = Config()
    (tmp_path / "m.gguf").write_bytes(b"w")
    (tmp_path / "p.gguf").write_bytes(b"p")
    cfg.model.gguf, cfg.model.mmproj = str(tmp_path / "m.gguf"), str(tmp_path / "p.gguf")
    cfg.model.device = device  # type: ignore[assignment]
    cfg.model.server_args = ["--flash-attn"]
    worker = choose_model(cfg).worker
    assert worker.gpu_layers == layers
    assert worker.server_args == (*extra, "--flash-attn")


def test_an_unpinned_catalogue_id_is_recorded_as_a_local_file(tmp_path: Path) -> None:
    cfg = Config()
    (tmp_path / "m.gguf").write_bytes(b"w")
    (tmp_path / "p.gguf").write_bytes(b"p")
    cfg.model.gguf, cfg.model.mmproj = str(tmp_path / "m.gguf"), str(tmp_path / "p.gguf")
    cfg.model.id = "qwen3.5-9b-q4km"
    notes: list[str] = []
    assert choose_model(cfg, notice=notes.append).model_id == "local:m.gguf"
    assert notes and "pinned" in notes[-1]


# -- the commit itself --------------------------------------------------------


def test_a_copy_that_reads_back_differently_is_removed(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_sha = local.sha256_file
    dest = tmp_path / "out"

    def flaky_disk(path: Path) -> str:
        digest = real_sha(path)
        return "0" * 64 if dest in path.parents else digest

    monkeypatch.setattr(local, "sha256_file", flaky_disk)
    summary = _run(photos, dest, cfg, tmp_path)
    assert all(o.state == State.FAILED_UPLOAD for o in summary.outcomes)
    assert not any(p.is_file() for p in dest.rglob("*"))
    assert summary.aborted is not None  # five in a row is past the failure budget


def test_commit_works_without_hard_links_and_still_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ram = tmp_path / "ram.jpg"
    ram.write_bytes(b"processed")

    def no_links(src: Any, dst: Any) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")  # FAT/exFAT

    monkeypatch.setattr(local.os, "link", no_links)
    dest = tmp_path / "out" / "a.jpg"
    assert local._commit(ram, dest, 0o644)
    assert dest.read_bytes() == b"processed"
    assert not local._commit(ram, dest, 0o644)  # something is there now: refuse
    assert not [p for p in dest.parent.iterdir() if p.name.startswith(".asif-")]


def test_commit_never_replaces_an_existing_file(tmp_path: Path) -> None:
    ram = tmp_path / "ram.jpg"
    ram.write_bytes(b"processed")
    dest = tmp_path / "out" / "a.jpg"
    dest.parent.mkdir()
    dest.write_bytes(b"someone else's")
    assert not local._commit(ram, dest, 0o644)
    assert dest.read_bytes() == b"someone else's"


# -- files the gates can't vouch for, and copies that went stale ------------


def test_a_copy_of_an_older_original_is_not_mistaken_for_done(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    import subprocess

    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "exif.jpg", src / "a.jpg")
    assert _states(_run(src, tmp_path / "out", cfg, tmp_path)) == {"a.jpg": State.CLEAN}
    # The owner fixes the date in the original; the pixels don't change.
    subprocess.run(
        [
            "exiftool",
            "-q",
            "-overwrite_original",
            "-DateTimeOriginal=2019:07:15 10:20:30",
            str(src / "a.jpg"),
        ],
        check=True,
    )
    rerun = _run(src, tmp_path / "out", cfg, tmp_path)
    (outcome,) = rerun.outcomes
    assert outcome.state == State.SKIPPED_CONFLICT
    assert "changed after its copy was made" in outcome.reason


def test_multi_page_tiffs_are_captioned(cfg: Config, tmp_path: Path) -> None:
    from tests.test_structure import pillow_tiff

    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):  # past the failure budget, if they were refused as bugs
        (src / f"scan{i}.tif").write_bytes(pillow_tiff(pages=3, compression="tiff_lzw"))
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    assert summary.aborted is None
    assert set(_states(summary).values()) == {State.CLEAN}


def test_tiffs_with_maker_notes_are_skipped_as_a_format_limit(cfg: Config, tmp_path: Path) -> None:
    from tests.test_gates import _canon_exif

    src = tmp_path / "src"
    src.mkdir()
    tiff, _note = _canon_exif(image=True)
    for i in range(6):
        (src / f"raw{i}.tif").write_bytes(tiff)
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    assert summary.aborted is None
    assert set(_states(summary).values()) == {State.SKIPPED_FORMAT}
    assert all("maker notes" in o.reason and not o.systematic for o in summary.outcomes)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").rglob("*.tif"))


# -- round 2 -----------------------------------------------------------------


@pytest.mark.parametrize("no_link_errno", [errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP])
def test_commit_falls_back_wherever_hard_links_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_link_errno: int
) -> None:
    ram = tmp_path / "ram.jpg"
    ram.write_bytes(b"processed")

    def no_links(src: Any, dst: Any) -> None:
        raise OSError(no_link_errno, os.strerror(no_link_errno))  # e.g. FUSE: rclone, sshfs

    monkeypatch.setattr(local.os, "link", no_links)
    dest = tmp_path / "out" / "a.jpg"
    assert local._commit(ram, dest, 0o644)
    assert dest.read_bytes() == b"processed"


def test_ctrl_c_during_the_read_back_leaves_no_unconfirmed_copy(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out"
    real_sha = local.sha256_file

    def interrupted_read(path: Path) -> str:
        if dest in path.parents:
            raise KeyboardInterrupt
        return real_sha(path)

    monkeypatch.setattr(local, "sha256_file", interrupted_read)
    summary = _run(photos, dest, cfg, tmp_path)
    assert summary.interrupted
    assert not any(p.is_file() for p in dest.rglob("*"))  # placed, never confirmed: removed


class _KeepOrder:
    """Stands in for the shuffle so a test can know the processing order."""

    def shuffle(self, items: list[Path]) -> None:
        items.sort()


def test_an_unsearchable_destination_subfolder_is_handled(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "out"
    real_exists = Path.exists

    def exists(self: Path, *a: Any, **kw: Any) -> bool:
        if self.name == "p2.jpg" and dest in self.parents:
            raise OSError(errno.ENAMETOOLONG, "File name too long")
        if self.name == "p3.jpg" and dest in self.parents:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_exists(self, *a, **kw)

    monkeypatch.setattr(local.secrets, "SystemRandom", _KeepOrder)
    monkeypatch.setattr(Path, "exists", exists)
    summary = _run(photos, dest, cfg, tmp_path)
    states = _states(summary)
    assert states["p2.jpg"] == State.FAILED_UPLOAD  # only this photo's path
    # Permission denied is that folder's answer: later photos bound for it
    # are skipped before the model runs, and the run goes on.
    assert summary.aborted is None
    assert states["p3.jpg"] == states["p4.jpg"] == State.FAILED_UPLOAD
    reasons = {o.rel: o.reason for o in summary.outcomes}
    assert reasons["p3.jpg"] == reasons["p4.jpg"] == "can't write into the destination folder"
    assert states["p0.jpg"] == states["p1.jpg"] == State.CLEAN


def test_a_different_photo_in_the_way_is_not_called_stale(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / "a.jpg")
    dest = tmp_path / "out"
    _run(src, dest, cfg, tmp_path)
    # Now a different photo wants the same name (think case-insensitive disks).
    (src / "a.jpg").unlink()
    shutil.copy2(fixtures_dir / "gps.jpg", src / "a.jpg")
    (outcome,) = _run(src, dest, cfg, tmp_path).outcomes
    assert outcome.state == State.SKIPPED_CONFLICT
    assert "different file" in outcome.reason


def test_files_exiftool_cant_rewrite_are_skipped_not_counted_as_bugs(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asif.meta.exiftool import ExifToolError

    def refuse(*_a: Any, **_kw: Any) -> Any:
        raise ExifToolError(
            "exiftool warned about the write (1 warning(s))",
            "Warning: [Minor] Duplicate XMP property: photoshop:City\n",
        )

    monkeypatch.setattr(local, "apply_write", refuse)
    for i in range(5, 8):  # more than the failure budget would allow if systematic
        shutil.copy2(photos / "p0.jpg", photos / f"p{i}.jpg")
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    assert summary.aborted is None
    assert set(_states(summary).values()) == {State.SKIPPED_FORMAT}
    assert not any(o.systematic for o in summary.outcomes)


def test_processing_order_comes_from_the_shuffle(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[Path]] = []

    class Reverse:
        def shuffle(self, items: list[Path]) -> None:
            seen.append(list(items))
            items.reverse()

    monkeypatch.setattr(local.secrets, "SystemRandom", Reverse)
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    assert len(seen) == 1 and len(seen[0]) == 5
    assert [o.rel for o in summary.outcomes] == [p.name for p in reversed(seen[0])]


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs a world-searchable dir")
def test_acl_group_bits_are_never_passed_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    # Under /dev/shm every parent is searchable, so only the ACL can narrow it.
    folder = Path(tempfile.mkdtemp(dir="/dev/shm"))
    try:
        photo = folder / "a.jpg"
        photo.write_bytes(b"x")
        photo.chmod(0o664)
        folder.chmod(0o775)
        monkeypatch.setattr(local, "_has_acl", lambda p: False)
        assert local._reach_mask(photo) & 0o070 == 0o070  # no ACL: group bits stand
        # With an ACL the group bits are only its mask, and the copy gets no ACL.
        monkeypatch.setattr(local, "_has_acl", lambda p: p == folder)
        assert local._reach_mask(photo) & 0o070 == 0
    finally:
        shutil.rmtree(folder)


def test_has_acl_is_false_for_a_plain_file(tmp_path: Path) -> None:
    plain = tmp_path / "a.jpg"
    plain.write_bytes(b"x")
    assert not local._has_acl(plain)


def test_a_late_signal_during_cleanup_keeps_the_summary(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_close = local.InferenceClient.close
    fired = {"done": False}

    def close_then_signal(self: Any) -> None:
        real_close(self)
        if not fired["done"]:
            fired["done"] = True
            raise KeyboardInterrupt  # as if Ctrl+C landed while the model unloads

    monkeypatch.setattr(local.InferenceClient, "close", close_then_signal)
    summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    assert summary.count(State.CLEAN) == 5  # every photo was done...
    assert not summary.interrupted  # ...so a late signal changes nothing
    assert _ram_is_empty(tmp_path)


def test_files_the_gates_cant_vouch_for_are_skipped_before_the_model(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    from tests.test_gates import _document_ancestors, _with_c2pa

    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):  # more than the failure budget would allow if these counted as bugs
        shutil.copy2(fixtures_dir / "plain.jpg", src / f"bloat{i}.jpg")
        _document_ancestors(src / f"bloat{i}.jpg", 1001)
    shutil.copy2(fixtures_dir / "plain.jpg", src / "signed.jpg")
    _with_c2pa("jpeg", src / "signed.jpg")
    Image.new("RGB", (64, 48), (10, 120, 200)).save(src / "scan.tif", compression="jpeg")
    shutil.copy2(fixtures_dir / "plain.jpg", src / "ordinary.jpg")

    described: list[str] = []
    real = local._describe
    monkeypatch.setattr(
        local, "_describe", lambda ctx, ram, img: described.append(img) or real(ctx, ram, img)
    )
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    states = _states(summary)
    assert summary.aborted is None
    assert states.pop("ordinary.jpg") == State.CLEAN
    assert set(states.values()) == {State.SKIPPED_FORMAT}
    reasons = {o.rel: o.reason for o in summary.outcomes}
    assert "Content Credentials" in reasons["signed.jpg"]
    assert "JPEG-compressed TIFF" in reasons["scan.tif"]
    assert len(described) == 1  # the model only ever saw the ordinary photo


# -- round 3 -----------------------------------------------------------------


def _other_gid() -> int:
    gids = [g for g in range(1, 70000) if g != os.getgid()]
    return next(g for g in gids if _gid_exists(g))


def _gid_exists(gid: int) -> bool:
    import grp

    try:
        grp.getgrgid(gid)
    except KeyError:
        return False
    return True


@pytest.mark.skipif(os.geteuid() != 0, reason="needs to chown a folder to another group")
def test_a_setgid_destination_never_gives_its_group_the_copy(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / "a.jpg")
    (src / "a.jpg").chmod(0o640)
    dest = tmp_path / "shared"
    dest.mkdir()
    os.chown(dest, -1, _other_gid())
    dest.chmod(0o2775)  # setgid: new files would take the folder's group
    _run(src, dest, cfg, tmp_path)
    copy = (dest / "a.jpg").stat()
    original = (src / "a.jpg").stat()
    assert copy.st_gid == original.st_gid or stat.S_IMODE(copy.st_mode) & 0o070 == 0


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="POSIX ACLs are Linux-only")
def test_a_default_acl_on_the_destination_is_not_inherited(tmp_path: Path) -> None:
    import struct

    dest_dir = tmp_path / "out"
    dest_dir.mkdir()
    # user::rwx, user:65534:r-x, group::r-x, mask::r-x, other::---
    entries = [
        (0x01, 7, 0xFFFFFFFF),
        (0x02, 5, 65534),
        (0x04, 5, 0xFFFFFFFF),
        (0x10, 5, 0xFFFFFFFF),
        (0x20, 0, 0xFFFFFFFF),
    ]
    acl = struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *e) for e in entries)
    try:
        os.setxattr(dest_dir, "system.posix_acl_default", acl)
    except OSError as exc:
        pytest.skip(f"no ACL support here ({exc.strerror})")
    ram = tmp_path / "ram.jpg"
    ram.write_bytes(b"processed")
    assert local._commit(ram, dest_dir / "a.jpg", 0o640)
    assert not local._has_acl(dest_dir / "a.jpg")


def test_a_stop_while_the_copy_is_placed_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ram = tmp_path / "ram.jpg"
    ram.write_bytes(b"processed")
    real_link = os.link

    def link_then_interrupt(src: Any, dst: Any) -> None:
        real_link(src, dst)
        raise KeyboardInterrupt

    monkeypatch.setattr(local.os, "link", link_then_interrupt)
    dest = tmp_path / "out" / "a.jpg"
    with pytest.raises(KeyboardInterrupt):
        local._commit(ram, dest, 0o644, expect_sha="0" * 64)
    assert not dest.exists()
    assert not dest.parent.exists()  # nor the folder made for it


def test_one_unwritable_subfolder_doesnt_stop_the_run(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("x", "y"):
        (photos / "theirs").mkdir(exist_ok=True)
        shutil.copy2(photos / "p0.jpg", photos / "theirs" / f"{name}.jpg")
    dest = tmp_path / "out"
    (dest / "theirs").mkdir(parents=True, mode=0o700)  # someone else's, say
    real_commit = local._commit

    def denied_in_theirs(ram: Path, target: Path, mode: int, **kw: Any) -> bool:
        if target.parent.name == "theirs":
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_commit(ram, target, mode, **kw)

    monkeypatch.setattr(local, "_commit", denied_in_theirs)
    summary = _run(photos, dest, cfg, tmp_path)
    states = _states(summary)
    assert summary.aborted is None
    assert states["theirs/x.jpg"] == states["theirs/y.jpg"] == State.FAILED_UPLOAD
    assert summary.count(State.CLEAN) == 5


def test_a_listable_destination_for_a_private_source_is_refused(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    import tempfile

    if not os.path.isdir("/dev/shm"):
        pytest.skip("needs a world-searchable dir")
    open_root = Path(tempfile.mkdtemp(dir="/dev/shm"))
    try:
        open_root.chmod(0o755)
        src = open_root / "private"
        src.mkdir(mode=0o700)
        shutil.copy2(fixtures_dir / "plain.jpg", src / "a.jpg")
        dest = open_root / "public"
        dest.mkdir(mode=0o755)
        dest.chmod(0o755)
        with pytest.raises(CaptionError, match="chmod 700"):
            _run(src, dest, cfg, tmp_path)
        assert not list(dest.iterdir())
    finally:
        shutil.rmtree(open_root)


def test_cleanup_finishes_even_if_a_stop_signal_lands_during_it(
    photos: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import signal

    from asif import proc, workdir

    saved = {s: signal.getsignal(s) for s in proc._STOP_SIGNALS}
    proc.install_stop_signals()
    try:
        real_wipe = workdir.wipe_file
        sent = {"done": False}

        def wipe_with_a_signal(path: Path) -> None:
            if not sent["done"]:
                sent["done"] = True
                os.kill(os.getpid(), signal.SIGTERM)  # lands mid-cleanup
            real_wipe(path)

        monkeypatch.setattr(workdir, "wipe_file", wipe_with_a_signal)
        summary = _run(photos, tmp_path / "out", cfg, tmp_path)
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    assert sent["done"]
    assert summary.count(State.CLEAN) == 5
    assert not summary.interrupted  # every photo was done before the signal
    assert not any((tmp_path / "ram").iterdir())  # the work dir was wiped completely


def test_a_listable_folder_above_the_copy_is_refused_too(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    import tempfile

    if not os.path.isdir("/dev/shm"):
        pytest.skip("needs a world-searchable dir")
    open_root = Path(tempfile.mkdtemp(dir="/dev/shm"))
    try:
        open_root.chmod(0o755)
        src = open_root / "src"
        (src / "health" / "2024").mkdir(parents=True)
        src.chmod(0o755)
        (src / "health").chmod(0o700)
        shutil.copy2(fixtures_dir / "plain.jpg", src / "health" / "2024" / "a.jpg")
        shutil.copy2(fixtures_dir / "plain.jpg", src / "b.jpg")
        dest = open_root / "dest"
        (dest / "health").mkdir(parents=True)
        dest.chmod(0o755)
        (dest / "health").chmod(0o755)  # "2024" would be listed here
        summary = _run(src, dest, cfg, tmp_path)
        states = _states(summary)
        assert states["health/2024/a.jpg"] == State.FAILED_UPLOAD
        assert states["b.jpg"] == State.CLEAN
        assert not list((dest / "health").iterdir())
    finally:
        shutil.rmtree(open_root)


# -- round 4 -----------------------------------------------------------------


def _a_second_gid() -> int | None:
    """A group other than our own we can give files to (root: any), or None."""
    if os.geteuid() == 0:
        return _other_gid()
    others = [g for g in os.getgroups() if g != os.getegid()]
    return others[0] if others else None


@pytest.fixture
def open_dir() -> Any:
    """A folder everyone can search into (under /dev/shm), like a real home's parent."""
    import tempfile

    if not os.path.isdir("/dev/shm"):
        pytest.skip("needs a world-searchable dir")
    root = Path(tempfile.mkdtemp(dir="/dev/shm"))
    root.chmod(0o755)
    yield root
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
    shutil.rmtree(root)


def _folder(path: Path, mode: int, gid: int | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if gid is not None:
        os.chown(path, -1, gid)
    path.chmod(mode)
    return path


def test_a_group_shared_destination_counts_its_group_not_just_its_bits(open_dir: Path) -> None:
    other = _a_second_gid()
    if other is None:
        pytest.skip("needs a second group")
    me = os.getegid()
    home = _folder(open_dir / "home", 0o750, me)
    src = _folder(home / "Pictures", 0o755)
    family = _folder(open_dir / "family", 0o2770, other)
    dest = _folder(family / "alice-captioned", 0o755, other)
    # Both are "group-listable", but by different groups.
    assert local._more_people_can_list(dest, src)
    # The everyday case still works: a 0755 folder next to the source.
    assert not local._more_people_can_list(_folder(home / "captioned", 0o755), src)


def test_others_one_level_down_are_counted(open_dir: Path) -> None:
    other = _a_second_gid()
    if other is None:
        pytest.skip("needs a second group")
    # 0705 inside a group-only folder: that group's members are "others" here.
    parent = _folder(open_dir / "shared", 0o750, other)
    dest = _folder(parent / "out", 0o705)
    src = _folder(open_dir / "private", 0o700)
    assert local._more_people_can_list(dest, src)


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="POSIX ACLs are Linux-only")
def test_a_named_user_acl_on_the_destination_counts(open_dir: Path) -> None:
    import struct

    src = _folder(open_dir / "private", 0o700)
    dest = _folder(open_dir / "out", 0o700)
    assert not local._more_people_can_list(dest, src)
    # user::rwx, user:65534:r-x, group::---, mask::r-x, other::---
    entries = [
        (0x01, 7, 0xFFFFFFFF),
        (0x02, 5, 65534),
        (0x04, 0, 0xFFFFFFFF),
        (0x10, 5, 0xFFFFFFFF),
        (0x20, 0, 0xFFFFFFFF),
    ]
    acl = struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *e) for e in entries)
    try:
        os.setxattr(dest, "system.posix_acl_access", acl)
    except OSError as exc:
        pytest.skip(f"no ACL support here ({exc.strerror})")
    assert local._more_people_can_list(dest, src)


@pytest.mark.skipif(not hasattr(os, "setxattr"), reason="POSIX ACLs are Linux-only")
def test_a_folder_only_we_can_enter_through_an_acl_is_still_private(open_dir: Path) -> None:
    import struct

    # Like /media/<user> on Ubuntu, where USB drives are mounted: 0750, and
    # an ACL lets just that user in.
    src = _folder(open_dir / "private", 0o700)
    media = _folder(open_dir / "media", 0o750)
    dest = _folder(media / "usb" / "captioned", 0o755)
    entries = [
        (0x01, 7, 0xFFFFFFFF),
        (0x02, 5, os.getuid()),
        (0x04, 0, 0xFFFFFFFF),
        (0x10, 5, 0xFFFFFFFF),
        (0x20, 0, 0xFFFFFFFF),
    ]
    acl = struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *e) for e in entries)
    try:
        os.setxattr(media, "system.posix_acl_access", acl)
    except OSError as exc:
        pytest.skip(f"no ACL support here ({exc.strerror})")
    assert not local._more_people_can_list(dest, src)


@pytest.mark.skipif(os.geteuid() != 0, reason="needs a folder owned by someone else")
def test_someone_elses_folder_counts_as_listable(open_dir: Path) -> None:
    src = _folder(open_dir / "private", 0o700)
    dest = _folder(open_dir / "theirs", 0o700)
    os.chown(dest, 65534, -1)
    assert local._more_people_can_list(dest, src)


def test_a_symlinked_destination_folder_is_judged_where_it_points(
    open_dir: Path, fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = _folder(open_dir / "private", 0o700)
    _folder(src / "2024", 0o755)
    shutil.copy2(fixtures_dir / "plain.jpg", src / "2024" / "a.jpg")
    public = _folder(open_dir / "public" / "2024", 0o755)
    dest = _folder(open_dir / "out", 0o700)
    (dest / "2024").symlink_to(public)
    summary = _run(src, dest, cfg, tmp_path)
    [outcome] = summary.outcomes
    assert outcome.state == State.FAILED_UPLOAD
    assert outcome.reason.startswith("other users can list the destination folder 2024/")
    assert not list(public.iterdir())


@pytest.mark.skipif(os.geteuid() != 0, reason="needs files in a group we're not the owner of")
def test_the_copy_keeps_group_access_only_if_that_group_could_reach_the_original(
    open_dir: Path,
) -> None:
    other = _other_gid()
    home = _folder(open_dir / "home", 0o750)  # our own group's
    photo = home / "a.jpg"
    photo.write_bytes(b"x")
    os.chown(photo, -1, other)
    photo.chmod(0o640)
    # Its group can't get into home, so it can't read the photo.
    assert local._reach_mask(photo) & 0o070 == 0
    os.chown(home, -1, other)
    assert local._reach_mask(photo) & 0o070 == 0o070


def test_a_source_folder_moved_mid_run_skips_its_photos_not_the_run(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    for rel in ("a/x.jpg", "a/y.jpg", "b.jpg"):
        shutil.copy2(fixtures_dir / "plain.jpg", src / rel)
    monkeypatch.setattr(local.secrets, "SystemRandom", _KeepOrder)

    def move_a(outcome: Any, index: int, total: int) -> None:
        if outcome.rel == "a/x.jpg":
            (src / "a").rename(src / "a-moved")  # a file manager, Syncthing...

    summary = _run(src, tmp_path / "out", cfg, tmp_path, on_progress=move_a)
    states = _states(summary)
    assert summary.aborted is None
    assert states == {
        "a/x.jpg": State.CLEAN,
        "a/y.jpg": State.SKIPPED_CONFLICT,
        "b.jpg": State.CLEAN,
    }


def test_a_refused_folder_doesnt_block_writable_folders_inside_it(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    (src / "theirs" / "deep").mkdir(parents=True)
    for rel in ("theirs/a.jpg", "theirs/deep/y.jpg", "mine.jpg"):
        shutil.copy2(fixtures_dir / "plain.jpg", src / rel)
    dest = tmp_path / "out"
    (dest / "theirs" / "deep").mkdir(parents=True, mode=0o700)
    real_commit = local._commit

    def theirs_refuses(ram: Path, target: Path, mode: int, **kw: Any) -> bool:
        if target.parent in (dest / "theirs", dest):  # the root refuses too
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_commit(ram, target, mode, **kw)

    monkeypatch.setattr(local, "_commit", theirs_refuses)
    monkeypatch.setattr(local.secrets, "SystemRandom", _KeepOrder)  # theirs/a.jpg first
    summary = _run(src, dest, cfg, tmp_path)
    states = _states(summary)
    assert summary.aborted is None
    assert states["theirs/a.jpg"] == states["mine.jpg"] == State.FAILED_UPLOAD
    assert states["theirs/deep/y.jpg"] == State.CLEAN


def test_a_copy_that_fails_its_read_back_is_removed_on_any_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sshfs and CIFS (noserverino) give each name its own inode number.
    monkeypatch.setattr(local.os.path, "samestat", lambda a, b: False)
    ram = tmp_path / "ram.jpg"
    ram.write_bytes(b"processed")
    dest = tmp_path / "out" / "a.jpg"
    with pytest.raises(local._ReadBackMismatch):
        local._commit(ram, dest, 0o644, expect_sha="0" * 64)
    assert not dest.parent.exists()  # nor the folder made for it


def _nfs4_acl(*aces: tuple[int, int, bytes]) -> bytes:
    """An NFSv4 ACL as Linux shows it: (type, access mask, who) per entry."""
    import struct

    out = struct.pack(">I", len(aces))
    for kind, access, who in aces:
        out += struct.pack(">IIII", kind, 0, access, len(who)) + who.ljust(
            -(-len(who) // 4) * 4, b"\0"
        )
    return out


def test_nfs4_acls_are_read_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    trivial = _nfs4_acl((0, 0x1, b"OWNER@"), (0, 0x1, b"GROUP@"), (0, 0x1, b"EVERYONE@"))
    named = _nfs4_acl((0, 0x1, b"OWNER@"), (0, 0x1, b"bob@example"))
    denied = _nfs4_acl((0, 0x1, b"OWNER@"), (1, 0x1, b"bob@example"))
    for raw, expected in [(trivial, False), (named, True), (denied, False), (named[:-3], True)]:
        monkeypatch.setattr(local.os, "getxattr", lambda fd, name, raw=raw: raw, raising=False)
        assert local._nfs4_named_readers(0) is expected


def test_an_nfs4_acl_that_stays_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    named = _nfs4_acl((0, 0x1, b"OWNER@"), (0, 0x1, b"bob@example"))

    def getxattr(target: Any, name: str) -> bytes:
        if name == "system.nfs4_acl":
            return named  # the server keeps granting bob, whatever the mode
        raise OSError(errno.ENODATA, "No data available")

    monkeypatch.setattr(local.os, "getxattr", getxattr, raising=False)
    ram = tmp_path / "ram.jpg"
    ram.write_bytes(b"processed")
    dest = tmp_path / "out" / "a.jpg"
    with pytest.raises(local._InheritedAcl):
        local._commit(ram, dest, 0o640)
    assert not dest.parent.exists()  # nor the folder made for it


# -- round 5 -----------------------------------------------------------------


def _count_model_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    runs: list[str] = []
    real = local._describe

    def counted(ctx: Any, ram: Path, img: str) -> Any:
        runs.append(img)
        return real(ctx, ram, img)

    monkeypatch.setattr(local, "_describe", counted)
    return runs


def _library(fixtures_dir: Path, root: Path, *rels: str) -> Path:
    for rel in rels:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixtures_dir / "plain.jpg", root / rel)
    return root


@pytest.mark.skipif(os.geteuid() != 0, reason="needs a folder owned by someone else")
def test_the_owner_of_a_folder_above_the_destination_counts(open_dir: Path) -> None:
    src = _folder(open_dir / "private", 0o700)
    share = _folder(open_dir / "bob" / "share", 0o700)
    for folder in (share, share.parent):
        os.chown(folder, 65534, -1)
    # Bob lets us in; we make a folder with the usual 0755 inside his.
    dest = _folder(share / "captioned", 0o755)
    assert local._more_people_can_list(dest, src)
    dest.chmod(0o700)
    assert not local._more_people_can_list(dest, src)


@pytest.mark.skipif(os.geteuid() != 0, reason="needs a folder owned by someone else")
def test_a_folder_shown_as_someone_elses_counts_as_theirs(open_dir: Path) -> None:
    # On a share it may be our own account under another number, or an
    # all_squash export every local user gets into: no telling, so refused.
    src = _folder(open_dir / "private", 0o700)
    dest = _folder(open_dir / "nas" / "captioned", 0o700)
    os.chown(dest, 65534, -1)
    assert local._more_people_can_list(dest, src)


def test_fuse_mounts_are_judged_by_who_can_get_in(
    open_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _folder(open_dir / "private", 0o700)
    home = _folder(open_dir / "home", 0o700)  # nobody else gets past this here...
    mount = _folder(home / "nas", 0o755)
    dest = _folder(mount / "captioned", 0o755)
    sshfs = (str(mount), "fuse.sshfs", ["rw", "user_id=0"])
    monkeypatch.setattr(
        local, "mount_of", lambda entry: sshfs if entry.startswith(str(mount)) else None
    )
    # ...but the share's other users see the folder as the share shows it.
    assert local._more_people_can_list(dest, src)
    dest.chmod(0o700)
    assert not local._more_people_can_list(dest, src)
    # An original on a mount only we can enter is private, whatever its modes.
    assert local._more_people_can_list(_folder(open_dir / "pub", 0o755), mount)
    assert local._reach_mask(mount) == 0o700
    # allow_other without default_permissions: every local user gets in as the share's account.
    unchecked = (str(mount), "fuse.sshfs", ["rw", "allow_other"])
    monkeypatch.setattr(
        local, "mount_of", lambda e: unchecked if e.startswith(str(mount)) else None
    )
    assert local._more_people_can_list(dest, src)


def _forced_modes(monkeypatch: pytest.MonkeyPatch, under: Path) -> None:
    """Make files and folders under ``under`` act like a FAT drive mounted umask=022."""
    real_fchmod, real_mkdir = os.fchmod, os.mkdir

    def fchmod(fd: int, mode: int) -> None:
        inside = Path(os.readlink(f"/proc/self/fd/{fd}")).is_relative_to(under)
        real_fchmod(fd, 0o644 if inside else mode)

    def mkdir(path: Any, mode: int = 0o777, **kw: Any) -> None:
        real_mkdir(path, mode, **kw)
        if Path(path).is_relative_to(under):
            os.chmod(path, 0o755)

    monkeypatch.setattr(local.os, "fchmod", fchmod)
    monkeypatch.setattr(local.os, "mkdir", mkdir)


def test_a_drive_that_cant_keep_files_private_is_refused_up_front(
    open_dir: Path, fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _library(fixtures_dir, _folder(open_dir / "private", 0o700), "a.jpg")
    dest = open_dir / "usb" / "captioned"
    _folder(dest.parent, 0o755)
    _forced_modes(monkeypatch, dest.parent)
    with pytest.raises(CaptionError, match="same permissions"):
        _run(src, dest, cfg, tmp_path)
    assert not dest.exists()  # nothing named after the photos is left behind


def test_such_a_drive_is_fine_where_nobody_else_can_get_in(
    open_dir: Path, fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # /media/<user> on Ubuntu: a FAT drive under a folder only we can enter.
    src = _library(fixtures_dir, _folder(open_dir / "private", 0o700), "a.jpg")
    media = _folder(open_dir / "media", 0o700)
    _folder(media / "usb", 0o755)
    _forced_modes(monkeypatch, media / "usb")
    summary = _run(src, media / "usb" / "captioned", cfg, tmp_path)
    assert _states(summary) == {"a.jpg": State.CLEAN}


def test_a_private_photo_isnt_copied_where_its_mode_wont_hold(
    open_dir: Path, fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The source folder is public, so the drive is allowed; one photo isn't.
    src = _library(fixtures_dir, _folder(open_dir / "public", 0o755), "open.jpg", "mine.jpg")
    (src / "open.jpg").chmod(0o644)
    (src / "mine.jpg").chmod(0o600)
    dest = open_dir / "usb"
    _forced_modes(monkeypatch, dest)
    summary = _run(src, dest, cfg, tmp_path)
    assert _states(summary) == {"open.jpg": State.CLEAN, "mine.jpg": State.FAILED_UPLOAD}
    assert not (dest / "mine.jpg").exists()


def test_an_unwritable_destination_costs_no_model_runs(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _library(fixtures_dir, tmp_path / "src", "a/1.jpg", "a/2.jpg", "b/1.jpg", "c.jpg")
    dest = tmp_path / "out"
    for folder in ("a", "b"):
        (dest / folder).mkdir(parents=True, mode=0o700)
    runs = _count_model_runs(monkeypatch)
    real_open = os.open

    def refuse_tests(path: Any, flags: int, mode: int = 0o777, **kw: Any) -> int:
        target = Path(path)
        if target.name.startswith(".asif-") and target.parent != dest:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, flags, mode, **kw)

    monkeypatch.setattr(local.os, "open", refuse_tests)
    summary = _run(src, dest, cfg, tmp_path)
    states = _states(summary)
    assert summary.aborted is None
    assert states.pop("c.jpg") == State.CLEAN
    assert set(states.values()) == {State.FAILED_UPLOAD}
    assert len(runs) == 1  # only c.jpg ran the model


def test_a_destination_that_vanishes_stops_the_run(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = _library(fixtures_dir, tmp_path / "src", "a.jpg", "b.jpg", "c.jpg")
    dest = tmp_path / "out"

    def unmount(outcome: Any, index: int, total: int) -> None:
        if index == 1:
            shutil.rmtree(dest)

    summary = _run(src, dest, cfg, tmp_path, on_progress=unmount)
    assert summary.aborted is not None and "is gone" in summary.aborted
    assert len(summary.outcomes) == 2
    assert not dest.exists()  # never made again (on whatever disk is left)


def test_finished_copies_in_a_refused_folder_still_count_as_done(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _library(fixtures_dir, tmp_path / "src", "a/1.jpg", "a/2.jpg")
    dest = tmp_path / "out"
    _run(src, dest, cfg, tmp_path)
    _library(fixtures_dir, src, "a/0.jpg")  # new, and first: the folder is refused then
    real_open = os.open

    def a_refuses(path: Any, flags: int, mode: int = 0o777, **kw: Any) -> int:
        if Path(path).parent == dest / "a" and Path(path).name.startswith(".asif-"):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, flags, mode, **kw)

    monkeypatch.setattr(local.os, "open", a_refuses)
    monkeypatch.setattr(local.secrets, "SystemRandom", _KeepOrder)
    states = _states(_run(src, dest, cfg, tmp_path))
    assert states == {
        "a/0.jpg": State.FAILED_UPLOAD,
        "a/1.jpg": State.SKIPPED_DONE,
        "a/2.jpg": State.SKIPPED_DONE,
    }


def test_a_folder_name_the_destination_cant_hold_costs_one_model_run(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _library(
        fixtures_dir,
        tmp_path / "src",
        "Paris: day 1/1.jpg",
        "Paris: day 1/2.jpg",
        "Paris: day 1/x/3.jpg",
        "ok/1.jpg",
    )
    dest = tmp_path / "out"
    runs = _count_model_runs(monkeypatch)
    real_mkdir = os.mkdir

    def fat_mkdir(path: Any, mode: int = 0o777, **kw: Any) -> None:
        if ":" in Path(path).name:  # FAT and exFAT refuse ':' in names
            raise OSError(errno.EINVAL, "Invalid argument")
        real_mkdir(path, mode, **kw)

    monkeypatch.setattr(local.os, "mkdir", fat_mkdir)
    summary = _run(src, dest, cfg, tmp_path)
    states = _states(summary)
    assert states.pop("ok/1.jpg") == State.CLEAN
    assert set(states.values()) == {State.FAILED_UPLOAD}
    assert len(runs) == 2  # one for ok/, one for the folder that couldn't be made
    reasons = {o.reason for o in summary.outcomes if o.state == State.FAILED_UPLOAD}
    assert reasons == {
        "the destination can't hold a folder named 'Paris: day 1' (Invalid argument)"
    }


def _nfs4_inherited(monkeypatch: pytest.MonkeyPatch, under: Path) -> None:
    """New files under ``under`` get an NFSv4 ACL letting bob read them."""
    named = _nfs4_acl((0, 0x1, b"OWNER@"), (0, 0x1, b"bob@example"))

    def getxattr(target: Any, name: str) -> bytes:
        path = Path(os.readlink(f"/proc/self/fd/{target}") if isinstance(target, int) else target)
        if name == "system.nfs4_acl" and path.is_relative_to(under) and path.is_file():
            return named
        raise OSError(errno.ENODATA, "No data available")

    monkeypatch.setattr(local.os, "getxattr", getxattr, raising=False)


def test_an_inherited_nfs4_acl_is_caught_before_the_model_runs(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _library(fixtures_dir, tmp_path / "src", "2024-01/1.jpg", "2024-02/1.jpg", "c.jpg")
    dest = tmp_path / "out"
    dest.mkdir(mode=0o700)
    _nfs4_inherited(monkeypatch, dest)
    runs = _count_model_runs(monkeypatch)
    summary = _run(src, dest, cfg, tmp_path)
    assert set(_states(summary).values()) == {State.FAILED_UPLOAD}
    assert runs == []
    assert not list(dest.iterdir())  # no folder named after a source folder


def test_an_nfs4_acl_on_a_destination_folder_counts(
    open_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _folder(open_dir / "private", 0o700)
    dest = _folder(open_dir / "nfs" / "out", 0o700)
    named = _nfs4_acl((0, 0x1, b"OWNER@"), (0, 0x1, b"family@example"))

    def getxattr(target: Any, name: str) -> bytes:
        if name == "system.nfs4_acl" and Path(target) == dest:
            return named
        raise OSError(errno.ENODATA, "No data available")

    monkeypatch.setattr(local.os, "getxattr", getxattr, raising=False)
    assert local._more_people_can_list(dest, src)


def test_a_source_acl_is_read_not_just_noticed() -> None:
    # (tag, permissions, id): user::rwx, group::r-x, other::r-x, mask::r-x
    base = [(0x01, 7, 0), (0x04, 5, 0), (0x20, 5, 0), (0x10, 5, 0)]
    gid = 4242  # not ours
    audience = local._acl_audience
    assert audience(base, gid, 4, wide=False) == local._ANYONE
    assert audience([*base, (0x02, 5, 54321)], gid, 4, wide=False) == local._ANYONE
    # Someone named and left out: nobody else is sure to get in.
    assert audience([*base, (0x02, 0, 54321)], gid, 4, wide=False) == local._NOBODY
    group_only = [(0x01, 7, 0), (0x04, 5, 0), (0x20, 0, 0), (0x10, 5, 0)]
    assert audience(group_only, gid, 4, wide=False) == gid


def test_a_private_group_isnt_private_if_someone_else_lives_in_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace as NS

    monkeypatch.setattr(local.os, "getgid", lambda: 1000)
    monkeypatch.setattr(local.os, "getuid", lambda: 1000)
    monkeypatch.setattr(local.pwd, "getpwuid", lambda uid: NS(pw_name="alice"))
    monkeypatch.setattr(local.grp, "getgrgid", lambda gid: NS(gr_name="alice", gr_mem=[]))
    alice = NS(pw_name="alice", pw_gid=1000)
    check = local._nobody_elses.__wrapped__
    monkeypatch.setattr(local.pwd, "getpwall", lambda: [alice])
    assert check(1000)
    monkeypatch.setattr(local.pwd, "getpwall", lambda: [alice, NS(pw_name="svc", pw_gid=1000)])
    assert not check(1000)


def test_a_destination_root_we_cant_write_in_still_takes_copies_in_our_folders(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A NAS share: the root is the admin's, the year folders are ours.
    src = _library(fixtures_dir, tmp_path / "src", "2024/a.jpg", "b.jpg")
    dest = tmp_path / "share"
    (dest / "2024").mkdir(parents=True, mode=0o700)
    real_open, real_mkdir = os.open, os.mkdir

    def root_refuses(path: Any, *a: Any, **kw: Any) -> Any:
        if Path(path).parent == dest and Path(path).name.startswith(".asif-"):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, *a, **kw)

    def root_refuses_mkdir(path: Any, *a: Any, **kw: Any) -> None:
        if Path(path).parent == dest:
            raise PermissionError(errno.EACCES, "Permission denied")
        real_mkdir(path, *a, **kw)

    monkeypatch.setattr(local.os, "open", root_refuses)
    monkeypatch.setattr(local.os, "mkdir", root_refuses_mkdir)
    summary = _run(src, dest, cfg, tmp_path)
    assert summary.aborted is None
    assert _states(summary) == {"2024/a.jpg": State.CLEAN, "b.jpg": State.FAILED_UPLOAD}


# -- round 6 -----------------------------------------------------------------


def test_a_drive_unmounted_from_its_mount_point_stops_the_run(
    fixtures_dir: Path, cfg: Config, tmp_path: Path
) -> None:
    src = _library(fixtures_dir, tmp_path / "src", "a.jpg", "b.jpg", "c.jpg")
    dest = tmp_path / "nas"

    def unmount(outcome: Any, index: int, total: int) -> None:
        if index == 1:  # the share goes; its empty mount point stays behind
            dest.rename(tmp_path / "was-mounted")
            dest.mkdir()

    summary = _run(src, dest, cfg, tmp_path, on_progress=unmount)
    assert summary.aborted is not None and "is gone" in summary.aborted
    assert not list(dest.iterdir())  # nothing landed on the disk underneath


@pytest.mark.parametrize("err", [errno.ENOTCONN, errno.EIO])
def test_a_share_that_stops_answering_stops_the_run_cleanly(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int
) -> None:
    src = _library(fixtures_dir, tmp_path / "src", "a.jpg", "b.jpg", "c.jpg")
    dest = tmp_path / "nas"
    dropped = {"now": False}
    real_stat = os.stat

    def stat(path: Any, *a: Any, **kw: Any) -> os.stat_result:
        if dropped["now"] and str(path).startswith(str(dest)):
            raise OSError(err, os.strerror(err))
        return real_stat(path, *a, **kw)

    def drop(outcome: Any, index: int, total: int) -> None:
        dropped["now"] = True

    monkeypatch.setattr(local.os, "stat", stat)
    summary = _run(src, dest, cfg, tmp_path, on_progress=drop)
    assert summary.aborted is not None and summary.aborted.endswith("nothing more will be written")
    assert len(summary.outcomes) == 2


def test_a_private_folder_isnt_copied_onto_a_drive_that_cant_keep_it_private(
    open_dir: Path, fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _library(
        fixtures_dir, _folder(open_dir / "public", 0o755), "open.jpg", "Private/sub/a.jpg"
    )
    (src / "open.jpg").chmod(0o644)
    (src / "Private").chmod(0o700)
    dest = open_dir / "usb"
    _forced_modes(monkeypatch, dest)
    runs = _count_model_runs(monkeypatch)
    summary = _run(src, dest, cfg, tmp_path)
    assert _states(summary) == {"open.jpg": State.CLEAN, "Private/sub/a.jpg": State.FAILED_UPLOAD}
    assert len(runs) == 1  # refused before the model ran
    assert not (dest / "Private").exists()  # its name never shows up there


def test_a_copy_that_cant_be_kept_private_never_holds_a_byte(
    open_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = open_dir / "usb" / "Private" / "a.jpg"
    _folder(open_dir / "usb", 0o755)
    _forced_modes(monkeypatch, open_dir / "usb")
    sizes: list[int] = []
    real = local._kept_private

    def spy(fd: int, mode: int, folder: Path) -> None:
        sizes.append(os.fstat(fd).st_size)
        real(fd, mode, folder)

    monkeypatch.setattr(local, "_kept_private", spy)
    ram = open_dir / "ram.jpg"
    ram.write_bytes(b"private bytes")
    with pytest.raises(local._ModeNotKept):
        local._commit(ram, dest, 0o600)
    assert sizes == [0]
    assert not dest.parent.exists()  # the folder made for it is gone again


def test_a_share_that_shows_our_files_as_someone_elses_still_takes_copies(
    fixtures_dir: Path, cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The kernel won't let a non-owner remove an ACL: EPERM, not a failure.
    def removexattr(*a: Any) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(local.os, "removexattr", removexattr, raising=False)
    src = _library(fixtures_dir, tmp_path / "src", "a.jpg")
    summary = _run(src, tmp_path / "out", cfg, tmp_path)
    assert _states(summary) == {"a.jpg": State.CLEAN}
