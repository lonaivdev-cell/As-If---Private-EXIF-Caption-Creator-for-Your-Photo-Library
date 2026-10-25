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

    def fill_up(ram: Path, dest: Path, mode: int) -> bool:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_commit(ram, dest, mode)

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
