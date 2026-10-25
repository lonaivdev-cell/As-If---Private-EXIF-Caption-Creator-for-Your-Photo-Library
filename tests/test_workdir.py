from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from asif.workdir import (
    NotInRamError,
    RamWorkDir,
    _mount_fstype,
    find_ram_dir,
    sweep_stale,
    wipe_file,
)

MOUNTS = """\
/dev/vda / ext4 rw,relatime 0 0
tmpfs /dev/shm tmpfs rw,nosuid,nodev 0 0
tmpfs /run/user/1000 tmpfs rw,nosuid,nodev,relatime,size=1G,mode=700 0 0
/dev/mapper/home /home ext4 rw 0 0
"""


def test_mount_fstype_picks_the_deepest_mount(tmp_path: Path) -> None:
    mounts = tmp_path / "mounts"
    mounts.write_text(MOUNTS)
    assert _mount_fstype(Path("/dev/shm"), mounts) == "tmpfs"
    assert _mount_fstype(Path("/home"), mounts) == "ext4"
    assert _mount_fstype(Path("/usr"), mounts) == "ext4"
    assert _mount_fstype(Path("/run/user/1000"), mounts) == "tmpfs"


def test_missing_mounts_file_is_unknown(tmp_path: Path) -> None:
    assert _mount_fstype(Path("/"), tmp_path / "missing") is None


def test_override_reports_whether_it_is_ram(tmp_path: Path) -> None:
    mounts = tmp_path / "mounts"
    mounts.write_text(MOUNTS)
    choice = find_ram_dir("/home/me/ram", mounts)
    assert choice.path == Path("/home/me/ram")
    assert not choice.in_ram
    assert "touch disk" in choice.reason


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm")
def test_prefers_a_tmpfs_on_linux() -> None:
    choice = find_ram_dir()
    if __import__("sys").platform.startswith("linux"):
        assert choice.in_ram, choice.reason


def test_ram_work_dir_is_private_and_wiped(tmp_path: Path) -> None:
    with RamWorkDir(str(tmp_path), allow_disk=True) as work:
        assert work.path is not None
        assert work.path.parent == tmp_path
        assert work.path.stat().st_mode & 0o777 == 0o700
        a, b = work.new_path(".JPG"), work.new_path(".jpg")
        assert a != b and a.suffix == ".jpg"
        a.write_bytes(b"secret pixels")
        assert work.free_bytes() > 0
        inside = work.path
        assert (inside / ".lock").exists()
    assert not inside.exists()
    assert list(tmp_path.iterdir()) == []


def test_wipe_file_overwrites_before_unlinking(tmp_path: Path) -> None:
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"\xff\xd8 private bytes" * 1000)
    witness = tmp_path / "witness"
    os.link(target, witness)  # same inode: shows what the wipe did to the data
    wipe_file(target)
    assert not target.exists()
    data = witness.read_bytes()
    assert len(data) == len(b"\xff\xd8 private bytes") * 1000
    assert data == b"\0" * len(data)


def test_wipe_missing_file_is_fine(tmp_path: Path) -> None:
    wipe_file(tmp_path / "nothing")


STACKED = """\
/dev/vda / ext4 rw 0 0
tmpfs /dev/shm tmpfs rw 0 0
/dev/vdb /dev/shm ext4 rw 0 0
"""


def test_the_last_of_two_stacked_mounts_wins(tmp_path: Path) -> None:
    # A disk mounted over /dev/shm hides the tmpfs: files land on the disk.
    mounts = tmp_path / "mounts"
    mounts.write_text(STACKED)
    assert _mount_fstype(Path("/dev/shm"), mounts) == "ext4"


def test_an_on_disk_work_dir_is_refused_unless_allowed(tmp_path: Path) -> None:
    work = RamWorkDir(str(tmp_path))
    if work.choice.in_ram:
        pytest.skip("tmp_path happens to be on a tmpfs here")
    with pytest.raises(NotInRamError, match="written to disk"), work:
        pass
    assert list(tmp_path.iterdir()) == []


def _dead_run(parent: Path, name: str, *, lock: bool = True) -> Path:
    leftover = parent / name
    leftover.mkdir(mode=0o700)
    (leftover / "0123456789abcdef.jpg").write_bytes(b"\xff\xd8 private")
    if lock:
        (leftover / ".lock").touch()
    return leftover


def test_leftovers_of_dead_runs_are_wiped_on_entry(tmp_path: Path) -> None:
    dead = _dead_run(tmp_path, "asif-0123456789ab")
    unrelated = tmp_path / "asif-notours"
    unrelated.mkdir()
    with RamWorkDir(str(tmp_path), allow_disk=True) as work:
        assert work.swept == 1
        assert not dead.exists()
        assert unrelated.exists()  # not one of our names: never touched


def test_a_live_run_is_never_swept(tmp_path: Path) -> None:
    with RamWorkDir(str(tmp_path), allow_disk=True) as first:
        assert first.path is not None
        photo = first.path / "aaaa.jpg"
        photo.write_bytes(b"in use")
        # Another run starting now must leave the first one alone.
        assert sweep_stale(tmp_path) == 0
        assert photo.read_bytes() == b"in use"


def test_unlocked_dirs_are_only_swept_once_clearly_abandoned(tmp_path: Path) -> None:
    fresh = _dead_run(tmp_path, "asif-aaaaaaaaaaaa", lock=False)
    assert sweep_stale(tmp_path) == 0  # might be a run that is just starting
    old = time.time() - 3600
    os.utime(fresh, (old, old))
    assert sweep_stale(tmp_path) == 1
    assert not fresh.exists()
