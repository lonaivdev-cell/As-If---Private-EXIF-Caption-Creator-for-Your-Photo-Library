from __future__ import annotations

import os
from pathlib import Path

import pytest

from asif.workdir import RamWorkDir, _mount_fstype, find_ram_dir, wipe_file

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
    with RamWorkDir(str(tmp_path)) as work:
        assert work.path is not None
        assert work.path.parent == tmp_path
        assert work.path.stat().st_mode & 0o777 == 0o700
        a, b = work.new_path(".JPG"), work.new_path(".jpg")
        assert a != b and a.suffix == ".jpg"
        a.write_bytes(b"secret pixels")
        assert work.free_bytes() > 0
        inside = work.path
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
