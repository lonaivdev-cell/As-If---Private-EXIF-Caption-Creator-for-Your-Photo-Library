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


def test_dead_runs_are_swept_wherever_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "runtime"
    elsewhere.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(elsewhere))
    dead = _dead_run(elsewhere, "asif-bbbbbbbbbbbb")
    here = tmp_path / "configured"
    with RamWorkDir(str(here), allow_disk=True) as work:
        assert work.swept >= 1
        assert not dead.exists()


def test_the_lock_is_held_and_never_left_under_a_temporary_name(tmp_path: Path) -> None:
    import fcntl

    with RamWorkDir(str(tmp_path), allow_disk=True) as work:
        assert work.path is not None
        names = sorted(p.name for p in work.path.iterdir())
        assert names == [".lock"]
        fd = os.open(work.path / ".lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


def test_a_work_dir_used_before_is_swept_after_the_setting_changes(tmp_path: Path) -> None:
    old = tmp_path / "old-ram"
    with RamWorkDir(str(old), allow_disk=True):
        pass
    dead = _dead_run(old, "asif-cccccccccccc")  # a run there was killed
    with RamWorkDir(str(tmp_path / "new-ram"), allow_disk=True) as work:
        assert work.swept >= 1
    assert not dead.exists()


def test_a_state_folder_that_cant_be_written_doesnt_stop_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "asif-home"
    home.mkdir()
    (home / "state").write_text("not a folder")
    monkeypatch.setenv("ASIF_HOME", str(home))
    with RamWorkDir(str(tmp_path / "ram"), allow_disk=True) as work:
        assert work.path is not None and work.path.is_dir()
        assert work.unremembered


def test_remote_mounts_are_told_apart_without_touching_them(tmp_path: Path) -> None:
    from asif import workdir

    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/sda1 / ext4 rw 0 0\n"
        "server:/photos /mnt/nas nfs4 rw 0 0\n"
        "me@host: /home/me/my\\040box fuse.sshfs rw 0 0\n"
        "/dev/sdb1 /mnt/nas/local ext4 rw 0 0\n"
    )
    assert workdir._is_remote("/mnt/nas/tmp", mounts)
    assert workdir._is_remote("/home/me/my box/work", mounts)
    assert not workdir._is_remote("/mnt/nas/local/work", mounts)  # deeper mount wins
    assert not workdir._is_remote("/mnt/nasty", mounts)
    assert not workdir._is_remote("/tmp", mounts)


def test_remembered_folders_on_remote_mounts_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asif import workdir

    remote, local_dir = tmp_path / "remote", tmp_path / "local"
    remote.mkdir()
    local_dir.mkdir()
    monkeypatch.setattr(workdir, "_remembered_parents", lambda: [remote, local_dir])
    monkeypatch.setattr(workdir, "_is_remote", lambda entry: entry == str(remote))
    parents = workdir._sweep_parents("")
    assert local_dir.resolve() in parents and remote.resolve() not in parents
    assert remote.resolve() in workdir._sweep_parents(str(remote))  # in use again: swept


def test_macos_mount_output_is_read() -> None:
    from asif import workdir

    out = (
        "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n"
        "//me@nas._smb._tcp.local/Photo Library on /Volumes/Photo Library "
        "(smbfs, nodev, nosuid, mounted by me)\n"
        "map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)\n"
    )
    table = workdir.parse_macos_mounts(out)
    assert ("/Volumes/Photo Library", "smbfs", ["nodev", "nosuid", "mounted by me"]) in table
    assert table[0][:2] == ("/", "apfs")
