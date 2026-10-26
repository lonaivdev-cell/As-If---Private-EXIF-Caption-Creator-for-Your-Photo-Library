"""Process hygiene: no core dumps, clean stop on signals, children die with us."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from asif import paths, proc


def _python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_cli_disables_core_dumps_for_itself_and_its_children() -> None:
    result = _python(
        """
        import ctypes, json, resource, subprocess, sys
        from asif import cli
        sys.argv = ["asif", "version"]
        try:
            cli.main()
        except SystemExit:
            pass
        report = {
            "limit": resource.getrlimit(resource.RLIMIT_CORE),
            "dumpable": ctypes.CDLL(None).prctl(3, 0, 0, 0, 0),  # PR_GET_DUMPABLE
            "filter": open("/proc/self/coredump_filter").read().strip(),
        }
        child = subprocess.run(
            [sys.executable, "-c",
             "import json, resource; print(json.dumps(resource.getrlimit(resource.RLIMIT_CORE)));"
             "print(open('/proc/self/coredump_filter').read().strip())"],
            capture_output=True, text=True)
        report["child"] = child.stdout.splitlines()
        print(json.dumps(report))
        """
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    # 1 byte: no core file, and the kernel won't pipe a dump to a crash
    # reporter either (0 if the hard limit was already 0).
    assert report["limit"] in ([1, 1], [0, 0])
    assert report["dumpable"] == 0
    assert int(report["filter"], 16) == 0
    # Inherited by children, and they can't raise it again.
    assert json.loads(report["child"][0]) == report["limit"]
    assert int(report["child"][1], 16) == 0


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP", "SIGQUIT"])
def test_stop_signals_unwind_through_finally(sig: str) -> None:
    result = _python(
        f"""
        import os, signal
        from asif import proc
        proc.install_stop_signals()
        try:
            try:
                os.kill(os.getpid(), signal.{sig})
                signal.pause()
            finally:
                print("cleanup ran")
        except KeyboardInterrupt:
            print("interrupted")
        # A second signal during cleanup is ignored, not fatal.
        os.kill(os.getpid(), signal.{sig})
        print("still alive")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[:3] == ["cleanup ran", "interrupted", "still alive"]


def test_an_ignored_signal_stays_ignored() -> None:
    # nohup ignores SIGHUP; As If must not undo that.
    result = _python(
        """
        import os, signal
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        from asif import proc
        proc.install_stop_signals()
        os.kill(os.getpid(), signal.SIGHUP)
        print("survived", signal.getsignal(signal.SIGHUP) is signal.SIG_IGN)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "survived True"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_die_with_parent_only_offered_when_single_threaded() -> None:
    assert proc.die_with_parent() is not None or threading.active_count() > 1
    stop = threading.Event()
    helper = threading.Thread(target=stop.wait)
    helper.start()
    try:
        assert proc.die_with_parent() is None
    finally:
        stop.set()
        helper.join()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_exiftool_dies_when_the_orchestrator_is_killed(tmp_path: Path) -> None:
    import shutil

    if not shutil.which("exiftool"):
        pytest.skip("exiftool not installed")
    pidfile = tmp_path / "pid"
    orchestrator = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from asif.meta.exiftool import ExifTool
                et = ExifTool()
                et.start()
                open({str(pidfile)!r}, "w").write(str(et._proc.pid))
                time.sleep(60)
                """
            ),
        ]
    )
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text():
                break
            time.sleep(0.1)
        exiftool_pid = int(pidfile.read_text())
        orchestrator.kill()  # SIGKILL: no handler, no cleanup can run
        orchestrator.wait()
        for _ in range(50):
            if not Path(f"/proc/{exiftool_pid}").exists():
                break
            state = Path(f"/proc/{exiftool_pid}/stat").read_text().split()[2]
            if state == "Z":
                break
            time.sleep(0.1)
        else:
            pytest.fail("exiftool outlived its SIGKILLed parent")
    finally:
        orchestrator.kill()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and signals")
def test_sigterm_mid_photo_leaves_no_plaintext_and_no_children(tmp_path: Path) -> None:
    """The real thing: `asif caption`, killed with SIGTERM while a photo is in RAM."""
    import shutil

    if not shutil.which("exiftool"):
        pytest.skip("exiftool not installed")
    from tests.fixtures.make_fixtures import MAKERS

    src = tmp_path / "src"
    src.mkdir()
    MAKERS["gps.jpg"](src / "gps.jpg")
    ram = Path("/dev/shm") / f"asif-test-{os.getpid()}"
    ready = tmp_path / "ready"
    driver = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        from asif import cli
        from asif.pipeline import local

        def slow_describe(ctx, ram, img):
            Path({str(ready)!r}).write_text(str(ram))
            time.sleep(60)

        local._describe = slow_describe
        sys.argv = ["asif", "caption", {str(src)!r}, {str(tmp_path / "out")!r}, "-m", "stub"]
        cli.main()
        """
    )
    env = {**os.environ, "ASIF_HOME": str(tmp_path / "home")}
    (tmp_path / "home" / "config").mkdir(parents=True)
    (tmp_path / "home" / paths.HOME_MARKER).touch()  # set up, as `asif config init` does
    (tmp_path / "home" / "config" / "config.toml").write_text(f'[resources]\nwork_dir = "{ram}"\n')
    run = subprocess.Popen([sys.executable, "-c", driver], env=env, stderr=subprocess.PIPE)
    try:
        for _ in range(300):
            if ready.exists():
                break
            time.sleep(0.1)
        else:
            pytest.fail("the run never reached the model")
        in_ram = Path(ready.read_text())
        assert in_ram.exists()  # the plaintext copy, right now
        children = _children(run.pid)
        run.send_signal(signal.SIGTERM)
        code = run.wait(timeout=60)
        assert code == 130
        assert not in_ram.exists()
        assert not any(p.is_file() for p in ram.rglob("*"))
        time.sleep(0.5)
        assert [pid for pid in children if _alive(pid)] == []
    finally:
        run.kill()
        shutil.rmtree(ram, ignore_errors=True)


def _children(pid: int) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                ppid = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == pid:
                found.append(int(entry.name))
    return found


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_shielded_cleanup_finishes_before_a_signal_is_acted_on() -> None:
    result = _python(
        """
        import os, signal
        from asif import proc
        steps = []
        try:
            with proc.shielded():
                os.kill(os.getpid(), signal.SIGTERM)
                steps.append("cleanup finished")
        except KeyboardInterrupt:
            steps.append("then stopped")
        print(steps)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['cleanup finished', 'then stopped']"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_exiftool_dies_with_its_parent_even_with_threads_running(tmp_path: Path) -> None:
    # The real CLI always has a thread running (rich's progress refresh) when
    # exiftool starts; a preexec_fn would be skipped then. The trampoline isn't.
    import shutil

    if not shutil.which("exiftool"):
        pytest.skip("exiftool not installed")
    pidfile = tmp_path / "pid"
    orchestrator = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import threading, time
                from asif.meta.exiftool import ExifTool
                threading.Thread(target=time.sleep, args=(60,), daemon=True).start()
                et = ExifTool()
                et.start()
                et.version()  # by now the trampoline has become exiftool
                open({str(pidfile)!r}, "w").write(str(et._proc.pid))
                time.sleep(60)
                """
            ),
        ]
    )
    try:
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text():
                break
            time.sleep(0.1)
        exiftool_pid = int(pidfile.read_text())
        orchestrator.kill()
        orchestrator.wait()
        for _ in range(50):
            if not _alive(exiftool_pid):
                break
            time.sleep(0.1)
        else:
            pytest.fail("exiftool outlived its SIGKILLed parent")
    finally:
        orchestrator.kill()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_sigkill_of_the_real_cli_leaves_no_helper_running(tmp_path: Path) -> None:
    """`asif caption` (with rich's progress thread running), killed with SIGKILL."""
    import shutil

    if not shutil.which("exiftool"):
        pytest.skip("exiftool not installed")
    from tests.fixtures.make_fixtures import MAKERS

    src = tmp_path / "src"
    src.mkdir()
    MAKERS["plain.jpg"](src / "a.jpg")
    ready = tmp_path / "ready"
    driver = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        from asif import cli
        from asif.pipeline import local

        def slow_describe(ctx, ram, img):
            Path({str(ready)!r}).write_text("x")
            time.sleep(60)

        local._describe = slow_describe
        sys.argv = ["asif", "caption", {str(src)!r}, {str(tmp_path / "out")!r}, "-m", "stub"]
        cli.main()
        """
    )
    env = {**os.environ, "ASIF_HOME": str(tmp_path / "home")}
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / paths.HOME_MARKER).touch()  # set up, as `asif config init` does
    run = subprocess.Popen([sys.executable, "-c", driver], env=env, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            if ready.exists():
                break
            time.sleep(0.1)
        else:
            pytest.fail("the run never reached the model")
        children = _children(run.pid)
        assert len(children) >= 2  # exiftool and the model worker
        run.kill()
        run.wait()
        for _ in range(100):
            if not [pid for pid in children if _alive(pid)]:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"helpers outlived the SIGKILLed orchestrator: {children}")
    finally:
        run.kill()
