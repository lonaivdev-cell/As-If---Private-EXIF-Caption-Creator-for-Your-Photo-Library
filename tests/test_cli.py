from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from asif import cli, paths
from asif.cli import EXIT_INTERRUPTED, EXIT_ITEMS_FAILED, EXIT_USAGE, app
from tests.conftest import requires_exiftool

runner = CliRunner()


def _strict_stdout(monkeypatch: pytest.MonkeyPatch) -> io.BytesIO:
    """Send the CLI's normal output to a strict UTF-8 stream, as on a desktop's en_US.UTF-8.

    Unlike C.UTF-8, such a stream refuses to print a name that isn't UTF-8.
    """
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict", write_through=True)
    monkeypatch.setattr(cli, "console", Console(file=stream, width=200))
    return raw


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "As If" in result.stdout


def test_config_path_init_show(tmp_path: Path) -> None:
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    path = Path(result.stdout.strip())
    assert not path.exists()

    assert runner.invoke(app, ["config", "init"]).exit_code == 0
    assert path.exists()
    assert runner.invoke(app, ["config", "init"]).exit_code == EXIT_USAGE  # no clobber
    assert runner.invoke(app, ["config", "init", "--force"]).exit_code == 0

    shown = runner.invoke(app, ["config", "show"])
    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["general"]["caption_lang"] == "en"


def test_starter_config_is_valid_and_matches_defaults(tmp_path: Path) -> None:
    from asif.cli import STARTER_CONFIG
    from asif.config import Config, load_config

    path = tmp_path / "c.toml"
    path.write_text(STARTER_CONFIG)
    assert load_config(path) == Config()


def test_bad_config_file_is_reported(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[general]\ncaption_lang = 'nope nope'\n")
    result = runner.invoke(app, ["--config", str(bad), "config", "show"])
    assert result.exit_code == EXIT_USAGE
    assert "BCP-47" in result.output


def test_a_config_file_that_isnt_utf8_is_a_config_problem(tmp_path: Path) -> None:
    latin1 = tmp_path / "config.toml"
    latin1.write_bytes(b'# Configura\xe7\xe3o\n[general]\ncaption_lang = "pt-BR"\n')
    for args in (["config", "show"], ["caption", str(tmp_path), str(tmp_path / "out")]):
        result = runner.invoke(app, ["--config", str(latin1), *args])
        assert result.exit_code == EXIT_USAGE, result.output
        assert "Config problem" in result.output
        assert "save it as UTF-8" in result.output


def test_models_list() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "qwen3.5-9b-q4km" in result.stdout


def test_caption_without_a_model_explains_what_to_do(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    result = runner.invoke(app, ["caption", str(tmp_path / "src"), str(tmp_path / "out")])
    assert result.exit_code == EXIT_USAGE
    assert "--model stub" in result.output


def test_caption_rejects_bad_language(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    result = runner.invoke(
        app, ["caption", str(tmp_path / "src"), str(tmp_path / "out"), "-m", "stub", "--lang", "!!"]
    )
    assert result.exit_code == EXIT_USAGE
    assert "BCP-47" in result.output


@requires_exiftool
def test_caption_stub_end_to_end(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for name in ("plain.jpg", "gps.jpg", "plain.png"):
        shutil.copy2(fixtures_dir / name, src / name)
    out = tmp_path / "out"
    result = runner.invoke(app, ["caption", str(src), str(out), "--model", "stub"])
    assert result.exit_code == 0, result.output
    assert "captioned" in result.output
    assert sorted(p.name for p in out.iterdir()) == ["gps.jpg", "plain.jpg", "plain.png"]


@requires_exiftool
def test_caption_exit_code_reports_quarantined_photos(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / "ok.jpg")
    shutil.copy2(fixtures_dir / "corrupt.jpg", src / "bad.jpg")
    result = runner.invoke(app, ["caption", str(src), str(tmp_path / "out"), "-m", "stub"])
    assert result.exit_code == EXIT_ITEMS_FAILED, result.output
    assert "bad.jpg" in result.output
    assert "Needs a look" in result.output


@requires_exiftool
def test_caption_dry_run_shows_captions(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / "a.jpg")
    result = runner.invoke(
        app, ["caption", str(src), str(tmp_path / "out"), "-m", "stub", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("args", [["doctor", "--json"]])
def test_doctor_json(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code in (0, EXIT_USAGE), result.output
    report = json.loads(result.stdout)
    assert "specs" in report and "checks" in report


def test_a_missing_config_file_is_an_error_not_silent_defaults(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(tmp_path / "typo.toml"), "config", "show"])
    assert result.exit_code == EXIT_USAGE
    assert "doesn't exist" in " ".join(result.output.split())


def test_doctor_failure_is_a_setup_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    import asif.doctor as doctor_mod

    real = doctor_mod.run_doctor

    def failing(cfg: object, probe: object = None) -> object:
        report = real(cfg)  # type: ignore[arg-type]
        report.checks.append(doctor_mod.Check("exiftool", "fail", "missing", "install it"))
        return report

    monkeypatch.setattr(doctor_mod, "run_doctor", failing)
    assert runner.invoke(app, ["doctor"]).exit_code == EXIT_USAGE


@requires_exiftool
def test_dry_run_preview_says_when_the_owners_caption_is_kept(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "user_xmp.jpg", src / "owner.jpg")
    result = runner.invoke(
        app, ["caption", str(src), str(tmp_path / "out"), "-m", "stub", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "your caption is kept" in result.output


@requires_exiftool
def test_ctrl_c_exits_130_after_cleaning_up(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asif.pipeline import local

    def interrupted(ctx: object, ram: Path, img: str) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(local, "_describe", interrupted)
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / "a.jpg")
    result = runner.invoke(app, ["caption", str(src), str(tmp_path / "out"), "-m", "stub"])
    assert result.exit_code == EXIT_INTERRUPTED, result.output
    assert "stopped" in result.output.lower()
    assert "No photos found" not in result.output


def test_summary_shows_names_that_arent_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    from asif.pipeline.local import ItemOutcome, RunSummary, iter_problems
    from asif.pipeline.states import State

    out = _strict_stdout(monkeypatch)
    name = os.fsdecode(b"caf\xe9")  # Latin-1 bytes, as from an old camera card
    summary = RunSummary(
        run_id="r_1",
        dry_run=False,
        outcomes=[
            ItemOutcome(f"{name}.jpg", State.CLEAN, "empty", note=f"see {name}.jpg"),
            ItemOutcome(f"{name}.heic", State.SKIPPED_FORMAT, "HEIC/AVIF: not yet"),
            ItemOutcome("b.jpg", State.FAILED_UPLOAD, f"can't write into the folder {name}/"),
        ],
        unreadable_dirs=[f"{name}-locked"],
        linked_dirs=[f"{name}-link"],
        aborted=f"the destination {name}/ is full",
    )
    cli._print_summary(summary, list(iter_problems(summary)))
    text = out.getvalue().decode("utf-8")
    for shown in (
        "caf\\xe9.jpg",  # a note
        "caf\\xe9.heic",  # a skipped photo
        "folder caf\\xe9/",  # a reason
        "caf\\xe9-locked/",
        "caf\\xe9-link/",
    ):
        assert shown in text


@requires_exiftool
@pytest.mark.parametrize("dry_run", [False, True])
def test_names_that_arent_utf8_dont_crash_the_summary(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    out = _strict_stdout(monkeypatch)
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / os.fsdecode(b"caf\xe9.jpg"))
    (src / os.fsdecode(b"F\xe9rias.heic")).write_bytes(b"not written yet")
    args = ["caption", str(src), str(tmp_path / "out"), "-m", "stub"]
    result = runner.invoke(app, [*args, "--dry-run"] if dry_run else args)
    assert result.exit_code == 0, result.output
    text = out.getvalue().decode("utf-8")
    assert "F\\xe9rias.heic" in text  # under "Needs a look"
    if dry_run:
        assert "caf\\xe9.jpg" in text  # in the preview
    else:
        assert "1" in text and "captioned" in text


def test_summary_escapes_control_characters_in_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from asif.pipeline.local import ItemOutcome, RunSummary, iter_problems
    from asif.pipeline.states import State

    out = _strict_stdout(monkeypatch)
    names = {
        "x\x1b]52;c;QQ==\x1b\\.heic": "x\\x1b]52;c;QQ==\\x1b\\.heic",  # writes the clipboard
        "y\x1b[2K\x1b[1A.heic": "y\\x1b[2K\\x1b[1A.heic",  # erases the line above
        "z\x9b2K.heic": "z\\x9b2K.heic",  # the same, as one C1 character
        "w\x7f\r.heic": "w\\x7f\\x0d.heic",  # DEL, and a return to the start of the line
    }
    summary = RunSummary(
        run_id="r_1",
        dry_run=False,
        outcomes=[ItemOutcome(name, State.SKIPPED_FORMAT, "HEIC/AVIF: not yet") for name in names],
    )
    cli._print_summary(summary, list(iter_problems(summary)))
    text = out.getvalue().decode("utf-8")
    for char in ("\x1b", "\x9b", "\x7f", "\r"):
        assert char not in text
    for shown in names.values():
        assert shown in text


@requires_exiftool
def test_control_characters_in_a_name_never_reach_the_terminal(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _strict_stdout(monkeypatch)
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / "a.jpg")
    (src / "x\x1b]52;c;QQ==\x1b\\.heic").write_bytes(b"from a downloaded zip")
    result = runner.invoke(app, ["caption", str(src), str(tmp_path / "out"), "-m", "stub"])
    assert result.exit_code == 0, result.output
    text = out.getvalue().decode("utf-8") + result.output
    assert "\x1b" not in text
    assert "x\\x1b]52;c;QQ==\\x1b\\.heic" in text  # under "Needs a look"


def test_the_model_log_keeps_its_lines_but_not_its_control_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asif.infer.client import InferenceError
    from asif.pipeline import local

    def failing(*args: object, **kwargs: object) -> object:
        raise InferenceError("start", "llama-server said:\nfirst line\nsecond\x1b[2K line")

    monkeypatch.setattr(local, "caption_folder", failing)
    (tmp_path / "src").mkdir()
    result = runner.invoke(
        app, ["caption", str(tmp_path / "src"), str(tmp_path / "out"), "-m", "stub"]
    )
    assert result.exit_code == EXIT_USAGE, result.output
    assert "\x1b" not in result.output
    lines = [line.strip(" │") for line in result.output.splitlines()]
    assert "first line" in lines
    assert "second\\x1b[2K line" in lines


@requires_exiftool
@pytest.mark.parametrize("stage", ["open", "commit"])
def test_a_timings_log_that_fills_its_disk_mid_run_changes_no_photo(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """The state disk fills up as a photo's ``stage`` event is logged."""
    import errno
    from typing import Any

    full = False
    real_open = Path.open

    class FillingLog:
        def __init__(self, fh: Any) -> None:
            self.fh = fh

        def __enter__(self) -> FillingLog:
            return self

        def __exit__(self, *exc: object) -> None:
            self.fh.close()

        def write(self, line: str) -> int:
            nonlocal full
            full = full or f'"stage":"{stage}"' in line
            if full:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            return int(self.fh.write(line))

    def filling_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        fh = real_open(self, mode, *args, **kwargs)
        return FillingLog(fh) if self.name == "timings.jsonl" and mode == "a" else fh

    monkeypatch.setattr(Path, "open", filling_open)
    src = tmp_path / "src"
    src.mkdir()
    for name in ("a.jpg", "b.jpg"):
        shutil.copy2(fixtures_dir / "plain.jpg", src / name)
    out = tmp_path / "out"
    result = runner.invoke(app, ["caption", str(src), str(out), "-m", "stub"])
    assert result.exit_code == 0, result.output
    message = " ".join(result.output.split())
    assert "2 captioned" in message
    assert "Needs a look" not in message
    assert sorted(p.name for p in out.iterdir()) == ["a.jpg", "b.jpg"]
    assert "stopped partway through the run (No space left on device)" in message
    assert "Your photos weren't affected" in message
    assert message.index("2 captioned") < message.index("stopped partway")  # after the summary
    assert full


@pytest.mark.parametrize("broken", ["state_is_a_file", "log_is_a_folder", "no_access"])
def test_a_state_folder_that_cant_be_written_stops_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    home = tmp_path / "home"
    (home / "state").mkdir(parents=True)
    (home / paths.HOME_MARKER).touch()
    if broken == "state_is_a_file":
        (home / "state").rmdir()
        (home / "state").write_text("")
    elif broken == "log_is_a_folder":
        (home / "state" / "timings.jsonl").mkdir()
    else:  # someone else's log, say after a sudo run (tests may run as root: fake it)
        from asif.pipeline import timings

        real = timings.TimingsLog

        def refused(path: Path | None, run_id: str, model: str) -> timings.TimingsLog:
            if path is not None:
                raise PermissionError(13, "Permission denied", str(path))
            return real(path, run_id=run_id, model=model)

        monkeypatch.setattr(timings, "TimingsLog", refused)
    monkeypatch.setenv("ASIF_HOME", str(home))
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    result = runner.invoke(app, ["caption", str(src), str(out), "-m", "stub"])
    assert result.exit_code == EXIT_USAGE, result.output
    message = " ".join(result.output.split())
    assert "Can't start" in message
    assert "mounted" in message
    assert "enabled = false under [telemetry]" in message
    assert not out.exists()


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# Each command but these stops when ASIF_HOME isn't set up.
_WORK_WITHOUT_HOME = {"version", "config path", "config init"}
_STOP_WITHOUT_HOME = [
    ("caption", ["caption", "{src}", "{out}", "-m", "stub"]),
    ("caption", ["caption", "{src}", "{out}", "-m", "stub", "--dry-run"]),
    ("config show", ["config", "show"]),
    ("config show", ["--config", "{cfg}", "config", "show"]),
    ("doctor", ["doctor", "--json"]),
    ("models list", ["models", "list"]),
    ("models recommend", ["models", "recommend"]),
]


def test_every_command_is_listed_as_needing_asif_home_or_not() -> None:
    import typer.main

    def leaves(command: object, words: tuple[str, ...] = ()) -> list[str]:
        subcommands = getattr(command, "commands", None)
        if subcommands is None:
            return [" ".join(words)]
        return [n for name, sub in subcommands.items() for n in leaves(sub, (*words, name))]

    listed = {name for name, _ in _STOP_WITHOUT_HOME} | _WORK_WITHOUT_HOME
    assert set(leaves(typer.main.get_command(app))) == listed


@pytest.mark.parametrize(
    ("home", "mount_point_left"),
    [
        ("Vault/asif", False),  # the mount point goes away when locked (/media, /Volumes)
        ("Vault/asif", True),  # an empty folder is left (gocryptfs, CryFS, fstab LUKS)
        ("Vault", True),  # ASIF_HOME is the mount point itself
    ],
)
@pytest.mark.parametrize(("command", "args"), _STOP_WITHOUT_HOME)
def test_an_asif_home_that_isnt_set_up_stops_before_the_config_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home: str,
    mount_point_left: bool,
    command: str,
    args: list[str],
) -> None:
    if mount_point_left:
        (tmp_path / "Vault").mkdir()
    monkeypatch.setenv("ASIF_HOME", str(tmp_path / home))
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"never read")
    cfg = tmp_path / "mine.toml"
    cfg.write_text('[general]\ncaption_lang = "pt-BR"\n')
    before = _tree(tmp_path)
    fill = {"src": str(src), "out": str(tmp_path / "out"), "cfg": str(cfg)}
    result = runner.invoke(app, [arg.format(**fill) for arg in args])
    assert result.exit_code == EXIT_USAGE, result.output
    message = " ".join(result.output.split())
    assert f"Can't start: ASIF_HOME {tmp_path / home} isn't set up here" in message
    assert "is its volume mounted?" in message
    assert "run `asif config init`" in message
    assert _tree(tmp_path) == before  # nothing made on the disk under a mount point


def test_version_and_config_path_work_when_asif_home_isnt_set_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Vault").mkdir()
    monkeypatch.setenv("ASIF_HOME", str(tmp_path / "Vault" / "asif"))
    assert runner.invoke(app, ["version"]).exit_code == 0
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0, result.output
    shown = "".join(result.stdout.split())  # rich wraps long paths
    assert shown == str(tmp_path / "Vault" / "asif" / "config" / "config.toml")
    assert _tree(tmp_path) == ["Vault"]


def test_config_init_never_creates_the_folders_asif_home_is_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASIF_HOME", str(tmp_path / "Vault" / "asif"))
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == EXIT_USAGE, result.output
    message = " ".join(result.output.split())
    assert f"Can't set up ASIF_HOME {tmp_path / 'Vault' / 'asif'}" in message
    assert f"{tmp_path / 'Vault'} doesn't exist" in message
    assert "check that it's mounted" in message
    assert _tree(tmp_path) == []


def test_config_init_sets_up_asif_home_for_the_commands_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Vault").mkdir()  # mounted
    home = tmp_path / "Vault" / "asif"
    monkeypatch.setenv("ASIF_HOME", str(home))
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0, result.output
    assert f"Set up ASIF_HOME {home}" in " ".join(result.output.split())
    assert (home / paths.HOME_MARKER).is_file()
    assert (home / "config" / "config.toml").is_file()
    assert home.stat().st_mode & 0o777 == 0o700
    assert runner.invoke(app, ["config", "show"]).exit_code == 0
    (tmp_path / "src").mkdir()
    result = runner.invoke(
        app, ["caption", str(tmp_path / "src"), str(tmp_path / "out"), "-m", "stub"]
    )
    assert result.exit_code == 0, result.output
    assert "No photos found" in result.output
    assert (home / "state" / "timings.jsonl").is_file()


def test_config_init_keeps_the_settings_in_an_asif_home_without_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "asif"
    (home / "config").mkdir(parents=True)  # made by an older As If
    (home / "config" / "config.toml").write_text('[general]\ncaption_lang = "pt-BR"\n')
    monkeypatch.setenv("ASIF_HOME", str(home))
    assert runner.invoke(app, ["config", "show"]).exit_code == EXIT_USAGE
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0, result.output
    assert "Kept your config file" in result.output
    shown = runner.invoke(app, ["config", "show"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout)["general"]["caption_lang"] == "pt-BR"


def test_a_library_of_linked_folders_says_so(tmp_path: Path) -> None:
    elsewhere = tmp_path / "Sync" / "DCIM"
    elsewhere.mkdir(parents=True)
    (elsewhere / "a.jpg").write_bytes(b"never read")
    src = tmp_path / "Pictures"
    src.mkdir()
    (src / "Camera").symlink_to(elsewhere, target_is_directory=True)
    result = runner.invoke(app, ["caption", str(src), str(tmp_path / "out"), "-m", "stub"])
    assert result.exit_code == EXIT_ITEMS_FAILED, result.output
    message = " ".join(result.output.split())
    assert "No photos found." not in message
    assert "some folders are links" in message
    assert "Camera/" in message
    assert "a link to a folder" in message  # under "Needs a look"


@requires_exiftool
def test_a_linked_folder_next_to_photos_is_listed(fixtures_dir: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "Sync" / "DCIM"
    elsewhere.mkdir(parents=True)
    shutil.copy2(fixtures_dir / "plain.jpg", elsewhere / "b.jpg")
    src = tmp_path / "Pictures"
    src.mkdir()
    shutil.copy2(fixtures_dir / "plain.jpg", src / "a.jpg")
    (src / "2024").mkdir()
    (src / "2024" / "Camera").symlink_to(elsewhere, target_is_directory=True)
    out = tmp_path / "out"
    result = runner.invoke(app, ["caption", str(src), str(out), "-m", "stub"])
    assert result.exit_code == EXIT_ITEMS_FAILED, result.output
    assert "captioned" in result.output
    assert "2024/Camera/" in result.output
    assert sorted(p.name for p in out.iterdir()) == ["a.jpg"]  # the link isn't followed
