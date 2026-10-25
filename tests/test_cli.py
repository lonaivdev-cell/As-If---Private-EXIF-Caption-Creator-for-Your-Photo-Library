from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from asif.cli import EXIT_INTERRUPTED, EXIT_ITEMS_FAILED, EXIT_USAGE, app
from tests.conftest import requires_exiftool

runner = CliRunner()


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
