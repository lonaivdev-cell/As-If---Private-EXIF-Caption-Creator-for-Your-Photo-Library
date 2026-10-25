from __future__ import annotations

from pathlib import Path

import pytest

from asif import paths
from asif.config import Config, ConfigError, load_config


def test_missing_file_means_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg == Config()
    assert cfg.general.caption_lang == "en"
    assert cfg.resources.max_edge_px == 1024
    assert cfg.guards.max_failure_rate == 0.02


def test_reads_values(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[general]\ncaption_lang = "pt-BR"\ncompat = "wide"\n'
        '[model]\nid = "stub"\nserver_args = ["--flash-attn"]\n'
    )
    cfg = load_config(path)
    assert cfg.general.caption_lang == "pt-BR"
    assert cfg.general.compat == "wide"
    assert cfg.model.server_args == ["--flash-attn"]


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("[general]\nmax_edge = 512\n", "general.max_edge"),  # typo'd key is an error
        ("[nonsense]\nx = 1\n", "nonsense"),
        ('[general]\ncaption_lang = "Portuguese"\n', "BCP-47"),
        ('[general]\ncaption_lang = "en\\n-if"\n', "BCP-47"),
        ('[general]\ncaption_lang = "all"\n', "BCP-47"),  # an exiftool wildcard
        ('[general]\ncompat = "iptc"\n', "general.compat"),
        ("[resources]\nmax_edge_px = 10\n", "max_edge_px"),
        ("[general\n", "not valid TOML"),
    ],
)
def test_bad_config_is_a_clear_error(tmp_path: Path, body: str, fragment: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(body)
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert fragment in str(info.value)
    assert str(path) in str(info.value)


@pytest.mark.parametrize("lang", ["en", "pt-BR", "zh-Hant", "zh-Hant-TW", "es-419", "fil"])
def test_language_tags_accepted(lang: str) -> None:
    cfg = Config()
    cfg.general.caption_lang = lang
    assert cfg.general.caption_lang == lang


def test_assignment_is_validated() -> None:
    cfg = Config()
    with pytest.raises(ValueError):
        cfg.general.caption_lang = "x-default"
    with pytest.raises(ValueError):
        cfg.resources.max_edge_px = 99999


def test_asif_home_puts_everything_in_one_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASIF_HOME", str(tmp_path))
    for path in (paths.config_dir(), paths.data_dir(), paths.state_dir(), paths.cache_dir()):
        assert tmp_path in path.parents
    assert paths.config_file() == tmp_path / "config" / "config.toml"
    assert Config().resources.staging_path() == tmp_path / "cache" / "staging"


@pytest.mark.skipif(__import__("sys").platform == "darwin", reason="XDG is Linux-only")
def test_xdg_variables_are_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASIF_HOME")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", "relative/paths/are/ignored")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert paths.config_dir() == tmp_path / "cfg" / "asif"
    assert paths.state_dir() == tmp_path / "home" / ".local" / "state" / "asif"


def test_ensure_private_dir(tmp_path: Path) -> None:
    path = paths.ensure_private_dir(tmp_path / "a" / "b")
    assert path.is_dir()
    assert path.stat().st_mode & 0o777 == 0o700
