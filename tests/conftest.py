"""Shared test fixtures.

The synthetic photos are generated once per test session by
``tests/fixtures/make_fixtures.py``. Tests must never modify them in place:
use ``fixture_copy`` to get a private copy.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fixtures.make_fixtures import FIXTURES, MAKERS, make_fixtures

HAVE_EXIFTOOL = shutil.which("exiftool") is not None

requires_exiftool = pytest.mark.skipif(not HAVE_EXIFTOOL, reason="exiftool not installed")


@pytest.fixture(autouse=True)
def _isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point As If at a throwaway home so tests never touch real config or state."""
    home = tmp_path_factory.mktemp("asif-home")
    monkeypatch.setenv("ASIF_HOME", str(home))


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAVE_EXIFTOOL:
        pytest.skip("exiftool not installed (needed to build the fixtures)")
    out = tmp_path_factory.mktemp("fixtures")
    make_fixtures(out, with_panorama=False)
    return out


@pytest.fixture(scope="session")
def panorama(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAVE_EXIFTOOL:
        pytest.skip("exiftool not installed")
    out = tmp_path_factory.mktemp("panorama") / "panorama.jpg"
    MAKERS["panorama.jpg"](out)
    return out


@pytest.fixture
def fixture_copy(fixtures_dir: Path, tmp_path: Path) -> Callable[[str], Path]:
    """``fixture_copy("exif.jpg")`` -> a private copy in this test's tmp dir."""

    def copy(name: str) -> Path:
        dest = tmp_path / "work" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixtures_dir / name, dest)
        return dest

    return copy


FIXTURE_EXPECT = {f.name: f.expect for f in FIXTURES}
