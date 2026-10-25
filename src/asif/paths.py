"""Where As If keeps its files.

Linux and WSL follow the XDG base-directory spec; macOS uses ``~/Library``.
Setting ``ASIF_HOME`` puts everything under one directory instead, which is
handy for tests, portable installs, or keeping all state on an encrypted
volume.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

APP = "asif"


def _home() -> Path:
    return Path.home()


def _asif_home() -> Path | None:
    value = os.environ.get("ASIF_HOME")
    return Path(value).expanduser() if value else None


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var)
    if value and Path(value).is_absolute():
        return Path(value) / APP
    return _home() / fallback / APP


def config_dir() -> Path:
    if root := _asif_home():
        return root / "config"
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP
    return _xdg("XDG_CONFIG_HOME", ".config")


def data_dir() -> Path:
    """Vault and models live here."""
    if root := _asif_home():
        return root / "data"
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP
    return _xdg("XDG_DATA_HOME", ".local/share")


def state_dir() -> Path:
    """Telemetry and the timings log live here."""
    if root := _asif_home():
        return root / "state"
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP / "state"
    return _xdg("XDG_STATE_HOME", ".local/state")


def cache_dir() -> Path:
    """Staging (ciphertext only, from M4 on) lives here."""
    if root := _asif_home():
        return root / "cache"
    if sys.platform == "darwin":
        return _home() / "Library" / "Caches" / APP
    return _xdg("XDG_CACHE_HOME", ".cache")


def config_file() -> Path:
    return config_dir() / "config.toml"


def models_dir() -> Path:
    return data_dir() / "models"


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) and make the leaf readable only by us.

    The folders ``ASIF_HOME`` sits in are never created: if they're missing
    (an encrypted volume that isn't mounted, a typo), this raises
    ``FileNotFoundError`` rather than quietly making them on whatever disk is
    underneath.
    """
    root = _asif_home()
    if root is not None and path.is_relative_to(root):
        try:
            root.mkdir(exist_ok=True)
        except FileNotFoundError as exc:
            missing = str(root.parent)
            raise FileNotFoundError(errno.ENOENT, f"{missing} doesn't exist", missing) from exc
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path
