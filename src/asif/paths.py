"""Where As If keeps its files.

Linux and WSL follow the XDG base-directory spec; macOS uses ``~/Library``.
Setting ``ASIF_HOME`` puts everything under one directory instead, which is
handy for tests, portable installs, or keeping all state on an encrypted
volume. ``asif config init`` sets that directory up, and it's only used while
the marker file it writes is there: a locked volume often leaves an empty
folder behind at the same path.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

APP = "asif"
HOME_MARKER = ".asif-home"
"""Written into ``ASIF_HOME`` by ``asif config init``."""


def _home() -> Path:
    return Path.home()


def _asif_home() -> Path | None:
    value = os.environ.get("ASIF_HOME")
    return Path(value).expanduser() if value else None


def asif_home_not_set_up() -> Path | None:
    """``ASIF_HOME`` if it's set but not set up here, say because its volume isn't mounted."""
    root = _asif_home()
    if root is None or os.path.isfile(root / HOME_MARKER):
        return None
    return root


def set_up_asif_home() -> bool:
    """Create ``ASIF_HOME`` and its marker. True if it wasn't set up before.

    The folders ``ASIF_HOME`` is in are never created: if they're missing (an
    encrypted volume that isn't mounted, a typo), this raises
    ``FileNotFoundError`` rather than quietly making them on whatever disk is
    underneath. Does nothing when ``ASIF_HOME`` isn't set.
    """
    root = asif_home_not_set_up()
    if root is None:
        return False
    try:
        root.mkdir(mode=0o700, exist_ok=True)
    except FileNotFoundError as exc:
        missing = str(root.parent)
        raise FileNotFoundError(errno.ENOENT, f"{missing} doesn't exist", missing) from exc
    (root / HOME_MARKER).write_text(
        "As If keeps its settings and state in this folder (ASIF_HOME).\n"
        "It only uses the folder while this file is here.\n",
        encoding="utf-8",
    )
    return True


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

    ``ASIF_HOME`` itself is never created here (only ``asif config init``
    does that): if it isn't set up (an encrypted volume that isn't mounted,
    a typo), this raises ``FileNotFoundError`` rather than quietly making
    folders on whatever disk is underneath.
    """
    root = asif_home_not_set_up()
    if root is not None and path.is_relative_to(root):
        raise FileNotFoundError(errno.ENOENT, f"ASIF_HOME {root} isn't set up here", str(root))
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path
