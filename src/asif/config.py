"""Configuration: ``config.toml`` parsed into typed, validated settings.

Every section mirrors §17 of ``docs/DESIGN.md``. Unknown keys are errors, so a
typo like ``max_edge = 512`` fails loudly instead of being silently ignored.
A missing file means "all defaults".
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from asif import paths

# BCP-47 tags as people actually write them: "en", "pt-BR", "zh-Hant-TW", "es-419".
# Deliberately strict: the value ends up inside an exiftool tag name.
_BCP47 = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z]{4})?(-([A-Za-z]{2}|[0-9]{3}))?$")


class ConfigError(ValueError):
    """The config file exists but can't be used. The message says why."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GeneralConfig(_Section):
    caption_lang: str = "en"
    """BCP-47 language for captions; also filters models by language support."""
    compat: Literal["xmp", "wide"] = "xmp"
    """``wide`` also mirrors keywords into IPTC:Keywords for older apps."""

    @field_validator("caption_lang")
    @classmethod
    def _check_lang(cls, value: str) -> str:
        if not _BCP47.fullmatch(value) or value.lower() == "all":
            # "all" would also turn into a wildcard inside exiftool tag names.
            raise ValueError(f"{value!r} is not a BCP-47 language tag like 'en' or 'pt-BR'")
        return value


class SourceConfig(_Section):
    adapter: Literal["local", "proton", "rclone", "ssh"] = "local"
    scope: str = ""
    dest: str = ""
    """``in-place`` or ``mirror:<path>``."""


class ResourcesConfig(_Section):
    mode: Literal["batch", "linear"] = "batch"
    staging_dir: str = ""
    """Empty means ``<cache dir>/staging``."""
    staging_cap_gb: float = Field(50, gt=0)
    ram_scratch_mb: int = Field(1024, ge=64)
    max_edge_px: int = Field(1024, ge=224, le=4096)
    """Longest edge sent to the model. The #1 speed lever on CPU (§9.3)."""
    work_dir: str = ""
    """Override for the RAM work dir. Empty means auto-detect a tmpfs."""

    def staging_path(self) -> Path:
        if self.staging_dir:
            return Path(self.staging_dir).expanduser()
        return paths.cache_dir() / "staging"


class ModelConfig(_Section):
    id: str = "auto"
    """``auto``, a manifest id, ``local`` (use ``gguf``/``mmproj`` below), or ``stub``."""
    device: Literal["auto", "cpu", "gpu", "hybrid"] = "auto"
    gguf: str = ""
    """Path to the model weights (.gguf). Needed until ``asif models pull`` lands (M2)."""
    mmproj: str = ""
    """Path to the vision projector (.gguf) that goes with ``gguf``."""
    llama_server: str = "llama-server"
    """The ``llama-server`` binary from llama.cpp (name on PATH, or a full path)."""
    threads: int = Field(0, ge=0)
    """CPU threads for inference. 0 lets llama.cpp decide."""
    gpu_layers: int | None = Field(None, ge=0)
    """Layers to offload to the GPU. None follows ``device``."""
    ctx: int = Field(4096, ge=1024)
    server_args: list[str] = Field(default_factory=list)
    """Extra arguments for llama-server, for flags this version doesn't know about."""
    load_timeout_s: int = Field(300, ge=5)
    request_timeout_s: int = Field(600, ge=5)


class UploadConfig(_Section):
    verify: Literal["full", "sampled", "fingerprint"] = "sampled"
    sample_rate: float = Field(0.05, ge=0, le=1)
    max_uploads_per_hour: int = Field(300, ge=1)
    bandwidth: str = "10M"


class GuardsConfig(_Section):
    cpu_temp_limit_c: float = Field(90, gt=0)
    thermal_window_s: int = Field(300, ge=1)
    max_failure_rate: float = Field(0.02, ge=0, le=1)
    require_ac_power: bool = True


class TelemetryConfig(_Section):
    enabled: bool = True
    retention_days: int = Field(90, ge=1)


class Config(_Section):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    guards: GuardsConfig = Field(default_factory=GuardsConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


def load_config(path: Path | None = None) -> Config:
    """Load ``path`` (default: the platform config file). Missing file = defaults."""
    path = path or paths.config_file()
    if not path.exists():
        return Config()
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{path}: {problems}") from exc
