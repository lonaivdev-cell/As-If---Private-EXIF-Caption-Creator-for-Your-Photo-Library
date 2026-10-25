"""The model manifest: ``manifest.toml`` next to this file.

The manifest is the catalogue of models As If knows how to run: the §9.2
shortlist of docs/DESIGN.md, with the sizes and memory thresholds the
recommender (§10) works from. It is checked into the repository so that a
model's provenance is reviewable: every entry names its upstream repo, and
from M2 on a pinned revision and a SHA-256 per file, so a swapped or
corrupted download is refused instead of loaded.

Parsing is strict, like the config file: unknown keys, duplicate ids, bad
tiers or malformed hashes are errors, because a typo here would silently
change which file gets trusted.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MANIFEST_PATH = Path(__file__).with_name("manifest.toml")

Tier = Literal["T0", "T1", "T2", "T3"]
TIERS: tuple[Tier, ...] = ("T0", "T1", "T2", "T3")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str
    """Lowercase hex SHA-256; "" until the entry is pinned."""
    role: Literal["weights", "projector"]


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    tier: Tier
    arch: Literal["dense", "moe"]
    params: str
    """Human-readable size, e.g. "25.2B total / 3.8B active"."""
    weights_gb: float
    projector_gb: float
    min_vram_gb: float | None
    """VRAM from which this model is the GPU pick; None = not a GPU-tier model."""
    min_ram_gb: float
    languages: tuple[str, ...]
    """Primary language subtags ("en", "zh"), or ``("*",)`` for multilingual."""
    licence: str
    repo: str
    revision: str
    """Upstream commit; "" until pinned."""
    files: tuple[ModelFile, ...]
    notes: str

    @property
    def pinned(self) -> bool:
        """True once the revision and every file's name and SHA-256 are fixed."""
        return (
            bool(self.revision)
            and bool(self.files)
            and all(f.name and _HEX64.match(f.sha256) for f in self.files)
        )

    @property
    def total_gb(self) -> float:
        """Weights plus vision projector: what has to fit in memory before runtime overhead."""
        return self.weights_gb + self.projector_gb


# -- parsing -----------------------------------------------------------------
# Pydantic does the type and key checking (same approach as config.py); the
# frozen dataclasses above are what the rest of the code sees.


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _FileSpec(_Strict):
    name: str
    sha256: str = Field(pattern=r"^([0-9a-f]{64})?$")
    role: Literal["weights", "projector"]


class _EntrySpec(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str = Field(min_length=1)
    tier: Tier
    arch: Literal["dense", "moe"]
    params: str
    weights_gb: float = Field(gt=0)
    projector_gb: float = Field(ge=0)
    min_vram_gb: float | None = Field(None, gt=0)
    min_ram_gb: float = Field(gt=0)
    languages: list[str] = Field(min_length=1)
    licence: str = Field(min_length=1)
    repo: str
    revision: str
    files: list[_FileSpec]
    notes: str = ""


class _ManifestSpec(_Strict):
    model: list[_EntrySpec] = Field(min_length=1)


_LANG = re.compile(r"^(\*|[a-z]{2,3})$")


def load_manifest(path: Path | None = None) -> list[ModelEntry]:
    """Parse and validate the manifest (default: the bundled ``manifest.toml``).

    Raises ``ValueError`` naming the file and the problem: invalid TOML,
    unknown or missing keys, a bad tier, a duplicate id, or a language that
    isn't ``"*"`` or a primary subtag.
    """
    path = path or MANIFEST_PATH
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: not valid TOML: {exc}") from exc
    try:
        spec = _ManifestSpec.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ValueError(f"{path}: {problems}") from exc

    entries: list[ModelEntry] = []
    seen: set[str] = set()
    for item in spec.model:
        if item.id in seen:
            raise ValueError(f"{path}: duplicate model id {item.id!r}")
        seen.add(item.id)
        bad = [lang for lang in item.languages if not _LANG.match(lang)]
        if bad:
            raise ValueError(
                f"{path}: model {item.id!r}: languages must be '*' or primary subtags "
                f"like 'en', got {bad}"
            )
        entries.append(
            ModelEntry(
                id=item.id,
                name=item.name,
                tier=item.tier,
                arch=item.arch,
                params=item.params,
                weights_gb=item.weights_gb,
                projector_gb=item.projector_gb,
                min_vram_gb=item.min_vram_gb,
                min_ram_gb=item.min_ram_gb,
                languages=tuple(item.languages),
                licence=item.licence,
                repo=item.repo,
                revision=item.revision,
                files=tuple(ModelFile(f.name, f.sha256, f.role) for f in item.files),
                notes=item.notes,
            )
        )
    return entries


def get_model(model_id: str, manifest: Sequence[ModelEntry] | None = None) -> ModelEntry:
    """The entry with this id. Raises ``KeyError`` if the catalogue has none."""
    for entry in manifest if manifest is not None else load_manifest():
        if entry.id == model_id:
            return entry
    raise KeyError(f"no model {model_id!r} in the catalogue; see `asif models list`")


def supports_language(entry: ModelEntry, lang: str) -> bool:
    """Whether ``entry`` lists the language of the BCP-47 tag ``lang``.

    Only the primary subtag counts ("pt-BR" needs "pt"): model cards list
    languages, not regional variants.
    """
    if "*" in entry.languages:
        return True
    primary = lang.split("-", 1)[0].lower()
    return bool(primary) and primary in entry.languages
