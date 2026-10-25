"""Pick a model for this machine (§10 of docs/DESIGN.md).

On Apple Silicon unified memory decides. Elsewhere a GPU decides if it has
enough VRAM for a GPU-tier model, and RAM decides otherwise. Within that
budget the highest tier wins, because bigger models describe photos more
accurately. T0 is the exception: on a small machine every GB counts, so the
smallest model that writes the caption language wins there (MiniCPM-V 4.6
for English or Chinese, Gemma 4 E2B for everything else).

The thresholds live in the manifest (``min_vram_gb``, ``min_ram_gb``), so the
catalogue and the recommender can't drift apart. Only models that list the
caption language are ever suggested, and every recommendation says why it
was made.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from asif.models.manifest import TIERS, ModelEntry, load_manifest, supports_language

Device = Literal["cpu", "gpu", "hybrid", "metal"]

APPLE_MEMORY_SHARE = 0.65
"""macOS keeps part of unified memory for itself, so a model gets about this share (§10, §20)."""
RUNTIME_OVERHEAD_GB = 2.0
"""Runtime plus a ~4k-token context on top of weights and projector (§9.1)."""
HYBRID_MIN_RAM_GB = 32.0
"""RAM from which a 6-10 GB GPU is offered a hybrid T3 setup: experts in RAM (§9.3, §10)."""

_RANK = {tier: i for i, tier in enumerate(TIERS)}
_WHERE: dict[Device, str] = {
    "cpu": "the CPU",
    "gpu": "the GPU",
    "hybrid": "the GPU and CPU together",
    "metal": "the Apple GPU (Metal)",
}


@dataclass(frozen=True)
class Recommendation:
    model_id: str
    model_name: str
    device: Device
    reason: str
    """Always says why, e.g. "...because the GPU has 12 GB of VRAM; X would need ~19 GB"."""
    alternatives: tuple[str, ...] = ()
    """Other sensible choices, as short sentences."""


def recommend(
    *,
    ram_total_gb: float,
    vram_gb: float | None,
    apple_silicon: bool,
    unified_memory_gb: float | None,
    lang: str,
    manifest: Sequence[ModelEntry] | None = None,
) -> Recommendation:
    """Recommend a model from the catalogue for this machine and caption language.

    The branches of §10, in order: Apple Silicon (largest tier whose weights,
    projector and 2 GB of runtime fit in 65 % of unified memory, on Metal);
    a GPU with enough VRAM for a GPU-tier model (T2 from 10 GB, T1 from 6 GB,
    with a hybrid T3 offer from 32 GB of RAM); otherwise the CPU by RAM (T3
    from 28 or 48 GB, T1 from 12 GB, else T0). If the model those rules
    prefer doesn't list ``lang``, the next smaller one that does is picked
    and the reason says so. ``ram_total_gb <= 0`` means "unknown" and leads
    to the smallest tier.

    Raises ``ValueError`` if no model in the catalogue lists ``lang``.
    """
    entries = list(manifest) if manifest is not None else load_manifest()
    candidates = [m for m in entries if supports_language(m, lang)]
    if not candidates:
        raise ValueError(
            f"no model in the catalogue lists the caption language {lang!r}; "
            "see `asif models list` and set [general] caption_lang"
        )
    budget = _budget(entries, ram_total_gb, vram_gb, apple_silicon, unified_memory_gb)

    preferred = _best(budget.pool(entries))
    pick = _best(budget.pool(candidates))
    if pick is None:
        # Nothing this machine can hold speaks the language: the next smaller
        # model that does, or failing that the smallest one there is.
        smaller = [m for m in candidates if preferred and _size(m) < _size(preferred)]
        pick = max(smaller, key=_size) if smaller else min(candidates, key=_size)

    first = f"Picked {pick.name} ({pick.tier}) to run on {_WHERE[budget.device]} because "
    first += budget.because
    if upgrade := budget.upgrade(pick, candidates):
        first += f"; {upgrade}"
    sentences = [first + "."]
    if preferred is not None and preferred != pick and not supports_language(preferred, lang):
        sentences.append(
            f"{preferred.name} would be the first choice, but it only lists "
            f"{', '.join(preferred.languages)} and captions are set to {lang}."
        )
    if shortfall := budget.shortfall(pick):
        sentences.append(shortfall)

    return Recommendation(
        model_id=pick.id,
        model_name=pick.name,
        device=budget.device,
        reason=" ".join(sentences),
        alternatives=_alternatives(pick, budget, candidates, ram_total_gb),
    )


# -- the budget: which device, how much memory, what fits --------------------


@dataclass(frozen=True)
class _Budget:
    device: Device
    have_gb: float
    """Memory that limits this device: RAM, VRAM or unified memory (0 = unknown)."""
    unit: str
    because: str

    def need_gb(self, m: ModelEntry) -> float:
        """Memory ``m`` needs here. Manifest thresholds where they exist, else an estimate."""
        if self.device == "metal":
            return (m.total_gb + RUNTIME_OVERHEAD_GB) / APPLE_MEMORY_SHARE
        if self.device == "gpu":
            return m.min_vram_gb or m.total_gb + RUNTIME_OVERHEAD_GB
        return m.min_ram_gb

    def eligible(self, m: ModelEntry) -> bool:
        if self.device == "gpu":
            # Only GPU-tier models are picked for the GPU (§10); MoE goes hybrid.
            return m.min_vram_gb is not None and m.min_vram_gb <= self.have_gb
        if self.device == "cpu" and m.tier == "T2":
            # T2 is the GPU tier: a dense 9.6B is slower on CPU than a 3-4B-active MoE.
            return False
        return self.need_gb(m) <= self.have_gb

    def pool(self, entries: Sequence[ModelEntry]) -> list[ModelEntry]:
        fitting = [m for m in entries if self.eligible(m)]
        if fitting or self.device != "cpu":
            return fitting
        return [m for m in entries if m.tier == "T0"]  # §10: "else: T0"

    def upgrade(self, pick: ModelEntry, candidates: Sequence[ModelEntry]) -> str:
        """What the next bigger model would need ("" when the budget is unknown)."""
        if self.have_gb <= 0:
            return ""
        bigger = [
            m
            for m in candidates
            if _size(m) > _size(pick)
            and not self.eligible(m)
            and not (self.device == "cpu" and m.tier == "T2")
        ]
        if not bigger:
            return "it's the biggest model in the catalogue for this setup"
        step = min(bigger, key=_size)
        exact = self.device == "cpu" or (self.device == "gpu" and step.min_vram_gb is not None)
        need = _num(self.need_gb(step)) if exact else f"~{math.ceil(self.need_gb(step))}"
        return f"{step.name} would need {need} GB of {self.unit}"

    def shortfall(self, pick: ModelEntry) -> str:
        if self.have_gb <= 0 or self.need_gb(pick) <= self.have_gb:
            return ""
        return (
            f"It's sized for {_num(self.need_gb(pick))} GB of {self.unit}, more than this "
            "machine has, so expect it to be slow."
        )


def _budget(
    entries: Sequence[ModelEntry],
    ram_gb: float,
    vram_gb: float | None,
    apple_silicon: bool,
    unified_gb: float | None,
) -> _Budget:
    if apple_silicon:
        memory = unified_gb or ram_gb
        because = (
            f"this Mac has {_num(memory)} GB of unified memory"
            if memory > 0
            else "the amount of unified memory couldn't be detected"
        )
        return _Budget("metal", memory, "unified memory", because)
    gpu_thresholds = [m.min_vram_gb for m in entries if m.min_vram_gb is not None]
    if vram_gb and gpu_thresholds and vram_gb >= min(gpu_thresholds):
        return _Budget("gpu", vram_gb, "VRAM", f"the GPU has {_num(vram_gb)} GB of VRAM")

    if ram_gb <= 0:
        because = "the amount of RAM couldn't be detected, so the smallest tier is the safe bet"
    else:
        because = f"this machine has {_num(ram_gb)} GB of RAM"
        if vram_gb and gpu_thresholds:
            because += (
                f" and the GPU's {_num(vram_gb)} GB of VRAM is less than the "
                f"{_num(min(gpu_thresholds))} GB a GPU-tier model needs"
            )
        elif gpu_thresholds:
            because += " and no GPU it can use"
    return _Budget("cpu", ram_gb, "RAM", because)


def _alternatives(
    pick: ModelEntry, budget: _Budget, candidates: Sequence[ModelEntry], ram_gb: float
) -> tuple[str, ...]:
    alts: list[str] = []
    if budget.device == "gpu" and pick.tier == "T1" and ram_gb >= HYBRID_MIN_RAM_GB:
        moe = _best([m for m in candidates if m.tier == "T3" and m.min_ram_gb <= ram_gb])
        if moe is not None:
            alts.append(
                f"{moe.name} ({moe.id}) in hybrid mode: attention on the GPU, experts in "
                f"your {_num(ram_gb)} GB of RAM. More knowledge, slower per photo."
            )
    lower = [
        m
        for m in candidates
        if _RANK[m.tier] < _RANK[pick.tier] and not (budget.device == "cpu" and m.tier == "T2")
    ]
    if down := _best(lower):
        alts.append(
            f"{down.name} ({down.id}), one tier down: faster but less accurate, "
            "if a benchmark says the full library would take too long."
        )
    return tuple(alts)


# -- helpers -------------------------------------------------------------------


def _size(m: ModelEntry) -> tuple[int, float]:
    return (_RANK[m.tier], m.total_gb)


def _best(pool: Sequence[ModelEntry]) -> ModelEntry | None:
    """Highest tier; within it the biggest model, except in T0 where the smallest wins."""
    if not pool:
        return None
    top = max(_RANK[m.tier] for m in pool)
    tier = [m for m in pool if _RANK[m.tier] == top]
    if top == 0:
        return min(tier, key=lambda m: m.total_gb)
    return max(tier, key=lambda m: m.total_gb)


def _num(gb: float) -> str:
    """12.0 -> "12", 15.6 -> "15.6": whole numbers stay whole."""
    return f"{gb:.0f}" if abs(gb - round(gb)) < 0.05 else f"{gb:.1f}"
