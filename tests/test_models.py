"""The model catalogue (§9.2) and the recommender (§10)."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import pytest

from asif.models.manifest import (
    MANIFEST_PATH,
    ModelEntry,
    ModelFile,
    get_model,
    load_manifest,
    supports_language,
)
from asif.models.recommend import Recommendation, recommend

IDS = [
    "minicpm-v-4.6-q4",
    "gemma-4-e2b-q4",
    "gemma-4-e4b-q4",
    "qwen3.5-9b-q4km",
    "gemma-4-26b-a4b-q4",
    "qwen3.5-35b-a3b-q4",
]

# (id, name, tier, arch, weights_gb, projector_gb) straight from §9.2.
DESIGN_TABLE = [
    ("minicpm-v-4.6-q4", "MiniCPM-V 4.6", "T0", "dense", 1.6, 0.0),
    ("gemma-4-e2b-q4", "Gemma 4 E2B", "T0", "dense", 3.1, 0.0),
    ("gemma-4-e4b-q4", "Gemma 4 E4B", "T1", "dense", 5.0, 0.0),
    ("qwen3.5-9b-q4km", "Qwen3.5-9B", "T2", "dense", 6.0, 0.9),
    ("gemma-4-26b-a4b-q4", "Gemma 4 26B-A4B", "T3", "moe", 17.0, 0.0),
    ("qwen3.5-35b-a3b-q4", "Qwen3.5-35B-A3B", "T3", "moe", 20.0, 0.86),
]


def entry(**overrides: object) -> ModelEntry:
    """A valid entry to tweak in tests."""
    base = ModelEntry(
        id="m",
        name="M",
        tier="T0",
        arch="dense",
        params="1B",
        weights_gb=1.0,
        projector_gb=0.0,
        min_vram_gb=None,
        min_ram_gb=8.0,
        languages=("*",),
        licence="Apache-2.0",
        repo="",
        revision="",
        files=(),
        notes="",
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


# -- the bundled manifest ------------------------------------------------------


def test_bundled_manifest_has_the_six_shortlisted_models_in_order() -> None:
    assert [m.id for m in load_manifest()] == IDS


@pytest.mark.parametrize(("model_id", "name", "tier", "arch", "weights", "projector"), DESIGN_TABLE)
def test_sizes_tiers_and_architecture_match_the_design(
    model_id: str, name: str, tier: str, arch: str, weights: float, projector: float
) -> None:
    m = get_model(model_id)
    assert (m.name, m.tier, m.arch) == (name, tier, arch)
    assert m.weights_gb == pytest.approx(weights)
    assert m.projector_gb == pytest.approx(projector)
    assert m.total_gb == pytest.approx(weights + projector)


def test_params_text_matches_the_design() -> None:
    params = {m.id: m.params for m in load_manifest()}
    assert params["gemma-4-26b-a4b-q4"] == "25.2B total / 3.8B active"
    assert params["qwen3.5-35b-a3b-q4"] == "35B total / 3B active"
    assert params["qwen3.5-9b-q4km"] == "9.6B"
    assert "SigLIP2-400M" in params["minicpm-v-4.6-q4"]


def test_languages_minicpm_is_english_and_chinese_only() -> None:
    for m in load_manifest():
        expected = ("en", "zh") if m.id == "minicpm-v-4.6-q4" else ("*",)
        assert m.languages == expected, m.id


def test_every_model_is_apache_licensed() -> None:
    assert {m.licence for m in load_manifest()} == {"Apache-2.0"}


def test_repos_only_where_the_design_names_one() -> None:
    repos = {m.id: m.repo for m in load_manifest()}
    assert repos.pop("qwen3.5-9b-q4km") == "unsloth/Qwen3.5-9B-GGUF"
    assert repos.pop("minicpm-v-4.6-q4") == "ggml-org/MiniCPM-V-4.6-GGUF"
    assert set(repos.values()) == {""}


def test_nothing_is_pinned_and_nothing_is_invented() -> None:
    for m in load_manifest():
        assert not m.pinned, m.id
        assert m.revision == "", m.id
        assert {f.role for f in m.files} == {"weights", "projector"}, m.id
        assert all(f.name == "" and f.sha256 == "" for f in m.files), m.id


def test_the_comment_says_when_pins_get_filled_in() -> None:
    assert MANIFEST_PATH.read_text().count("filled in when pinning (M2)") >= 6


def test_thresholds_match_section_10_and_the_section_17_example() -> None:
    m = {e.id: e for e in load_manifest()}
    assert (m["qwen3.5-9b-q4km"].min_vram_gb, m["qwen3.5-9b-q4km"].min_ram_gb) == (10, 12)
    assert m["gemma-4-e4b-q4"].min_vram_gb == 6
    assert m["gemma-4-e4b-q4"].min_ram_gb == 12
    # RAM thresholds sit below the nominal size, because Linux reports less than
    # is installed: 7 for "8 GB", 12 for "16 GB", 28 for "32 GB", 44 for "48 GB".
    assert m["minicpm-v-4.6-q4"].min_ram_gb == m["gemma-4-e2b-q4"].min_ram_gb == 7
    assert m["gemma-4-26b-a4b-q4"].min_ram_gb == 28
    assert m["qwen3.5-35b-a3b-q4"].min_ram_gb == 44
    # Only the GPU tiers have a VRAM threshold.
    assert {e.id for e in m.values() if e.min_vram_gb is not None} == {
        "gemma-4-e4b-q4",
        "qwen3.5-9b-q4km",
    }


# -- parsing and validation ----------------------------------------------------

GOOD = """
[[model]]
id = "x-q4"
name = "X"
tier = "T1"
arch = "moe"
params = "10B total / 1B active"
weights_gb = 4
projector_gb = 0.5
min_vram_gb = 6
min_ram_gb = 12
languages = ["en", "de"]
licence = "Apache-2.0"
repo = "org/X-GGUF"
revision = "{rev}"
files = [
    {{ role = "weights", name = "x.Q4_K_M.gguf", sha256 = "{sha}" }},
    {{ role = "projector", name = "mmproj-x-F16.gguf", sha256 = "{sha}" }},
]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "manifest.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_fully_pinned_entry_parses(tmp_path: Path) -> None:
    sha = "ab" * 32
    [m] = load_manifest(write(tmp_path, GOOD.format(rev="0123abc", sha=sha)))
    assert m.pinned
    assert m.weights_gb == 4.0  # TOML integers are fine for sizes
    assert m.languages == ("en", "de")
    assert m.notes == ""  # optional
    assert m.files == (
        ModelFile("x.Q4_K_M.gguf", sha, "weights"),
        ModelFile("mmproj-x-F16.gguf", sha, "projector"),
    )


def test_min_vram_is_optional(tmp_path: Path) -> None:
    text = GOOD.format(rev="", sha="").replace("min_vram_gb = 6\n", "")
    [m] = load_manifest(write(tmp_path, text))
    assert m.min_vram_gb is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.replace('tier = "T1"', 'tier = "T4"'), "tier"),
        (lambda t: t.replace('arch = "moe"', 'arch = "sparse"'), "arch"),
        (
            lambda t: t.replace(
                'licence = "Apache-2.0"\n', 'licence = "Apache-2.0"\nlicense = "x"\n'
            ),
            "license",
        ),
        (lambda t: t.replace('role = "weights",', 'role = "weights", size = 1,'), "size"),
        (lambda t: "version = 1\n" + t, "version"),
        (lambda t: t.replace('params = "10B total / 1B active"\n', ""), "params"),
        (lambda t: t.replace("weights_gb = 4", 'weights_gb = "4"'), "weights_gb"),
        (lambda t: t.replace("weights_gb = 4", "weights_gb = 0"), "weights_gb"),
        (lambda t: t.replace('role = "weights"', 'role = "tokenizer"'), "role"),
        (lambda t: t.replace('["en", "de"]', '["pt-BR"]'), "languages"),
        (lambda t: t.replace('["en", "de"]', '["English"]'), "languages"),
        (lambda t: t.replace('["en", "de"]', "[]"), "languages"),
        (lambda t: t.replace('id = "x-q4"', 'id = "X Q4"'), "id"),
        (lambda t: t + t, "duplicate model id 'x-q4'"),
        (lambda t: t.replace("[[model]]", "[model"), "not valid TOML"),
        (lambda t: "", "model"),
    ],
)
def test_bad_manifests_are_refused(tmp_path: Path, change: object, message: str) -> None:
    assert callable(change)
    text = change(GOOD.format(rev="", sha=""))
    with pytest.raises(ValueError, match=r"manifest\.toml") as excinfo:
        load_manifest(write(tmp_path, text))
    assert message in str(excinfo.value)


@pytest.mark.parametrize("sha", ["AB" * 32, "ab" * 31, "zz" * 32, "ab" * 32 + " "])
def test_malformed_hashes_are_refused(tmp_path: Path, sha: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        load_manifest(write(tmp_path, GOOD.format(rev="r", sha=sha)))


def test_pinned_needs_revision_and_every_file_named_and_hashed() -> None:
    sha = "0f" * 32
    ok = (ModelFile("w.gguf", sha, "weights"), ModelFile("p.gguf", sha, "projector"))
    assert entry(revision="abc", files=ok).pinned
    assert not entry(revision="", files=ok).pinned
    assert not entry(revision="abc", files=()).pinned
    assert not entry(revision="abc", files=(ok[0], ModelFile("p.gguf", "", "projector"))).pinned
    assert not entry(revision="abc", files=(ok[0], ModelFile("", sha, "projector"))).pinned


def test_get_model() -> None:
    assert get_model("qwen3.5-9b-q4km").name == "Qwen3.5-9B"
    custom = [entry(id="only")]
    assert get_model("only", custom) is custom[0]
    with pytest.raises(KeyError, match="asif models list"):
        get_model("qwen3.5-9b-q4km", custom)
    with pytest.raises(KeyError):
        get_model("llama-4-scout")  # excluded on purpose (§9.2)


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("en", True),
        ("en-GB", True),
        ("EN-us", True),
        ("zh", True),
        ("zh-Hant-TW", True),
        ("pt-BR", False),
        ("pt", False),
        ("es-419", False),
        ("", False),
    ],
)
def test_supports_language_uses_the_primary_subtag(lang: str, expected: bool) -> None:
    assert supports_language(get_model("minicpm-v-4.6-q4"), lang) is expected


@pytest.mark.parametrize("lang", ["en", "pt-BR", "ja", "zh-Hant-TW", "es-419"])
def test_star_means_multilingual(lang: str) -> None:
    assert supports_language(get_model("gemma-4-e2b-q4"), lang)


# -- recommend: every branch of §10 --------------------------------------------


def rec(
    *,
    ram: float = 16,
    vram: float | None = None,
    apple: bool = False,
    unified: float | None = None,
    lang: str = "en",
    manifest: list[ModelEntry] | None = None,
    gpu_detected: bool = False,
) -> Recommendation:
    return recommend(
        ram_total_gb=ram,
        vram_gb=vram,
        apple_silicon=apple,
        unified_memory_gb=unified,
        lang=lang,
        manifest=manifest,
        gpu_detected=gpu_detected,
    )


def memtotal_gb(kb: int) -> float:
    """/proc/meminfo's MemTotal as the doctor hands it over: kB -> MiB -> GiB."""
    return kb // 1024 / 1024


@pytest.mark.parametrize(
    ("unified", "lang", "expected"),
    [
        (128, "en", "qwen3.5-35b-a3b-q4"),
        (64, "en", "qwen3.5-35b-a3b-q4"),
        (32, "en", "gemma-4-26b-a4b-q4"),  # 35B-A3B needs (20.86 + 2) / 0.65 = ~35 GB
        (24, "en", "qwen3.5-9b-q4km"),
        (16, "en", "qwen3.5-9b-q4km"),
        (8, "en", "minicpm-v-4.6-q4"),
        (8, "pt-BR", "gemma-4-e2b-q4"),
        (8, "zh-Hans", "minicpm-v-4.6-q4"),
    ],
)
def test_apple_silicon_picks_the_largest_tier_that_fits_65_percent(
    unified: float, lang: str, expected: str
) -> None:
    r = rec(apple=True, unified=unified, ram=unified, lang=lang)
    assert (r.model_id, r.device) == (expected, "metal")
    assert f"{unified} GB of unified memory" in r.reason


def test_apple_silicon_beats_the_gpu_branch_and_falls_back_to_ram() -> None:
    r = rec(apple=True, unified=None, ram=32, vram=12)
    assert (r.model_id, r.device) == ("gemma-4-26b-a4b-q4", "metal")
    assert "Qwen3.5-35B-A3B would need ~36 GB of unified memory" in r.reason


@pytest.mark.parametrize("vram", [10, 12, 16])
def test_10_gb_of_vram_or_more_means_t2_on_the_gpu(vram: float) -> None:
    r = rec(vram=vram, ram=16)
    assert (r.model_id, r.device) == ("qwen3.5-9b-q4km", "gpu")
    assert f"the GPU has {vram} GB of VRAM" in r.reason
    assert "Gemma 4 26B-A4B would need ~19 GB of VRAM" in r.reason
    assert any("gemma-4-e4b-q4" in alt and "one tier down" in alt for alt in r.alternatives)


@pytest.mark.parametrize("vram", [19, 20, 24, 48, 80])
def test_a_big_gpu_is_not_told_the_next_model_needs_less_vram_than_it_has(vram: float) -> None:
    r = rec(vram=vram, ram=16)
    assert (r.model_id, r.device) == ("qwen3.5-9b-q4km", "gpu")
    assert f"the GPU has {vram} GB of VRAM" in r.reason
    assert "would need" not in r.reason
    assert "biggest GPU-tier model" in r.reason and "Gemma 4 26B-A4B would fit" in r.reason


@pytest.mark.parametrize("vram", [6, 8, 9.9])
def test_6_to_10_gb_of_vram_means_t1_on_the_gpu(vram: float) -> None:
    r = rec(vram=vram, ram=16)
    assert (r.model_id, r.device) == ("gemma-4-e4b-q4", "gpu")
    assert "Qwen3.5-9B would need 10 GB of VRAM" in r.reason
    assert not any("hybrid" in alt for alt in r.alternatives)  # 16 GB RAM: no hybrid offer


@pytest.mark.parametrize(
    ("vram", "expected"),
    [
        (6141 / 1024, "gemma-4-e4b-q4"),  # a 6 GB RTX 4050 Laptop, as nvidia-smi reports it
        (6 - 16 / 1024, "gemma-4-e4b-q4"),  # a 6 GB AMD card, 16 MiB short like the 7900 XTX
        (8188 / 1024, "gemma-4-e4b-q4"),  # an 8 GB RTX 4060
        (10 - 16 / 1024, "qwen3.5-9b-q4km"),  # a 10 GB AMD card
        (24564 / 1024, "qwen3.5-9b-q4km"),  # an RTX 4090
    ],
)
def test_vram_a_few_mib_short_of_a_whole_gb_counts_as_that_gb(vram: float, expected: str) -> None:
    r = rec(vram=vram, ram=16)
    assert (r.model_id, r.device) == (expected, "gpu")
    assert f"the GPU has {round(vram)} GB of VRAM" in r.reason


@pytest.mark.parametrize(("vram", "shown"), [(5.5, "5.5"), (5.9, "5.9"), (5.93, "5.9")])
def test_vram_well_short_of_6_gb_still_means_the_cpu(vram: float, shown: str) -> None:
    r = rec(vram=vram, ram=16)
    assert r.device == "cpu"
    assert f"GPU's {shown} GB of VRAM is less than the 6 GB" in r.reason


def test_the_vram_reason_never_contradicts_itself() -> None:
    manifest = load_manifest()
    for mib in range(5 * 1024, 26 * 1024, 3):
        reason = rec(vram=mib / 1024, ram=16, manifest=manifest).reason
        assert "GPU's 6 GB of VRAM is less than the 6 GB" not in reason, mib
        assert "has 10 GB of VRAM; Qwen3.5-9B would need 10 GB" not in reason, mib
        assert "would need ~19 GB" not in reason or mib / 1024 < 19, mib


@pytest.mark.parametrize(
    ("ram", "moe"),
    [
        (28, "gemma-4-26b-a4b-q4"),
        (memtotal_gb(32803612), "gemma-4-26b-a4b-q4"),  # a "32 GB" machine as Linux reports it
        (40, "gemma-4-26b-a4b-q4"),
        (memtotal_gb(49250000), "qwen3.5-35b-a3b-q4"),  # a "48 GB" machine
        (64, "qwen3.5-35b-a3b-q4"),
    ],
)
def test_6_gb_gpu_with_32_gb_ram_is_offered_hybrid_t3(ram: float, moe: str) -> None:
    r = rec(vram=8, ram=ram)
    assert (r.model_id, r.device) == ("gemma-4-e4b-q4", "gpu")
    hybrid = [alt for alt in r.alternatives if "hybrid" in alt]
    assert len(hybrid) == 1 and moe in hybrid[0]


def test_no_hybrid_offer_below_what_a_32_gb_machine_reports() -> None:
    assert not any("hybrid" in alt for alt in rec(vram=8, ram=27.9).alternatives)
    assert not any("hybrid" in alt for alt in rec(vram=8, ram=0).alternatives)  # unknown RAM


@pytest.mark.parametrize(
    ("ram", "expected"),
    [
        (256, "qwen3.5-35b-a3b-q4"),
        (46.9, "qwen3.5-35b-a3b-q4"),  # a "48 GB" machine as Linux reports it
        (44, "qwen3.5-35b-a3b-q4"),
        (43.9, "gemma-4-26b-a4b-q4"),
        (31.3, "gemma-4-26b-a4b-q4"),  # a "32 GB" machine
        (28, "gemma-4-26b-a4b-q4"),
        (27.9, "gemma-4-e4b-q4"),
        (15.6, "gemma-4-e4b-q4"),  # a "16 GB" machine
        (12, "gemma-4-e4b-q4"),
        (11.9, "minicpm-v-4.6-q4"),
        (8, "minicpm-v-4.6-q4"),
        (7.6, "minicpm-v-4.6-q4"),  # an "8 GB" machine
        (4, "minicpm-v-4.6-q4"),
    ],
)
def test_cpu_branches_by_ram(ram: float, expected: str) -> None:
    r = rec(ram=ram)
    assert (r.model_id, r.device) == (expected, "cpu")
    assert f"{ram:g} GB of RAM" in r.reason


@pytest.mark.parametrize(
    ("kb", "lang", "expected"),
    [
        (7990140, "en", "minicpm-v-4.6-q4"),
        (7849000, "en", "minicpm-v-4.6-q4"),  # 8 GB with some of it kept for an iGPU
        (7990140, "pt-BR", "gemma-4-e2b-q4"),
    ],
)
def test_an_8_gb_machine_is_not_told_its_t0_model_will_be_slow(
    kb: int, lang: str, expected: str
) -> None:
    r = rec(ram=memtotal_gb(kb), lang=lang)
    assert (r.model_id, r.device) == (expected, "cpu")
    assert "slow" not in r.reason
    assert "Gemma 4 E4B would need 12 GB of RAM" in r.reason


@pytest.mark.parametrize("vram", [None, 0, 4, 5.9])
def test_a_small_or_missing_gpu_means_the_cpu(vram: float | None) -> None:
    r = rec(vram=vram, ram=16)
    assert (r.model_id, r.device) == ("gemma-4-e4b-q4", "cpu")
    if vram:
        assert f"GPU's {vram:g} GB of VRAM is less than the 6 GB" in r.reason
    else:
        assert "no GPU was detected" in r.reason


def test_a_gpu_whose_vram_is_unknown_is_not_called_missing() -> None:
    r = rec(vram=None, ram=16, gpu_detected=True)
    assert (r.model_id, r.device) == ("gemma-4-e4b-q4", "cpu")
    assert "a GPU whose VRAM couldn't be read" in r.reason
    assert "no GPU" not in r.reason


def test_t2_is_never_the_cpu_pick() -> None:
    for ram in (12, 16, 20, 27):
        assert rec(ram=ram).model_id != "qwen3.5-9b-q4km"
    # ...and the "next step up" on a CPU skips it too.
    assert "Gemma 4 26B-A4B would need 28 GB of RAM" in rec(ram=16).reason


def test_the_reason_names_the_next_step_up_or_says_it_is_the_biggest() -> None:
    assert "Gemma 4 E4B would need 12 GB of RAM" in rec(ram=8).reason
    assert "Qwen3.5-35B-A3B would need 44 GB of RAM" in rec(ram=32).reason
    assert "biggest model in the catalogue" in rec(ram=64).reason


def test_a_bigger_t0_is_never_the_step_up() -> None:
    # In T0 the smallest model wins, so more RAM would never get English Gemma 4 E2B.
    r = rec(ram=4, lang="en")
    assert r.model_id == "minicpm-v-4.6-q4"
    assert "Gemma 4 E2B" not in r.reason
    assert "Gemma 4 E4B would need 12 GB of RAM" in r.reason


def test_pt_br_on_a_small_machine_never_gets_minicpm() -> None:
    r = rec(ram=8, lang="pt-BR")
    assert r.model_id == "gemma-4-e2b-q4"
    assert "MiniCPM-V 4.6 would be the first choice" in r.reason
    assert "en, zh" in r.reason and "pt-BR" in r.reason


def test_english_on_a_small_machine_gets_minicpm_without_a_language_note() -> None:
    r = rec(ram=8, lang="en-GB")
    assert r.model_id == "minicpm-v-4.6-q4"
    assert "first choice" not in r.reason


def test_no_language_note_when_the_language_never_mattered() -> None:
    assert "first choice" not in rec(ram=64, lang="pt-BR").reason


def test_too_little_ram_says_it_will_be_slow() -> None:
    r = rec(ram=4, lang="pt-BR")
    assert r.model_id == "gemma-4-e2b-q4"
    assert "sized for 7 GB of RAM" in r.reason and "slow" in r.reason


def test_unknown_ram_picks_the_smallest_tier_and_says_so() -> None:
    r = rec(ram=0)
    assert (r.model_id, r.device) == ("minicpm-v-4.6-q4", "cpu")
    assert "couldn't be detected" in r.reason
    assert "slow" not in r.reason  # no shortfall claim without a number


def test_one_tier_down_is_offered_except_at_the_bottom() -> None:
    assert any("one tier down" in alt for alt in rec(ram=16).alternatives)
    assert not any("one tier down" in alt for alt in rec(ram=8).alternatives)


MACHINES = [
    {"ram": ram, "vram": vram, "apple": apple, "unified": ram if apple else None}
    for ram, vram, apple in itertools.product(
        [0, 4, 8, 11.9, 12, 16, 27.9, 28, 32, 48, 64, 128],
        [None, 4, 6, 8, 10, 24],
        [False, True],
    )
]


@pytest.mark.parametrize("lang", ["en", "pt-BR", "zh-Hant-TW", "ja", "de-CH"])
def test_never_recommends_a_model_that_does_not_speak_the_language(lang: str) -> None:
    for machine in MACHINES:
        r = rec(lang=lang, **machine)  # type: ignore[arg-type]
        m = get_model(r.model_id)
        assert supports_language(m, lang), (machine, r)
        assert r.model_name == m.name
        assert r.reason.startswith(f"Picked {m.name} ({m.tier}) to run on ")
        assert " because " in r.reason
        assert r.reason.endswith(".")


def test_recommendation_is_frozen() -> None:
    r = rec()
    assert isinstance(r.alternatives, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.model_id = "x"  # type: ignore[misc]


# -- recommend: language fallback with other catalogues ------------------------


def catalogue(*, english_only: set[str]) -> list[ModelEntry]:
    return [
        dataclasses.replace(m, languages=("en",)) if m.id in english_only else m
        for m in load_manifest()
    ]


def test_falls_back_to_the_next_smaller_t3_and_says_why() -> None:
    r = rec(ram=64, lang="de", manifest=catalogue(english_only={"qwen3.5-35b-a3b-q4"}))
    assert r.model_id == "gemma-4-26b-a4b-q4"
    assert "Qwen3.5-35B-A3B would be the first choice, but it only lists en" in r.reason
    assert "captions are set to de" in r.reason


def test_falls_back_from_t2_to_t1_on_the_gpu() -> None:
    r = rec(vram=12, lang="fr", manifest=catalogue(english_only={"qwen3.5-9b-q4km"}))
    assert (r.model_id, r.device) == ("gemma-4-e4b-q4", "gpu")
    assert "Qwen3.5-9B would be the first choice" in r.reason


def test_falls_back_below_the_gpu_tiers_when_none_of_them_speaks_the_language() -> None:
    manifest = catalogue(english_only={"qwen3.5-9b-q4km", "gemma-4-e4b-q4"})
    r = rec(vram=12, lang="fr", manifest=manifest)
    assert r.model_id == "gemma-4-e2b-q4"  # the biggest one smaller than Qwen3.5-9B
    assert supports_language(get_model(r.model_id, manifest), "fr")
    assert "Qwen3.5-9B would be the first choice" in r.reason


def test_when_nothing_smaller_speaks_it_the_smallest_that_does_is_used() -> None:
    manifest = catalogue(english_only={"minicpm-v-4.6-q4", "gemma-4-e2b-q4"})
    r = rec(ram=8, lang="fr", manifest=manifest)
    assert r.model_id == "gemma-4-e4b-q4"
    assert "sized for 12 GB of RAM" in r.reason and "slow" in r.reason


def test_raises_when_no_model_speaks_the_language() -> None:
    manifest = catalogue(english_only=set(IDS))
    with pytest.raises(ValueError, match="'pt-BR'"):
        rec(lang="pt-BR", manifest=manifest)
    assert rec(lang="en", manifest=manifest).model_id == "gemma-4-e4b-q4"
