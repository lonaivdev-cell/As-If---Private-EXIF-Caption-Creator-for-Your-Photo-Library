"""The additive-only writer: the pure planning rules first, then real
exiftool round trips on private copies of the synthetic fixtures."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from asif.infer.schema import CAPTION_MAX, CaptionResult
from asif.meta.exiftool import ExecResult, ExifTool, ExifToolError
from asif.meta.writer import (
    IPTC_KEYWORD_MAX_BYTES,
    ExistingMeta,
    Provenance,
    WritePlan,
    WriteRecord,
    apply_write,
    build_args,
    plan_write,
    read_existing,
)
from tests.conftest import HAVE_EXIFTOOL

HASH = "ab" * 32
OTHER_HASH = "cd" * 32
PROV = Provenance(
    tool_version="0.1",
    model_id="stub",
    model_sha256="",
    prompt_version="caption_v1",
    run_id="r_test",
    timestamp="2026:10:25 12:00:00Z",
)
PROV2 = Provenance(
    tool_version="0.2",
    model_id="qwen3.5-9b-q4km",
    model_sha256="ef" * 32,
    prompt_version="caption_v1",
    run_id="r_second",
    timestamp="2026:10:26 08:30:00Z",
)

# Model output that tries everything: option injection, line breaks, argfile
# comments, exiftool's redirection and interpolation syntax, bidi controls,
# characters XML forbids, and sheer length.
HOSTILE = [
    "line one\nline two",
    "-if",
    "-o /etc/x",
    "-execute",
    "x\n-execute\n",
    "x\r\n-o\r\n/etc/x",
    "nul\0byte",
    "\u202eevil\u202c bidi \u2066x\u2069",
    "#[CSTR]x\\n",
    "=",
    "<=foo",
    "${filename}",
    "-XMP-dc:Subject=",
    "-tagsFromFile @",
    "\u2028sep\u2029",
    "\ufffe\uffff",
    "a" * 300,
]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def result(
    caption: str = "A red shed in a garden", keywords: Sequence[str] = ("garden", "shed", "sky")
) -> CaptionResult:
    return CaptionResult(
        caption=caption, keywords=tuple(keywords), has_text=False, people_count=0, setting="outdoor"
    )


def raw_result(caption: str, keywords: Sequence[str]) -> CaptionResult:
    """Skip validation and cleaning, as a buggy caller might."""
    return CaptionResult.model_construct(
        caption=caption, keywords=tuple(keywords), has_text=False, people_count=0, setting="unclear"
    )


def ours(caption: str, lang: str = "en", **asif: Any) -> ExistingMeta:
    """A file As If captioned earlier that nobody has touched since."""
    return ExistingMeta(
        descriptions={"x-default": caption, lang: caption},
        asif={"CaptionDigest": sha(caption), "CaptionLang": lang, **asif},
    )


def plan(
    existing: ExistingMeta | None = None,
    res: CaptionResult | None = None,
    *,
    lang: str = "en",
    compat: str = "xmp",
) -> WritePlan:
    return plan_write(
        existing or ExistingMeta(),
        res or result(),
        lang=lang,
        compat=compat,  # type: ignore[arg-type]
        image_hash=HASH,
    )


BASE_PLAN = WritePlan(
    caption=None,
    caption_reason="kept-user",
    remove_caption_langs=(),
    add_subjects=(),
    add_iptc=(),
    create_iptc=False,
    iptc_note=None,
    write_original_hash=False,
    image_hash=HASH,
    all_added_subjects=(),
    all_added_iptc=(),
    wrote_original_hash=False,
    caption_digest=None,
    caption_lang=None,
)


# -- plan_write: captions ------------------------------------------------------


def test_empty_file_gets_everything() -> None:
    p = plan()
    assert p.caption == "A red shed in a garden"
    assert p.caption_reason == "empty"
    assert p.remove_caption_langs == ()
    assert p.add_subjects == ("garden", "shed", "sky")
    assert p.add_iptc == () and not p.create_iptc and p.iptc_note is None
    assert p.write_original_hash and p.wrote_original_hash
    assert p.image_hash == HASH
    assert p.all_added_subjects == ("garden", "shed", "sky")
    assert p.all_added_iptc == ()
    assert p.caption_digest == sha("A red shed in a garden")
    assert p.caption_lang == "en"


def test_blank_descriptions_hold_nothing_to_protect() -> None:
    p = plan(ExistingMeta(descriptions={"x-default": "", "fr": "  "}))
    assert p.caption_reason == "empty"
    assert p.caption is not None


@pytest.mark.parametrize(
    "descriptions",
    [
        {"x-default": "Grandma's garden, summer 2003"},
        {"fr": "Le jardin de mamie"},
        {"": "a description with xml:lang=''"},
    ],
)
def test_owner_caption_is_kept(descriptions: dict[str, str]) -> None:
    p = plan(ExistingMeta(descriptions=descriptions))
    assert p.caption is None
    assert p.caption_reason == "kept-user"
    assert p.caption_digest is None and p.caption_lang is None
    assert p.remove_caption_langs == ()
    # Keywords and provenance still go in: only the caption is held back.
    assert p.add_subjects == ("garden", "shed", "sky")


def test_own_untouched_caption_is_replaced() -> None:
    p = plan(ours("An old caption"), result("A new caption"))
    assert p.caption == "A new caption"
    assert p.caption_reason == "ours"
    assert p.remove_caption_langs == ()
    assert p.caption_digest == sha("A new caption")
    assert p.caption_lang == "en"


def test_own_caption_with_only_x_default_is_ours() -> None:
    existing = ExistingMeta(descriptions={"x-default": "Old"}, asif={"CaptionDigest": sha("Old")})
    assert plan(existing).caption_reason == "ours"


def test_language_change_removes_only_the_old_own_language() -> None:
    p = plan(ours("An old caption", "en"), result("Uma legenda nova"), lang="pt-BR")
    assert p.caption_reason == "ours"
    assert p.remove_caption_langs == ("en",)
    assert p.caption_lang == "pt-BR"


def test_language_tags_compare_case_insensitively() -> None:
    # exiftool stores "pt-br" as "pt-BR"; the record keeps what the config said.
    existing = ExistingMeta(
        descriptions={"x-default": "Old", "pt-BR": "Old"},
        asif={"CaptionDigest": sha("Old"), "CaptionLang": "pt-br"},
    )
    p = plan(existing, lang="PT-BR")
    assert p.caption_reason == "ours"
    assert p.remove_caption_langs == ()


@pytest.mark.parametrize(
    ("descriptions", "asif"),
    [
        # The owner edited As If's caption (in one language or the other).
        ({"x-default": "My own words", "en": "Old"}, {"CaptionLang": "en"}),
        ({"x-default": "Old", "en": "My own words"}, {"CaptionLang": "en"}),
        # The owner added a language, even with As If's own text in it.
        ({"x-default": "Old", "en": "Old", "fr": "Vieux"}, {"CaptionLang": "en"}),
        ({"x-default": "Old", "en": "Old", "fr": "Old"}, {"CaptionLang": "en"}),
        # No (or an unusable) record of which language As If wrote.
        ({"x-default": "Old", "en": "Old"}, {}),
        ({"x-default": "Old", "all": "Old"}, {"CaptionLang": "all"}),
        ({"x-default": "Old", "i-klingon": "Old"}, {"CaptionLang": "i-klingon"}),
        ({"x-default": "Old", "": "Old"}, {"CaptionLang": ""}),
    ],
)
def test_edited_own_caption_is_kept(descriptions: dict[str, str], asif: dict[str, str]) -> None:
    existing = ExistingMeta(descriptions=descriptions, asif={"CaptionDigest": sha("Old"), **asif})
    p = plan(existing, lang="de")
    assert p.caption is None
    assert p.caption_reason == "kept-edited"
    assert p.remove_caption_langs == ()
    assert p.caption_digest is None and p.caption_lang is None


def test_caption_deleted_by_owner_is_written_again() -> None:
    existing = ExistingMeta(asif={"CaptionDigest": sha("Old"), "CaptionLang": "en"})
    p = plan(existing)
    assert p.caption_reason == "empty"
    assert p.caption == "A red shed in a garden"


# -- plan_write: keywords ------------------------------------------------------


def test_keywords_dedupe_against_the_owners_case_insensitively() -> None:
    existing = ExistingMeta(subjects=("Garden", "family"))
    p = plan(existing, result(keywords=("garden", "FAMILY", "dog", "shed")))
    assert p.add_subjects == ("dog", "shed")
    assert p.all_added_subjects == ("dog", "shed")


def test_keyword_dedupe_sees_through_unicode_normalisation() -> None:
    existing = ExistingMeta(subjects=("Cafe\u0301", "ice  cream"))
    p = plan(existing, result(keywords=("café", "ice cream", "table")))
    assert p.add_subjects == ("table",)


def test_model_duplicates_and_blank_keywords_are_dropped() -> None:
    p = plan(res=raw_result("ok", ["dog", "Dog", " ", "\n", "\u202e", "DOG ", "cat"]))
    assert p.add_subjects == ("dog", "cat")


def test_added_keywords_accumulate_across_runs() -> None:
    existing = ExistingMeta(
        subjects=("Garden", "shed"), asif={"AddedKeywords": ("shed",), "AddedIPTCKeywords": ()}
    )
    p = plan(existing, result(keywords=("shed", "garden", "sky")))
    assert p.add_subjects == ("sky",)
    assert p.all_added_subjects == ("shed", "sky")


def test_added_keywords_record_every_spelling_as_if_added() -> None:
    # As If added "Dog"; the owner removed it; a later run adds "dog". Undo
    # must know about both spellings.
    existing = ExistingMeta(subjects=("cat",), asif={"AddedKeywords": ("Dog",)})
    p = plan(existing, result(keywords=("dog",)))
    assert p.add_subjects == ("dog",)
    assert p.all_added_subjects == ("Dog", "dog")


def test_previous_iptc_record_survives_an_xmp_only_run() -> None:
    existing = ExistingMeta(asif={"AddedIPTCKeywords": ("garden", "shed")})
    p = plan(existing, compat="xmp")
    assert p.add_iptc == ()
    assert p.all_added_iptc == ("garden", "shed")


# -- plan_write: IPTC ----------------------------------------------------------


def test_compat_xmp_never_touches_iptc() -> None:
    for existing in (ExistingMeta(), ExistingMeta(iptc_present=True, iptc_utf8=True)):
        p = plan(existing, compat="xmp")
        assert p.add_iptc == () and not p.create_iptc and p.iptc_note is None


def test_plan_compat_wide_creates_utf8_iptc() -> None:
    p = plan(compat="wide")
    assert p.add_iptc == ("garden", "shed", "sky")
    assert p.create_iptc
    assert p.iptc_note is None
    assert p.all_added_iptc == ("garden", "shed", "sky")


def test_compat_wide_appends_to_existing_utf8_iptc() -> None:
    existing = ExistingMeta(iptc_present=True, iptc_utf8=True, iptc_keywords=("Garden", "old"))
    p = plan(existing, compat="wide")
    assert p.add_iptc == ("shed", "sky")
    assert not p.create_iptc  # never relabel (or re-create) IPTC that exists


def test_iptc_dedupe_is_independent_of_xmp() -> None:
    existing = ExistingMeta(subjects=("garden", "shed", "sky"))
    p = plan(existing, compat="wide")
    assert p.add_subjects == ()
    assert p.add_iptc == ("garden", "shed", "sky")


def test_plan_compat_wide_leaves_latin1_iptc_alone() -> None:
    existing = ExistingMeta(iptc_present=True, iptc_utf8=False, iptc_keywords=("café", "old"))
    p = plan(existing, compat="wide")
    assert p.add_iptc == () and not p.create_iptc
    assert p.iptc_note is not None and "UTF-8" in p.iptc_note
    assert p.add_subjects == ("garden", "shed", "sky")  # XMP still gets them


def test_compat_wide_skips_formats_without_standard_iptc() -> None:
    p = plan(ExistingMeta(iptc_supported=False), compat="wide")
    assert p.add_iptc == () and not p.create_iptc
    assert p.iptc_note is not None and "format" in p.iptc_note


def test_iptc_byte_limit_trims_without_naming_the_keyword() -> None:
    long_cjk = "庭" * 22  # 22 characters, 66 UTF-8 bytes
    exact = "é" * 32  # 64 bytes: fits
    assert len(long_cjk.encode()) > IPTC_KEYWORD_MAX_BYTES == len(exact.encode())
    p = plan(res=result(keywords=("garden", long_cjk, exact)), compat="wide")
    assert p.add_subjects == ("garden", long_cjk, exact)
    assert p.add_iptc == ("garden", exact)
    assert p.iptc_note is not None
    assert "1 keyword" in p.iptc_note and long_cjk not in p.iptc_note and "庭" not in p.iptc_note


def test_no_iptc_is_created_when_nothing_fits() -> None:
    p = plan(res=result(keywords=("庭" * 30, "園" * 30)), compat="wide")
    assert p.add_iptc == () and not p.create_iptc
    assert p.iptc_note is not None and p.iptc_note.startswith("2 keywords")


# -- plan_write: the image hash ------------------------------------------------


def test_plan_existing_original_hash_is_never_overwritten() -> None:
    p = plan(ExistingMeta(original_image_hash=OTHER_HASH))
    assert not p.write_original_hash
    assert not p.wrote_original_hash


def test_even_an_empty_original_hash_is_left_alone() -> None:
    assert not plan(ExistingMeta(original_image_hash="")).write_original_hash


def test_own_original_hash_is_remembered() -> None:
    existing = ExistingMeta(original_image_hash=HASH, asif={"WroteOriginalImageHash": "True"})
    p = plan(existing)
    assert not p.write_original_hash
    assert p.wrote_original_hash


# -- plan_write: inputs ----------------------------------------------------------


@pytest.mark.parametrize(
    "lang",
    ["", "english", "en_US", "x-default", "all", "en-all", "en\n-o", "en\n", "en=x", "e"],
)
def test_plan_refuses_bad_languages(lang: str) -> None:
    with pytest.raises(ValueError, match="language"):
        plan(lang=lang)


@pytest.mark.parametrize("lang", ["en", "pt-BR", "pt-br", "zh-Hant-TW", "es-419", "de-CH-1996"])
def test_plan_accepts_real_languages(lang: str) -> None:
    assert plan(lang=lang).caption_lang == lang


@pytest.mark.parametrize(
    "image_hash", ["", "AB" * 32, "zz" * 32, HASH[:-1], HASH + "0", HASH + "\n"]
)
def test_plan_refuses_bad_image_hashes(image_hash: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        plan_write(ExistingMeta(), result(), lang="en", compat="xmp", image_hash=image_hash)


def test_plan_refuses_unknown_compat() -> None:
    with pytest.raises(ValueError, match="compat"):
        plan(compat="iptc")


def test_plan_refuses_a_caption_that_cleans_to_nothing() -> None:
    with pytest.raises(ValueError, match="empty"):
        plan(res=raw_result("\n\u202e \t\ufffe", ["dog"]))


def test_plan_is_pure() -> None:
    existing = replace(ours("Old"), subjects=("x",), iptc_present=True, iptc_utf8=True)
    snapshot = repr(existing)
    first = plan(existing, compat="wide")
    assert plan(existing, compat="wide") == first
    assert repr(existing) == snapshot


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_model_output_is_cleaned_in_the_plan(text: str) -> None:
    p = plan(res=raw_result(f"cap {text}", [text, f"k {text}"]))
    for value in (p.caption or "", *p.add_subjects, *p.all_added_subjects):
        assert not any(ch in value for ch in "\n\r\0\u2028\u2029\u202e\u2066\ufffe\uffff")
        assert value == value.strip()
    assert p.caption is not None and len(p.caption) <= CAPTION_MAX
    assert p.caption_digest == sha(p.caption)


# -- build_args ----------------------------------------------------------------


def test_build_args_full() -> None:
    p = replace(
        BASE_PLAN,
        caption="A red shed",
        caption_reason="ours",
        remove_caption_langs=("de",),
        add_subjects=("shed", "sky"),
        add_iptc=("shed",),
        create_iptc=True,
        write_original_hash=True,
        all_added_subjects=("garden", "shed", "sky"),
        all_added_iptc=("shed",),
        wrote_original_hash=True,
        caption_digest=sha("A red shed"),
        caption_lang="en",
    )
    assert build_args(p, PROV, lang="en") == [
        "-overwrite_original",
        "-P",
        "-XMP-dc:Description-x-default=A red shed",
        "-XMP-dc:Description-en=A red shed",
        "-XMP-dc:Description-de=",
        "-XMP-dc:Subject+=shed",
        "-XMP-dc:Subject+=sky",
        "-charset",
        "iptc=UTF8",
        "-IPTC:CodedCharacterSet=UTF8",
        "-IPTC:Keywords+=shed",
        f"-XMP-et:OriginalImageHash={HASH}",
        "-XMP-et:OriginalImageHashType=SHA256",
        "-XMP-asif:ToolVersion=0.1",
        "-XMP-asif:ModelId=stub",
        "-XMP-asif:ModelSHA256=",
        "-XMP-asif:PromptVersion=caption_v1",
        "-XMP-asif:RunId=r_test",
        "-XMP-asif:Timestamp=2026:10:25 12:00:00Z",
        f"-XMP-asif:CaptionDigest={sha('A red shed')}",
        "-XMP-asif:CaptionLang=en",
        "-XMP-asif:AddedKeywords=garden",
        "-XMP-asif:AddedKeywords=shed",
        "-XMP-asif:AddedKeywords=sky",
        "-XMP-asif:AddedIPTCKeywords=shed",
        "-XMP-asif:WroteOriginalImageHash=True",
    ]


def test_build_args_when_only_provenance_changes() -> None:
    assert build_args(BASE_PLAN, PROV2, lang="en") == [
        "-overwrite_original",
        "-P",
        "-XMP-asif:ToolVersion=0.2",
        "-XMP-asif:ModelId=qwen3.5-9b-q4km",
        f"-XMP-asif:ModelSHA256={'ef' * 32}",
        "-XMP-asif:PromptVersion=caption_v1",
        "-XMP-asif:RunId=r_second",
        "-XMP-asif:Timestamp=2026:10:26 08:30:00Z",
        "-XMP-asif:AddedKeywords=",  # one empty assignment clears the list
        "-XMP-asif:AddedIPTCKeywords=",
        "-XMP-asif:WroteOriginalImageHash=False",
    ]


def test_build_args_appending_to_existing_iptc_sets_no_charset() -> None:
    p = replace(BASE_PLAN, add_iptc=("dog",), create_iptc=False, all_added_iptc=("dog",))
    args = build_args(p, PROV, lang="en")
    assert "-IPTC:Keywords+=dog" in args
    assert "-charset" not in args
    assert not any(a.startswith("-IPTC:CodedCharacterSet") for a in args)


def test_build_args_skips_charset_when_no_iptc_keyword_survives() -> None:
    p = replace(BASE_PLAN, add_iptc=(" ", "\n"), create_iptc=True)
    args = build_args(p, PROV, lang="en")
    assert not any("IPTC" in a or "iptc" in a for a in args if not a.startswith("-XMP-asif"))


def test_build_args_from_plan_write_round_trip() -> None:
    p = plan(ours("Old", "en"), result("Nova"), lang="pt-BR", compat="wide")
    args = build_args(p, PROV, lang="pt-BR")
    assert args[:5] == [
        "-overwrite_original",
        "-P",
        "-XMP-dc:Description-x-default=Nova",
        "-XMP-dc:Description-pt-BR=Nova",
        "-XMP-dc:Description-en=",
    ]
    assert "-XMP-asif:CaptionLang=pt-BR" in args
    assert f"-XMP-asif:CaptionDigest={sha('Nova')}" in args


@pytest.mark.parametrize(
    ("changes", "lang", "prov", "match"),
    [
        ({}, "en\n-o", PROV, "language"),
        ({}, "all", PROV, "language"),
        ({}, "en", replace(PROV, timestamp="2026-10-25T12:00:00Z"), "timestamp"),
        ({}, "en", replace(PROV, timestamp="2026:10:25 12:00:00"), "timestamp"),
        ({}, "en", replace(PROV, timestamp="2026:10:25 12:00:00Z\n"), "timestamp"),
        ({"caption": "x", "remove_caption_langs": ("EN",)}, "en", PROV, "both"),
        ({"caption": "x", "remove_caption_langs": ("x-default",)}, "en", PROV, "language"),
        ({"caption": "x", "remove_caption_langs": ("en=evil",)}, "en", PROV, "language"),
        ({"remove_caption_langs": ("de",)}, "en", PROV, "only removed"),
        ({"caption": " \n\u202e "}, "en", PROV, "empty caption"),
        ({"add_iptc": ("庭" * 22,)}, "en", PROV, "64 bytes"),
        ({"write_original_hash": True, "image_hash": "nope"}, "en", PROV, "SHA-256"),
        ({"caption": "x", "caption_digest": "abc"}, "en", PROV, "SHA-256"),
        ({"caption": "x", "caption_lang": "en\r"}, "en", PROV, "language"),
    ],
)
def test_build_args_refuses_inconsistent_plans(
    changes: dict[str, Any], lang: str, prov: Provenance, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_args(replace(BASE_PLAN, **changes), prov, lang=lang)


_FIXED_TAGS = {
    "XMP-dc:Subject+",
    "IPTC:Keywords+",
    "IPTC:CodedCharacterSet",
    "XMP-et:OriginalImageHash",
    "XMP-et:OriginalImageHashType",
    *(
        f"XMP-asif:{name}"
        for name in (
            "ToolVersion",
            "ModelId",
            "ModelSHA256",
            "PromptVersion",
            "RunId",
            "Timestamp",
            "CaptionDigest",
            "CaptionLang",
            "AddedKeywords",
            "AddedIPTCKeywords",
            "WroteOriginalImageHash",
        )
    ),
}


def assert_safe_arg_stream(args: list[str], langs: set[str]) -> list[str]:
    """Parse ``args`` the way exiftool does and check every argument is either
    one of the writer's fixed options or an assignment to one of its own tags.
    Returns the assigned values."""
    allowed = _FIXED_TAGS | {f"XMP-dc:Description-{lang}" for lang in langs | {"x-default"}}
    values = []
    i = 0
    while i < len(args):
        arg = args[i]
        assert not any(ch in arg for ch in "\n\r\0"), "a line break would split the argument"
        if arg in ("-overwrite_original", "-P"):
            i += 1
            continue
        if arg == "-charset":
            assert args[i + 1] == "iptc=UTF8"
            i += 2
            continue
        assert arg.startswith("-")
        tag, eq, value = arg[1:].partition("=")  # exiftool splits on the first "="
        assert eq and tag in allowed, tag
        values.append(value)
        i += 1
    return values


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_values_can_never_become_options(text: str) -> None:
    # A plan built by hand with raw hostile values: build_args must still
    # produce a clean, well-formed argument stream.
    p = replace(
        BASE_PLAN,
        caption=f"cap {text}",
        caption_reason="empty",
        add_subjects=(text, f"k {text}"),
        add_iptc=(f"k {text}"[:20],),
        create_iptc=True,
        all_added_subjects=(text,),
        all_added_iptc=(text,),
        caption_digest=sha("whatever"),
        caption_lang="en",
    )
    prov = replace(PROV, model_id=text, run_id=f"r {text}", tool_version=text)
    values = assert_safe_arg_stream(build_args(p, prov, lang="en"), {"en"})
    for value in values:
        assert value == value.strip()
        assert not any(ch in value for ch in "\u2028\u2029\u202e\u2066\ufffe\uffff")


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_model_output_through_plan_write(text: str) -> None:
    res = raw_result(f"cap {text}", [text, f"k {text}", "dog"])
    p = plan(ours("Old", "fr"), res, lang="en", compat="wide")
    assert_safe_arg_stream(build_args(p, PROV, lang="en"), {"en", "fr"})


# -- real exiftool -------------------------------------------------------------


@pytest.fixture(scope="module")
def et() -> Iterator[ExifTool]:
    if not HAVE_EXIFTOOL:
        pytest.skip("exiftool not installed")
    with ExifTool() as tool:
        yield tool


Copy = Callable[[str], Path]


def run_writer(
    et: ExifTool,
    path: Path,
    res: CaptionResult | None = None,
    *,
    lang: str = "en",
    compat: str = "xmp",
    prov: Provenance = PROV,
) -> tuple[WritePlan, WriteRecord]:
    before = et.image_data_hash(path)
    p = plan_write(
        read_existing(et, path),
        res or result(),
        lang=lang,
        compat=compat,  # type: ignore[arg-type]
        image_hash=before,
    )
    record = apply_write(et, path, p, prov, lang=lang)
    # Sanity only: the gates module does the real checking.
    assert et.image_data_hash(path) == before
    return p, record


def exiftool_set(et: ExifTool, path: Path, *assignments: str) -> None:
    """Change a file the way its owner (or another app) might."""
    et.execute("-overwrite_original", *assignments, str(path))


def test_read_existing_on_a_bare_file(et: ExifTool, fixture_copy: Copy) -> None:
    assert read_existing(et, fixture_copy("plain.jpg")) == ExistingMeta(iptc_supported=True)


def test_read_existing_on_the_owners_xmp(et: ExifTool, fixture_copy: Copy) -> None:
    meta = read_existing(et, fixture_copy("user_xmp.jpg"))
    assert meta.descriptions == {"x-default": "Grandma's garden, summer 2003"}
    assert meta.subjects == ("Garden", "family")
    assert not meta.iptc_present
    assert meta.asif == {} and meta.original_image_hash is None


def test_read_existing_on_latin1_iptc(et: ExifTool, fixture_copy: Copy) -> None:
    meta = read_existing(et, fixture_copy("user_iptc.jpg"))
    assert meta.iptc_present and not meta.iptc_utf8
    assert meta.iptc_keywords == ("café", "old")


@pytest.mark.parametrize(
    ("name", "supported"),
    [("plain.png", False), ("lossy.webp", False), ("plain.tif", True), ("mpo.jpg", True)],
)
def test_read_existing_knows_which_formats_take_iptc(
    et: ExifTool, fixture_copy: Copy, name: str, supported: bool
) -> None:
    assert read_existing(et, fixture_copy(name)).iptc_supported is supported


def test_read_existing_keeps_numbers_and_booleans_as_text(et: ExifTool, fixture_copy: Copy) -> None:
    # exiftool's JSON would hand these back as 2019, 1.5, true, ... without
    # the filter As If uses.
    path = fixture_copy("plain.jpg")
    words = ["2019", "1.50", "TRUE", "false", "007", "-0", "1e5", "null"]
    exiftool_set(
        et,
        path,
        "-XMP-dc:Description-x-default=1999",
        "-XMP-dc:Description-en=0.10",
        *(f"-XMP-dc:Subject={w}" for w in words),
    )
    meta = read_existing(et, path)
    assert meta.descriptions == {"x-default": "1999", "en": "0.10"}
    assert meta.subjects == tuple(words)
    # ...and so the plan sees them for what they are.
    p = plan_write(
        meta, result(keywords=("2019", "true", "1.5")), lang="en", compat="xmp", image_hash=HASH
    )
    assert p.add_subjects == ("1.5",)
    assert p.caption_reason == "kept-user"


def test_read_existing_single_item_lists_are_tuples(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    exiftool_set(et, path, "-XMP-dc:Subject=2019")
    assert read_existing(et, path).subjects == ("2019",)


def test_read_existing_sees_every_language(
    et: ExifTool, fixture_copy: Copy, tmp_path: Path
) -> None:
    xmp = tmp_path / "owner.xmp"
    xmp.write_text(
        "<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<rdf:Description rdf:about='' xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        "<dc:description><rdf:Alt>"
        "<rdf:li xml:lang='x-default'>Default</rdf:li>"
        "<rdf:li xml:lang='EN-us'>English</rdf:li>"
        "<rdf:li xml:lang='de-CH-1996'>Swiss</rdf:li>"
        "<rdf:li xml:lang=''>No language</rdf:li>"
        "</rdf:Alt></dc:description>"
        "</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end='w'?>",
        encoding="utf-8",
    )
    path = fixture_copy("plain.jpg")
    exiftool_set(et, path, f"-xmp<={xmp}")
    assert read_existing(et, path).descriptions == {
        "x-default": "Default",
        "en-US": "English",
        "de-CH-1996": "Swiss",
        "": "No language",
    }


def test_read_existing_raises_for_unreadable_files(
    et: ExifTool, fixture_copy: Copy, tmp_path: Path
) -> None:
    with pytest.raises(ExifToolError):
        read_existing(et, tmp_path / "missing.jpg")
    with pytest.raises(ExifToolError):
        read_existing(et, fixture_copy("corrupt.jpg"))


def test_write_plain_jpg(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    p, record = run_writer(et, path)
    assert record == WriteRecord(
        caption_written=True,
        caption_reason="empty",
        subjects_added=("garden", "shed", "sky"),
        iptc_added=(),
        original_hash_written=True,
        iptc_note=None,
    )
    meta = read_existing(et, path)
    assert meta.descriptions == {"x-default": p.caption, "en": p.caption}
    assert meta.subjects == ("garden", "shed", "sky")
    assert not meta.iptc_present
    assert meta.original_image_hash == p.image_hash
    assert meta.asif == {
        "ToolVersion": "0.1",
        "ModelId": "stub",
        "PromptVersion": "caption_v1",
        "RunId": "r_test",
        "Timestamp": "2026:10:25 12:00:00Z",
        "CaptionDigest": sha("A red shed in a garden"),
        "CaptionLang": "en",
        "AddedKeywords": ("garden", "shed", "sky"),
        "WroteOriginalImageHash": "True",
    }
    raw = et.read_tags(path, "-XMP-et:OriginalImageHashType", "-XMP-asif:ModelSHA256")
    assert raw == {"XMP-et:OriginalImageHashType": "SHA256"}  # empty SHA: no tag at all


def test_write_keeps_file_times_to_the_nanosecond(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("exif.jpg")
    times = (1_600_000_000_123_456_789, 1_500_000_000_987_654_321)
    p = plan_write(read_existing(et, path), result(), lang="en", compat="xmp", image_hash=HASH)
    os.utime(path, ns=times)
    apply_write(et, path, p, PROV, lang="en")
    st = path.stat()
    assert (st.st_atime_ns, st.st_mtime_ns) == times


def test_owner_caption_and_keywords_are_kept(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("user_xmp.jpg")
    creator_before = et.read_tags(path, "-XMP-dc:Creator")
    _, record = run_writer(et, path, result(keywords=("garden", "Family", "shed")))
    assert not record.caption_written
    assert record.caption_reason == "kept-user"
    assert record.subjects_added == ("shed",)
    meta = read_existing(et, path)
    assert meta.descriptions == {"x-default": "Grandma's garden, summer 2003"}
    assert meta.subjects == ("Garden", "family", "shed")
    assert "CaptionDigest" not in meta.asif and "CaptionLang" not in meta.asif
    assert meta.asif["AddedKeywords"] == ("shed",)
    assert et.read_tags(path, "-XMP-dc:Creator") == creator_before


def test_rerun_on_own_output(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    first, _ = run_writer(et, path)
    _, record = run_writer(
        et,
        path,
        result("A shed and a dog", ("Garden", "shed", "dog", "SKY")),
        prov=PROV2,
    )
    assert record.caption_written and record.caption_reason == "ours"
    assert record.subjects_added == ("dog",)
    assert not record.original_hash_written
    meta = read_existing(et, path)
    assert meta.descriptions == {"x-default": "A shed and a dog", "en": "A shed and a dog"}
    assert meta.subjects == ("garden", "shed", "sky", "dog")
    assert meta.original_image_hash == first.image_hash
    assert meta.asif["AddedKeywords"] == ("garden", "shed", "sky", "dog")
    assert meta.asif["CaptionDigest"] == sha("A shed and a dog")
    assert meta.asif["WroteOriginalImageHash"] == "True"
    assert meta.asif["RunId"] == "r_second"
    assert meta.asif["ModelSHA256"] == "ef" * 32
    # A third identical run changes nothing that matters and duplicates nothing.
    _, again = run_writer(et, path, result("A shed and a dog", ("dog", "shed")), prov=PROV2)
    assert again.caption_reason == "ours" and again.subjects_added == ()
    assert read_existing(et, path).subjects == ("garden", "shed", "sky", "dog")


def test_caption_edited_by_the_owner_is_kept(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    first, _ = run_writer(et, path)
    exiftool_set(et, path, "-XMP-dc:Description-x-default=My own words")
    _, record = run_writer(et, path, result("Something else entirely"))
    assert not record.caption_written
    assert record.caption_reason == "kept-edited"
    meta = read_existing(et, path)
    assert meta.descriptions == {"x-default": "My own words", "en": first.caption}
    assert meta.asif["CaptionDigest"] == first.caption_digest  # the record stays as it was


def test_language_change_replaces_own_caption(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    run_writer(et, path, lang="en")
    p, record = run_writer(et, path, result("Um galpão vermelho"), lang="pt-BR")
    assert record.caption_reason == "ours"
    assert p.remove_caption_langs == ("en",)
    meta = read_existing(et, path)
    assert meta.descriptions == {
        "x-default": "Um galpão vermelho",
        "pt-BR": "Um galpão vermelho",
    }
    assert meta.asif["CaptionLang"] == "pt-BR"


def test_language_change_keeps_a_language_the_owner_added(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    first, _ = run_writer(et, path, lang="en")
    exiftool_set(et, path, "-XMP-dc:Description-fr=Une remise rouge")
    _, record = run_writer(et, path, result("Um galpão vermelho"), lang="pt-BR")
    assert record.caption_reason == "kept-edited"
    assert read_existing(et, path).descriptions == {
        "x-default": first.caption,
        "en": first.caption,
        "fr": "Une remise rouge",
    }


def test_language_tag_case_normalised_by_exiftool(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    run_writer(et, path, lang="pt-br")
    assert set(read_existing(et, path).descriptions) == {"x-default", "pt-BR"}
    p, record = run_writer(et, path, result("Outra"), lang="pt-BR")
    assert record.caption_reason == "ours" and p.remove_caption_langs == ()
    assert read_existing(et, path).descriptions == {"x-default": "Outra", "pt-BR": "Outra"}


def test_long_caption_round_trips_and_stays_ours(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    first, _ = run_writer(et, path, result("word " * 60))  # 300 characters, cut to 240
    assert first.caption is not None and len(first.caption) <= CAPTION_MAX
    assert read_existing(et, path).descriptions["en"] == first.caption
    _, record = run_writer(et, path, result("Short now"))
    assert record.caption_reason == "ours"


def test_unicode_caption_round_trips(et: ExifTool, fixture_copy: Copy) -> None:
    text = "Família 👨\u200d👩\u200d👧 no jardim — 庭園 <&> \"quoted\" 'single' عربى"
    path = fixture_copy("plain.jpg")
    first, _ = run_writer(et, path, result(text))
    assert read_existing(et, path).descriptions["x-default"] == first.caption == text
    _, record = run_writer(et, path)
    assert record.caption_reason == "ours"


def test_compat_wide_creates_utf8_iptc(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    _, record = run_writer(et, path, result(keywords=("garden", "café", "shed")), compat="wide")
    assert record.iptc_added == ("garden", "café", "shed")
    assert record.iptc_note is None
    meta = read_existing(et, path)
    assert meta.iptc_present and meta.iptc_utf8
    assert meta.iptc_keywords == ("garden", "café", "shed")
    assert meta.asif["AddedIPTCKeywords"] == ("garden", "café", "shed")
    assert "café".encode() in path.read_bytes()
    assert et.read_tags(path, "-IPTC:CodedCharacterSet") == {"IPTC:CodedCharacterSet": "UTF8"}


def test_compat_wide_leaves_latin1_iptc_alone(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("user_iptc.jpg")
    iptc_before = et.read_tags(path, "-IPTC:all")
    _, record = run_writer(et, path, result(keywords=("garden", "old")), compat="wide")
    assert record.iptc_added == ()
    assert record.iptc_note is not None and "UTF-8" in record.iptc_note
    assert record.subjects_added == ("garden", "old")  # XMP still gets them
    assert et.read_tags(path, "-IPTC:all") == iptc_before
    assert b"caf\xe9" in path.read_bytes()  # still Latin-1, byte for byte


def test_compat_wide_appends_to_utf8_iptc(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    run_writer(et, path, result(keywords=("garden", "café")), compat="wide")
    p, record = run_writer(et, path, result(keywords=("Garden", "dog", "CAFÉ")), compat="wide")
    assert not p.create_iptc
    assert record.iptc_added == ("dog",)
    meta = read_existing(et, path)
    assert meta.iptc_keywords == ("garden", "café", "dog")
    assert meta.asif["AddedIPTCKeywords"] == ("garden", "café", "dog")


def test_compat_wide_respects_owner_utf8_iptc(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    exiftool_set(
        et, path, "-charset", "iptc=UTF8", "-IPTC:CodedCharacterSet=UTF8", "-IPTC:Keywords=Jardim"
    )
    p, record = run_writer(et, path, result(keywords=("jardim", "flor")), compat="wide")
    assert not p.create_iptc
    assert record.iptc_added == ("flor",)
    assert read_existing(et, path).iptc_keywords == ("Jardim", "flor")


def test_existing_original_hash_is_never_overwritten(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    exiftool_set(
        et, path, f"-XMP-et:OriginalImageHash={OTHER_HASH}", "-XMP-et:OriginalImageHashType=MD5"
    )
    _, record = run_writer(et, path)
    assert not record.original_hash_written
    meta = read_existing(et, path)
    assert meta.original_image_hash == OTHER_HASH
    assert meta.asif["WroteOriginalImageHash"] == "False"
    assert et.read_tags(path, "-XMP-et:OriginalImageHashType") == {
        "XMP-et:OriginalImageHashType": "MD5"
    }


@pytest.mark.parametrize(
    ("name", "iptc"),
    [
        ("plain.png", False),
        ("alpha.png", False),
        ("lossy.webp", False),
        ("lossless.webp", False),
        ("plain.tif", True),
        ("exif.jpg", True),
        ("gps.jpg", True),
        ("rotated.jpg", True),
        ("progressive.jpg", True),
        ("motion.jpg", True),
        ("mpo.jpg", True),
    ],
)
def test_every_writable_format(et: ExifTool, fixture_copy: Copy, name: str, iptc: bool) -> None:
    path = fixture_copy(name)
    p, record = run_writer(et, path, compat="wide")
    assert record.caption_written and record.subjects_added == ("garden", "shed", "sky")
    meta = read_existing(et, path)
    assert meta.descriptions == {"x-default": p.caption, "en": p.caption}
    assert meta.original_image_hash == p.image_hash
    if iptc:
        assert record.iptc_added == ("garden", "shed", "sky")
        assert meta.iptc_keywords == ("garden", "shed", "sky")
    else:
        assert record.iptc_added == ()
        assert record.iptc_note is not None and "format" in record.iptc_note
        assert not meta.iptc_present
    # A second pass is still clean on every format.
    _, again = run_writer(et, path, compat="wide")
    assert again.caption_reason == "ours" and again.subjects_added == ()


@pytest.mark.parametrize("text", HOSTILE)
@pytest.mark.parametrize("validated", [True, False], ids=["validated", "raw"])
def test_hostile_model_output_cannot_escape(
    et: ExifTool, fixture_copy: Copy, tmp_path: Path, text: str, validated: bool
) -> None:
    evil = tmp_path / "evil.jpg"
    caption = f"cap {text}\n-o\n{evil}\n-execute\n"
    keywords = [text, f"k {text}", f"-o {evil}", "dog"]
    res = result(caption, keywords) if validated else raw_result(caption, keywords)
    path = fixture_copy("plain.jpg")
    p, record = run_writer(et, path, res, compat="wide")
    assert not evil.exists()
    assert not Path("/etc/x").exists()
    meta = read_existing(et, path)
    assert meta.descriptions == {"x-default": p.caption, "en": p.caption}
    assert meta.subjects == record.subjects_added
    assert meta.iptc_keywords == record.iptc_added
    # exiftool is still in step with us: the next command answers normally.
    assert et.version()


@pytest.mark.slow
def test_panorama(et: ExifTool, panorama: Path, tmp_path: Path) -> None:
    path = tmp_path / "panorama.jpg"
    path.write_bytes(panorama.read_bytes())
    _, record = run_writer(et, path, compat="wide")
    assert record.caption_written and record.iptc_added


# -- apply_write's own checks (a scripted exiftool) ----------------------------


class ScriptedExifTool:
    """Answers ``execute`` and ``execute_json`` from a script."""

    def __init__(self, *, stdout: str, stderr: str = "", rows: list[dict[str, Any]]) -> None:
        self.stdout, self.stderr, self.rows = stdout, stderr, rows
        self.calls: list[tuple[str, ...]] = []

    def execute(self, *args: str) -> ExecResult:
        self.calls.append(args)
        warnings = [line for line in self.stderr.splitlines() if line.startswith("Warning")]
        return ExecResult(self.stdout, self.stderr, warnings)

    def execute_json(self, *args: str) -> list[dict[str, Any]]:
        return self.rows


def marked(row: dict[str, Any]) -> list[dict[str, Any]]:
    """A JSON row as exiftool returns it with the writer's read filter on."""

    def mark(value: Any) -> Any:
        return [f"s:{v}" for v in value] if isinstance(value, list) else f"s:{value}"

    return [{"SourceFile": "x.jpg", **{k: mark(v) for k, v in row.items()}}]


SECRET_PLAN = replace(
    BASE_PLAN,
    caption="secret caption",
    caption_reason="empty",
    add_subjects=("secretkeyword",),
    all_added_subjects=("secretkeyword",),
    caption_digest=sha("secret caption"),
    caption_lang="en",
)
GOOD_ROW: dict[str, Any] = {
    "XMP-dc:Description": "secret caption",
    "XMP-dc:Description-en": "secret caption",
    "XMP-dc:Subject": ["secretkeyword"],
    "XMP-asif:CaptionDigest": sha("secret caption"),
    "XMP-asif:AddedKeywords": ["secretkeyword"],
    "File:FileType": "JPEG",
}
GOOD_ROWS = marked(GOOD_ROW)


@pytest.fixture
def ram_file(tmp_path: Path) -> Path:
    path = tmp_path / "x.jpg"
    path.write_bytes(b"not really a photo; the scripted exiftool never reads it")
    return path


def scripted_write(tool: ScriptedExifTool, path: Path) -> WriteRecord:
    return apply_write(cast(ExifTool, tool), path, SECRET_PLAN, PROV, lang="en")


def test_apply_write_with_a_good_read_back(ram_file: Path) -> None:
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=GOOD_ROWS)
    record = scripted_write(tool, ram_file)
    assert record.caption_written and record.subjects_added == ("secretkeyword",)
    assert tool.calls[0][-1] == str(ram_file)
    assert tool.calls[0][:-1] == tuple(build_args(SECRET_PLAN, PROV, lang="en"))


def test_apply_write_passes_an_absolute_path(
    ram_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ram_file.parent)
    option_like = Path("-o")
    option_like.write_bytes(b"x")
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=GOOD_ROWS)
    scripted_write(tool, option_like)
    assert tool.calls[0][-1] == str(ram_file.parent / "-o")


def test_apply_write_refuses_a_missing_file(tmp_path: Path) -> None:
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=GOOD_ROWS)
    with pytest.raises(ExifToolError):
        scripted_write(tool, tmp_path / "gone.jpg")
    assert tool.calls == []


def test_apply_write_refuses_any_warning(ram_file: Path) -> None:
    tool = ScriptedExifTool(
        stdout="    1 image files updated\n",
        stderr="Warning: something about secret caption\n",
        rows=GOOD_ROWS,
    )
    with pytest.raises(ExifToolError, match="warn") as info:
        scripted_write(tool, ram_file)
    assert "secret" not in str(info.value)


@pytest.mark.parametrize(
    "stdout", ["    0 image files updated\n    1 image files unchanged\n", "", "Nothing to do.\n"]
)
def test_apply_write_needs_exiftool_to_report_the_update(stdout: str, ram_file: Path) -> None:
    with pytest.raises(ExifToolError, match="updated"):
        scripted_write(ScriptedExifTool(stdout=stdout, rows=GOOD_ROWS), ram_file)


@pytest.mark.parametrize(
    ("change", "tag"),
    [
        ({"XMP-dc:Description-en": "secret ???"}, "XMP-dc:Description"),
        ({"XMP-dc:Subject": ["other"]}, "XMP-dc:Subject"),
        ({"XMP-asif:CaptionDigest": OTHER_HASH}, "XMP-asif:CaptionDigest"),
        ({"XMP-asif:AddedKeywords": ["secretkeyword", "extra"]}, "XMP-asif:AddedKeywords"),
    ],
)
def test_apply_write_checks_the_read_back(change: dict[str, Any], tag: str, ram_file: Path) -> None:
    rows = marked({**GOOD_ROW, **change})
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=rows)
    with pytest.raises(ExifToolError, match=tag) as info:
        scripted_write(tool, ram_file)
    assert "secret" not in str(info.value)


def test_read_existing_refuses_values_the_filter_did_not_mark() -> None:
    tool = ScriptedExifTool(
        stdout="", rows=[{"SourceFile": "x.jpg", "XMP-dc:Subject": [2019, "s:dog"]}]
    )
    with pytest.raises(ExifToolError, match="XMP-dc:Subject") as info:
        read_existing(cast(ExifTool, tool), Path("/ram/x.jpg"))
    assert "2019" not in str(info.value)


def test_read_existing_refuses_an_empty_answer() -> None:
    with pytest.raises(ExifToolError):
        read_existing(cast(ExifTool, ScriptedExifTool(stdout="", rows=[])), Path("/ram/x.jpg"))
