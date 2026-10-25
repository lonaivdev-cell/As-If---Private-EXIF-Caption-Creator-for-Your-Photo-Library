"""The additive-only writer: the pure planning rules first, then real
exiftool round trips on private copies of the synthetic fixtures."""

from __future__ import annotations

import contextlib
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


@pytest.mark.parametrize(
    "legacy",
    [
        ("Grandpa at the lake, 1998",),  # IPTC Caption-Abstract (Picasa and friends)
        ("Our wedding day, Lisbon",),  # EXIF ImageDescription
        ("OLYMPUS DIGITAL CAMERA", "Our wedding day"),  # filler next to a real one
        ("Digital camera fun at the beach",),
        ("Sony DSC-RX100 on a tripod",),
        # A brand first and "camera" last is still a caption, not firmware filler.
        ("Kodak Brownie, my grandfather's first camera",),
        ("Samsung Galaxy S23 rear camera",),
        ("Pentax K1000 film camera",),
        ("Minolta X-700, Dad's camera",),
        ("Konica C35 rangefinder camera",),
        ("Agfa Isolette folding camera",),
        ("Kodak moment with the new camera",),
        ("Kodak and my old digital camera",),
    ],
)
def test_owner_caption_in_older_fields_is_kept(legacy: tuple[str, ...]) -> None:
    # An XMP caption next to it would hide it in apps that read XMP first.
    p = plan(ExistingMeta(legacy_captions=legacy))
    assert p.caption is None
    assert p.caption_reason == "kept-user"
    assert p.caption_digest is None and p.caption_lang is None
    assert p.add_subjects == ("garden", "shed", "sky")  # keywords still go in


@pytest.mark.parametrize(
    "filler",
    [
        "",
        "                               ",  # Nikon and others pad with spaces
        "\0\0\0",
        "OLYMPUS DIGITAL CAMERA",
        "olympus digital camera\0\0",
        "SONY DSC",
        "DSC",
        "DIGITAL CAMERA",
        "LEICA DIGITAL CAMERA",
        "KODAK Digital Still Camera",
        "KODAK CX7530 ZOOM DIGITAL CAMERA",
        "KODAK EASYSHARE C813 ZOOM DIGITAL CAMERA",
        "KODAK Z710 ZOOM DIGITAL CAMERA",
        "KONICA MINOLTA DIGITAL CAMERA",
        "SAMSUNG CAMERA PICTURES",
        "MINOLTA DIGITAL CAMERA",
        "Exif_JPEG_PICTURE",
        "EXIF_JPEG_422",
    ],
)
def test_camera_filler_is_not_a_caption(filler: str) -> None:
    p = plan(ExistingMeta(legacy_captions=(filler, " ")))
    assert p.caption_reason == "empty"
    assert p.caption is not None


def test_own_caption_copied_to_older_fields_is_still_ours() -> None:
    # Apps that keep the fields in step copy As If's caption into them.
    existing = replace(ours("Old"), legacy_captions=("Old", "SONY DSC"))
    p = plan(existing, result("New"))
    assert p.caption_reason == "ours"
    assert p.caption == "New"


def test_older_field_edited_after_as_if_keeps_the_caption() -> None:
    existing = replace(ours("Old"), legacy_captions=("Grandpa at the lake",))
    p = plan(existing, result("New"))
    assert p.caption_reason == "kept-edited"
    assert p.caption is None


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


def test_keywords_dedupe_against_the_owners_iptc_keywords() -> None:
    existing = ExistingMeta(
        subjects=("shed",), iptc_present=True, iptc_utf8=True, iptc_keywords=("GARDEN",)
    )
    p = plan(existing, result(keywords=("garden", "shed", "sky")))
    assert p.add_subjects == ("sky",)
    assert p.copy_subjects == ()  # XMP already has the owner's own list
    assert p.all_added_subjects == ("sky",)


@pytest.mark.parametrize(
    ("utf8", "owner"),
    [
        (True, ("Jardim", "família", "jardim")),
        (False, ("Garden", "family")),  # unmarked, but plain ASCII reads the same either way
    ],
)
def test_iptc_only_keywords_are_copied_into_xmp_first(utf8: bool, owner: tuple[str, ...]) -> None:
    # XMP keywords alone would hide the owner's IPTC ones in apps that read
    # XMP first, so the new XMP list starts with theirs.
    existing = ExistingMeta(iptc_present=True, iptc_utf8=utf8, iptc_keywords=owner)
    p = plan(existing, result(keywords=(owner[-1].upper(), "dog")))
    assert p.copy_subjects == owner  # every spelling, exactly
    assert p.add_subjects == ("dog",)
    assert p.subject_note is None
    # Recorded as copied, never as As If's own.
    assert p.all_added_subjects == ("dog",)
    assert p.all_copied_subjects == owner


def test_nothing_is_copied_when_there_is_nothing_to_add() -> None:
    existing = ExistingMeta(iptc_present=True, iptc_utf8=True, iptc_keywords=("garden",))
    p = plan(existing, result(keywords=("Garden",)))
    assert p.copy_subjects == () and p.add_subjects == () and p.subject_note is None


def test_unmarked_non_ascii_iptc_keywords_are_never_copied() -> None:
    # Latin-1 and unmarked UTF-8 can't be told apart: a wrong guess would
    # copy the owner's keywords garbled.
    existing = ExistingMeta(iptc_present=True, iptc_keywords=("cafÃ©", "dog"))
    p = plan(existing, result(keywords=("dog", "cat")))
    assert p.copy_subjects == () and p.add_subjects == ()
    assert p.subject_note is not None and "UTF-8" in p.subject_note
    assert "caf" not in p.subject_note


def test_copied_keywords_are_remembered_across_runs() -> None:
    existing = ExistingMeta(
        subjects=("Garden", "dog"),
        iptc_present=True,
        iptc_utf8=True,
        iptc_keywords=("Garden",),
        asif={"AddedKeywords": ("dog",), "CopiedKeywords": ("Garden",)},
    )
    p = plan(existing, result(keywords=("dog", "cat")))
    assert p.copy_subjects == () and p.add_subjects == ("cat",)
    assert p.all_copied_subjects == ("Garden",)
    assert p.all_added_subjects == ("dog", "cat")


def test_language_tagged_keywords_get_no_more_keywords() -> None:
    # exiftool would label As If's English keywords with the owner's language.
    existing = ExistingMeta(subjects=("Hund", "chien"), subject_langs=True)
    p = plan(existing, result(keywords=("dog", "beach")))
    assert p.add_subjects == () and p.copy_subjects == ()
    assert p.all_added_subjects == ()
    assert p.subject_note is not None and "language" in p.subject_note
    assert p.caption_reason == "empty"  # the caption still goes in


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
    # XMP keywords alone would hide the owner's in apps that read XMP first,
    # and Latin-1 can't be told from unmarked UTF-8 to copy them across.
    assert p.add_subjects == () and p.copy_subjects == ()
    assert p.subject_note is not None and "café" not in p.subject_note


def test_created_iptc_record_is_remembered() -> None:
    first = plan(compat="wide")
    assert first.create_iptc and first.created_iptc
    # The next run finds the record, so it no longer creates it, but undo
    # still needs to know whose it is: the flag sticks, in any compat mode.
    again = ExistingMeta(
        iptc_present=True,
        iptc_utf8=True,
        iptc_keywords=("garden",),
        asif={"CreatedIPTC": "True"},
    )
    for compat in ("wide", "xmp"):
        p = plan(again, compat=compat)
        assert not p.create_iptc and p.created_iptc


def test_owner_iptc_record_is_never_claimed() -> None:
    existing = ExistingMeta(iptc_present=True, iptc_utf8=True, iptc_keywords=("old",))
    assert not plan(existing, compat="wide").created_iptc
    assert not plan(compat="xmp").created_iptc
    assert not plan(res=result(keywords=("庭" * 30,)), compat="wide").created_iptc


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


def test_existing_xmp_toolkit_is_kept() -> None:
    assert plan(ExistingMeta(xmp_toolkit="Adobe XMP Core 7.0")).keep_xmp_toolkit == (
        "Adobe XMP Core 7.0"
    )
    assert plan(ExistingMeta(xmp_toolkit="")).keep_xmp_toolkit is None
    assert plan().keep_xmp_toolkit is None
    args = build_args(plan(ExistingMeta(xmp_toolkit="x\n-o")), PROV, lang="en")
    # The value is copied inside exiftool; it never becomes an argument.
    assert "-XMP-x:XMPToolkit<XMP-x:XMPToolkit" in args
    assert not any("\n" in a or a.endswith("-o") for a in args)


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
        copy_subjects=("Garden",),
        all_copied_subjects=("Garden",),
        created_iptc=True,
        keep_xmp_toolkit="Adobe XMP Core 7.0",
    )
    assert build_args(p, PROV, lang="en") == [
        "-overwrite_original",
        "-P",
        "-XMP-dc:Description-x-default=A red shed",
        "-XMP-dc:Description-en=A red shed",
        "-XMP-dc:Description-de=",
        "-XMP-dc:Subject+=Garden",
        "-XMP-dc:Subject+=shed",
        "-XMP-dc:Subject+=sky",
        "-XMP-x:XMPToolkit<XMP-x:XMPToolkit",
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
        "-XMP-asif:CopiedKeywords=Garden",
        "-XMP-asif:AddedIPTCKeywords=shed",
        "-XMP-asif:WroteOriginalImageHash=True",
        "-XMP-asif:CreatedIPTC=True",
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
        "-XMP-asif:CopiedKeywords=",
        "-XMP-asif:AddedIPTCKeywords=",
        "-XMP-asif:WroteOriginalImageHash=False",
        "-XMP-asif:CreatedIPTC=False",
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
            "CopiedKeywords",
            "AddedIPTCKeywords",
            "WroteOriginalImageHash",
            "CreatedIPTC",
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
        if arg in ("-overwrite_original", "-P", "-XMP-x:XMPToolkit<XMP-x:XMPToolkit"):
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
def test_hostile_iptc_keywords_copied_into_xmp_stay_data(text: str) -> None:
    # A downloaded photo's own keywords are as untrusted as the model's.
    existing = ExistingMeta(iptc_present=True, iptc_utf8=True, iptc_keywords=(text, "x"))
    p = plan(existing, result(keywords=("dog",)))
    assert p.copy_subjects
    values = assert_safe_arg_stream(build_args(p, PROV, lang="en"), {"en"})
    assert all(v == v.strip() for v in values)


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
        "CreatedIPTC": "False",
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
    # XMP keywords alone would hide the owner's ("café", "old") from apps that
    # read XMP first, and they can't be copied across safely.
    assert record.subjects_added == () and record.subjects_copied == ()
    assert record.subject_note is not None
    assert read_existing(et, path).subjects == ()
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


def mwg(et: ExifTool, path: Path) -> dict[str, Any]:
    """Caption and keywords as apps following the Metadata Working Group's
    rules (Lightroom, Bridge, ...) choose them among XMP, IPTC and EXIF."""
    rows = et.execute_json("-use", "MWG", "-MWG:Description", "-MWG:Keywords", str(path))
    row = dict(rows[0])
    row.pop("SourceFile")
    return row


@pytest.mark.parametrize("charset", ["UTF8", "Latin"])
def test_owner_iptc_only_caption_and_keywords_stay_visible(
    et: ExifTool, fixture_copy: Copy, charset: str
) -> None:
    # A Picasa-era photo: caption and keywords in IPTC only, no XMP at all.
    path = fixture_copy("plain.jpg")
    marker = ["-IPTC:CodedCharacterSet=UTF8"] if charset == "UTF8" else []
    exiftool_set(
        et,
        path,
        "-charset",
        f"iptc={charset}",
        *marker,
        "-IPTC:Caption-Abstract=Grandpa at the lake, 1998",
        "-IPTC:Keywords=Grandpa",
        "-IPTC:Keywords=lake",
    )
    before = mwg(et, path)
    assert read_existing(et, path).legacy_captions == ("Grandpa at the lake, 1998",)
    _, record = run_writer(et, path, result(keywords=("lake", "boat")))
    assert not record.caption_written and record.caption_reason == "kept-user"
    meta = read_existing(et, path)
    assert meta.descriptions == {}
    assert meta.subjects == ("Grandpa", "lake", "boat")  # the owner's first, as copied
    assert record.subjects_copied == ("Grandpa", "lake") and record.subjects_added == ("boat",)
    assert meta.asif["CopiedKeywords"] == ("Grandpa", "lake")
    assert meta.asif["AddedKeywords"] == ("boat",)
    after = mwg(et, path)
    assert after["Description"] == before["Description"] == "Grandpa at the lake, 1998"
    assert after["Keywords"] == ["Grandpa", "lake", "boat"]


def test_owner_exif_caption_is_kept(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("exif.jpg")
    exiftool_set(et, path, "-EXIF:ImageDescription=Our wedding day, Lisbon")
    _, record = run_writer(et, path)
    assert record.caption_reason == "kept-user"
    assert read_existing(et, path).descriptions == {}


def test_camera_filler_in_exif_still_gets_a_caption(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("exif.jpg")
    exiftool_set(et, path, "-EXIF:ImageDescription=OLYMPUS DIGITAL CAMERA         ")
    p, record = run_writer(et, path)
    assert record.caption_written and record.caption_reason == "empty"
    assert read_existing(et, path).descriptions == {"x-default": p.caption, "en": p.caption}


def test_unmarked_latin1_iptc_keywords_get_no_xmp_keywords(
    et: ExifTool, fixture_copy: Copy
) -> None:
    path = fixture_copy("user_iptc.jpg")
    _, record = run_writer(et, path, result(keywords=("garden", "café")))
    assert record.subjects_added == () and record.subjects_copied == ()
    assert record.subject_note is not None
    assert mwg(et, path)["Keywords"] == ["café", "old"]  # still the owner's


def owner_xmp(et: ExifTool, path: Path, tmp_path: Path, subject: str) -> None:
    """Replace the photo's XMP with a packet holding this dc:subject list."""
    xmp = tmp_path / "owner.xmp"
    xmp.write_text(
        "<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        "<x:xmpmeta xmlns:x='adobe:ns:meta/' x:xmptk='Adobe XMP Core 7.0-c000'>"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<rdf:Description rdf:about='' xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        f"<dc:subject><rdf:Bag>{subject}</rdf:Bag></dc:subject>"
        "</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end='w'?>",
        encoding="utf-8",
    )
    exiftool_set(et, path, f"-xmp<={xmp}")


def test_language_tagged_keywords_are_left_alone(
    et: ExifTool, fixture_copy: Copy, tmp_path: Path
) -> None:
    path = fixture_copy("plain.jpg")
    owner_xmp(
        et,
        path,
        tmp_path,
        "<rdf:li xml:lang='de'>Hund</rdf:li><rdf:li xml:lang='fr'>chien</rdf:li>",
    )
    assert read_existing(et, path).subject_langs
    p, record = run_writer(et, path, result(keywords=("dog", "beach")))
    # exiftool would have written <rdf:li xml:lang='de'>dog</rdf:li>.
    assert record.subjects_added == () and record.subject_note is not None
    assert record.caption_written
    meta = read_existing(et, path)
    assert meta.subjects == ("Hund", "chien")
    assert meta.asif["CaptionLang"] == "en" and meta.descriptions["en"] == p.caption
    packet = et.execute("-b", "-XMP", str(path)).stdout
    assert "xml:lang='de'>Hund<" in packet and "dog" not in packet


def test_plain_keywords_are_not_language_tagged(
    et: ExifTool, fixture_copy: Copy, tmp_path: Path
) -> None:
    path = fixture_copy("plain.jpg")
    owner_xmp(et, path, tmp_path, "<rdf:li>Hund</rdf:li><rdf:li xml:lang='fr'>chien</rdf:li>")
    assert read_existing(et, path).subject_langs  # any tagged item counts
    path = fixture_copy("user_xmp.jpg")
    assert not read_existing(et, path).subject_langs
    _, record = run_writer(et, path, result(keywords=("dog",)))
    assert record.subjects_added == ("dog",)
    assert not read_existing(et, path).subject_langs


def test_nul_padded_xmp_still_gets_keywords(et: ExifTool, fixture_copy: Copy) -> None:
    # Padding isn't a language tag: As If's keywords go in, next to the owner's.
    path = fixture_copy("plain.jpg")
    exiftool_set(et, path, "-XMP-dc:Subject=owner kw")
    data = path.read_bytes()
    start = data.index(b"http://ns.adobe.com/xap/1.0/\x00") - 4  # the APP1 marker
    length = int.from_bytes(data[start + 2 : start + 4], "big")
    end = start + 2 + length
    padded = data[start + 4 : end] + b"\0" * 200
    segment = b"\xff\xe1" + (len(padded) + 2).to_bytes(2, "big") + padded
    path.write_bytes(data[:start] + segment + data[end:])
    assert et.execute("-b", "-XMP", str(path)).stdout.endswith("\0")
    assert not read_existing(et, path).subject_langs
    _, record = run_writer(et, path, result(keywords=("dog", "beach")))
    assert record.subjects_added == ("dog", "beach") and record.subject_note is None
    assert read_existing(et, path).subjects == ("owner kw", "dog", "beach")


def test_owner_xmp_toolkit_is_kept(et: ExifTool, fixture_copy: Copy, tmp_path: Path) -> None:
    path = fixture_copy("plain.jpg")
    owner_xmp(et, path, tmp_path, "<rdf:li>Garden</rdf:li>")
    toolkit = {"XMP-x:XMPToolkit": "Adobe XMP Core 7.0-c000"}
    assert et.read_tags(path, "-XMP-x:XMPToolkit") == toolkit
    run_writer(et, path)
    assert et.read_tags(path, "-XMP-x:XMPToolkit") == toolkit
    # Without XMP there is nothing to keep: exiftool's stamp goes with the
    # packet it made, and undo removes both.
    bare = fixture_copy("exif.jpg")
    run_writer(et, bare)
    assert et.read_tags(bare, "-XMP-x:XMPToolkit")["XMP-x:XMPToolkit"].startswith("Image::ExifTool")


def test_created_iptc_record_is_labelled(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    _, record = run_writer(et, path, compat="wide")
    assert record.iptc_created
    assert read_existing(et, path).asif["CreatedIPTC"] == "True"
    # A later run finds the record and creates nothing, but still says whose it is.
    p, again = run_writer(et, path, result(keywords=("garden", "dog")), compat="wide")
    assert not p.create_iptc and not again.iptc_created and again.iptc_added == ("dog",)
    assert read_existing(et, path).asif["CreatedIPTC"] == "True"
    run_writer(et, path, compat="xmp")
    assert read_existing(et, path).asif["CreatedIPTC"] == "True"


def test_owner_iptc_record_is_not_labelled(et: ExifTool, fixture_copy: Copy) -> None:
    path = fixture_copy("plain.jpg")
    exiftool_set(
        et, path, "-charset", "iptc=UTF8", "-IPTC:CodedCharacterSet=UTF8", "-IPTC:Keywords=Jardim"
    )
    _, record = run_writer(et, path, result(keywords=("flor",)), compat="wide")
    assert not record.iptc_created
    assert read_existing(et, path).asif["CreatedIPTC"] == "False"


def read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while chunk := os.read(fd, 1 << 16):
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.parametrize("hard_links", [True, False], ids=["link", "no-hard-links"])
def test_write_zeroes_the_copy_it_replaces(
    et: ExifTool, fixture_copy: Copy, monkeypatch: pytest.MonkeyPatch, hard_links: bool
) -> None:
    # exiftool renames a new file over the old one: the photo as it was
    # would otherwise be freed (or left on disk) without being overwritten.
    path = fixture_copy("exif.jpg")
    original = path.read_bytes()
    if not hard_links:  # as on FAT or exFAT

        def no_link(*args: Any, **kwargs: Any) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(os, "link", no_link)
    fd = os.open(path, os.O_RDONLY)  # what a forensic reader could keep hold of
    try:
        run_writer(et, path)
        assert os.fstat(fd).st_ino != path.stat().st_ino  # exiftool did replace it
        old = read_all(fd)
    finally:
        os.close(fd)
    assert len(old) == len(original) and old == bytes(len(old))
    assert path.read_bytes() != original and path.read_bytes()[:2] == b"\xff\xd8"
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no names left behind


def test_write_never_zeroes_a_copy_with_another_name(et: ExifTool, fixture_copy: Copy) -> None:
    # Only a private copy may be zeroed: one still reachable elsewhere is
    # someone's file.
    path = fixture_copy("plain.jpg")
    original = path.read_bytes()
    other = path.with_name("other.jpg")
    os.link(path, other)
    run_writer(et, path)
    assert other.read_bytes() == original
    assert sorted(p.name for p in path.parent.iterdir()) == sorted([path.name, other.name])


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
    """Answers ``execute`` and ``execute_json`` from a script; ``xmp`` is
    the raw packet a ``-b -XMP`` read gets."""

    def __init__(
        self, *, stdout: str, stderr: str = "", rows: list[dict[str, Any]], xmp: str = ""
    ) -> None:
        self.stdout, self.stderr, self.rows, self.xmp = stdout, stderr, rows, xmp
        self.calls: list[tuple[str, ...]] = []

    def execute(self, *args: str) -> ExecResult:
        if args[:2] == ("-b", "-XMP"):
            return ExecResult(self.xmp, "", [])
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
        ({"XMP-asif:CopiedKeywords": ["secretkeyword"]}, "XMP-asif:CopiedKeywords"),
        ({"XMP-asif:CreatedIPTC": "True"}, "XMP-asif:CreatedIPTC"),
    ],
)
def test_apply_write_checks_the_read_back(change: dict[str, Any], tag: str, ram_file: Path) -> None:
    rows = marked({**GOOD_ROW, **change})
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=rows)
    with pytest.raises(ExifToolError, match=tag) as info:
        scripted_write(tool, ram_file)
    assert "secret" not in str(info.value)


def xmp_packet(subject_items: str, extra: str = "") -> str:
    return (
        "<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        "<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<rdf:Description rdf:about='' xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        f"<dc:subject><rdf:Bag>{subject_items}</rdf:Bag></dc:subject>{extra}"
        "</rdf:Description></rdf:RDF></x:xmpmeta>\n   \n<?xpacket end='w'?>"
    )


def test_apply_write_catches_keywords_labelled_with_a_language(ram_file: Path) -> None:
    # exiftool's JSON never shows xml:lang on keywords; the raw packet does.
    tagged = xmp_packet("<rdf:li xml:lang='de'>secretkeyword</rdf:li>")
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=GOOD_ROWS, xmp=tagged)
    with pytest.raises(ExifToolError, match="XMP-dc:Subject") as info:
        scripted_write(tool, ram_file)
    assert "secret" not in str(info.value)


def test_apply_write_checks_the_xmp_toolkit_was_kept(ram_file: Path) -> None:
    rows = marked({**GOOD_ROW, "XMP-x:XMPToolkit": "Image::ExifTool 12.76"})
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=rows)
    p = replace(SECRET_PLAN, keep_xmp_toolkit="Adobe XMP Core 7.0")
    with pytest.raises(ExifToolError, match="XMP-x:XMPToolkit"):
        apply_write(cast(ExifTool, tool), ram_file, p, PROV, lang="en")


DTD_BOMB = (
    "<?xml version='1.0'?><!DOCTYPE x [<!ENTITY a 'aaaaaaaaaa'>"
    "<!ENTITY b '&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;'>]>"
)


@pytest.mark.parametrize(
    ("packet", "tagged"),
    [
        ("", False),
        (xmp_packet("<rdf:li>Hund</rdf:li><rdf:li>chien</rdf:li>"), False),
        (xmp_packet("<rdf:li xml:lang='de'>Hund</rdf:li>"), True),
        (xmp_packet("<rdf:li>Hund</rdf:li><rdf:li xml:lang=''>chien</rdf:li>"), True),
        # A language on a caption is normal and changes nothing for keywords.
        (
            xmp_packet(
                "<rdf:li>Hund</rdf:li>",
                "<dc:description><rdf:Alt><rdf:li xml:lang='x-default'>Hi</rdf:li>"
                "</rdf:Alt></dc:description>",
            ),
            False,
        ),
        # Some writers pad the packet with NULs, and exiftool passes them on.
        pytest.param(xmp_packet("<rdf:li>Hund</rdf:li>") + "\0", False, id="one-nul"),
        pytest.param(xmp_packet("<rdf:li>Hund</rdf:li>") + "\0" * 200, False, id="nul-padded"),
        pytest.param(
            xmp_packet("<rdf:li xml:lang='de'>Hund</rdf:li>") + "\0" * 200,
            True,
            id="nul-padded-tagged",
        ),
        # Anything that can't be read with certainty counts as tagged.
        ("not xml at all", True),
        (xmp_packet("<rdf:li>Hund</rdf:li>") + "<junk/>", True),
        (DTD_BOMB + xmp_packet("<rdf:li>&b;</rdf:li>"), True),
        ("<?xml version='1.0' encoding='nonsense'?>" + xmp_packet("<rdf:li>x</rdf:li>"), True),
    ],
)
def test_read_existing_reads_keyword_languages_from_the_raw_packet(
    packet: str, tagged: bool
) -> None:
    rows = marked({"XMP-dc:Subject": ["Hund"], "File:FileType": "JPEG"})
    tool = ScriptedExifTool(stdout="", rows=rows, xmp=packet)
    assert read_existing(cast(ExifTool, tool), Path("/ram/x.jpg")).subject_langs is tagged


class ReplacingExifTool(ScriptedExifTool):
    """Writes the way exiftool does: a new file renamed over the old one."""

    def execute(self, *args: str) -> ExecResult:
        if "-overwrite_original" in args:
            path = Path(args[-1])
            tmp = path.with_name(path.name + "_exiftool_tmp")
            tmp.write_bytes(b"the new file")
            os.replace(tmp, path)
        return super().execute(*args)


@pytest.mark.parametrize("hard_links", [True, False], ids=["link", "no-hard-links"])
def test_apply_write_zeroes_the_file_it_replaced(
    ram_file: Path, monkeypatch: pytest.MonkeyPatch, hard_links: bool
) -> None:
    if not hard_links:

        def no_link(*args: Any, **kwargs: Any) -> None:
            raise OSError(95, "Operation not supported")

        monkeypatch.setattr(os, "link", no_link)
    size = ram_file.stat().st_size
    fd = os.open(ram_file, os.O_RDONLY)
    try:
        scripted_write(
            ReplacingExifTool(stdout="    1 image files updated\n", rows=GOOD_ROWS), ram_file
        )
        assert read_all(fd) == bytes(size)
    finally:
        os.close(fd)
    assert ram_file.read_bytes() == b"the new file"
    assert [p.name for p in ram_file.parent.iterdir()] == [ram_file.name]


@pytest.mark.parametrize("hard_links", [True, False], ids=["link", "no-hard-links"])
@pytest.mark.parametrize("stderr", ["", "Warning: something\n"], ids=["ok", "refused"])
def test_apply_write_leaves_a_file_it_did_not_replace_alone(
    ram_file: Path, monkeypatch: pytest.MonkeyPatch, hard_links: bool, stderr: str
) -> None:
    # Nothing replaced it, so it is still the caller's file: never zeroed here.
    if not hard_links:

        def no_link(*args: Any, **kwargs: Any) -> None:
            raise OSError(95, "Operation not supported")

        monkeypatch.setattr(os, "link", no_link)
    before = ram_file.read_bytes()
    tool = ScriptedExifTool(stdout="    1 image files updated\n", stderr=stderr, rows=GOOD_ROWS)
    with contextlib.suppress(ExifToolError):
        scripted_write(tool, ram_file)
    assert ram_file.read_bytes() == before
    assert [p.name for p in ram_file.parent.iterdir()] == [ram_file.name]


class KilledExifTool(ScriptedExifTool):
    """Killed mid-write (a timeout): its half-written new file stays behind."""

    tmp_fd: int | None = None

    def execute(self, *args: str) -> ExecResult:
        if "-overwrite_original" in args:
            path = Path(args[-1])
            tmp = path.with_name(path.name + "_exiftool_tmp")
            tmp.write_bytes(b"the start of the new file, photo and all")
            # Kept open, so the test can see whether it was zeroed before removal.
            self.tmp_fd = os.open(tmp, os.O_RDONLY)
            raise ExifToolError("exiftool timed out after 120 s")
        return super().execute(*args)


@pytest.mark.parametrize("hard_links", [True, False], ids=["link", "no-hard-links"])
def test_a_killed_write_leaves_no_half_written_copy(
    ram_file: Path, monkeypatch: pytest.MonkeyPatch, hard_links: bool
) -> None:
    if not hard_links:

        def no_link(*args: Any, **kwargs: Any) -> None:
            raise OSError(95, "Operation not supported")

        monkeypatch.setattr(os, "link", no_link)
    before = ram_file.read_bytes()
    tool = KilledExifTool(stdout="", rows=GOOD_ROWS)
    with pytest.raises(ExifToolError, match="timed out"):
        scripted_write(tool, ram_file)
    assert tool.tmp_fd is not None
    try:
        size = len(b"the start of the new file, photo and all")
        assert read_all(tool.tmp_fd) == bytes(size)  # zeroed, not just unlinked
    finally:
        os.close(tool.tmp_fd)
    assert ram_file.read_bytes() == before  # never replaced, so never zeroed here
    assert [p.name for p in ram_file.parent.iterdir()] == [ram_file.name]


def test_apply_write_refuses_a_symlink(tmp_path: Path) -> None:
    # Writing (and wiping) through a link could reach an original.
    target = tmp_path / "original.jpg"
    target.write_bytes(b"someone's photo")
    link = tmp_path / "x.jpg"
    link.symlink_to(target)
    tool = ScriptedExifTool(stdout="    1 image files updated\n", rows=GOOD_ROWS)
    with pytest.raises(ExifToolError, match="regular file"):
        scripted_write(tool, link)
    assert tool.calls == []
    assert target.read_bytes() == b"someone's photo"


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


def test_the_source_hash_is_recorded_when_known() -> None:
    p = plan()
    with_source = replace(PROV, source_sha256="ab" * 32)
    assert f"-XMP-asif:SourceSHA256={'ab' * 32}" in build_args(p, with_source, lang="en")
    assert not any("SourceSHA256" in a for a in build_args(p, PROV, lang="en"))
    with pytest.raises(ValueError):
        build_args(p, replace(PROV, source_sha256="not-a-hash"), lang="en")
