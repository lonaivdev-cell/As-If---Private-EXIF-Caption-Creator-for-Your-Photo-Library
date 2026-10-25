"""The additive-only metadata writer (§7, §8 of docs/DESIGN.md).

As If only writes to fields that are empty or already carry As If's own
provenance. It never removes or rewrites your data. Everything it adds is
recorded inside the file (XMP-asif) so ``asif undo`` (M6) can strip exactly
that and nothing else, even without the vault.

What goes where:

- caption -> ``XMP-dc:Description`` (``x-default`` + the caption language),
  only when the photo has no description yet, or when the one it has is As
  If's and nobody has edited it since (it still hashes to
  ``XMP-asif:CaptionDigest``);
- keywords -> appended to ``XMP-dc:Subject`` (and ``IPTC:Keywords`` with
  ``compat = "wide"``), skipping any already there in any letter case;
- ``XMP-et:OriginalImageHash`` once, never overwritten;
- provenance (``XMP-asif:*``), rewritten every time: tool, model, prompt,
  run, time, and exactly what As If added.

Planning is pure (``plan_write`` -> ``build_args``), so the rules are tested
exhaustively without exiftool; only ``read_existing`` and ``apply_write``
touch a file.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from asif.infer.schema import CAPTION_MAX, KEYWORD_MAX, CaptionResult, clean_text
from asif.meta.exiftool import ExifTool, ExifToolError, check_arg

CaptionReason = Literal["empty", "ours", "kept-user", "kept-edited"]

IPTC_KEYWORD_MAX_BYTES = 64
"""IPTC (IIM) allows at most 64 bytes per keyword."""

# Formats with a standard home for IPTC. exiftool writes a non-standard IPTC
# block into PNG (with a warning) and silently writes nothing into WebP, so
# "wide" falls back to XMP-only there.
_IPTC_FILE_TYPES = frozenset({"JPEG", "MPO", "TIFF"})

# Provenance values are short identifiers. Keywords recorded by an earlier run
# are re-recorded whole (undo must match them exactly), within reason.
_PROV_MAX = 200
_RECORD_MAX = 200

# All matched with fullmatch: "$" alone would also accept a trailing newline.
_HEX64 = re.compile(r"[0-9a-f]{64}")
_TIMESTAMP = re.compile(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}Z")
# A language tag ends up inside an exiftool *tag name*, so only letters,
# digits and hyphens get through: "en", "pt-BR", "zh-Hant-TW", "es-419".
_LANG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}")
_IPTC_GROUP = re.compile(r"IPTC\d*")

# XML 1.0 forbids these two; clean_text lets them through and exiftool turns
# them into "?", which would make the caption read back differently.
_XML_FORBIDDEN = {0xFFFE: " ", 0xFFFF: " "}

# exiftool's JSON turns "2019" into 2019, "1.50" into 1.50 and "TRUE" into
# true, and 12.76 has no option to quote them (StructFormat=JSONQ only covers
# structures). Its Filter option, a Perl expression run on every returned value
# (list items one by one), can: prefix each value so none looks like a number
# or a boolean, and strip the prefix here. The expression is a constant; tag
# values are only ever data inside it, never code.
_MARK = "s:"
_READ_FILTER = f'Filter=$_ = "{_MARK}$_"'
_READ_TAGS = (
    "-XMP-dc:Description*",  # every language; plain "Description" is x-default
    "-XMP-dc:Subject",
    "-XMP-asif:all",
    "-XMP-et:OriginalImageHash",
    "-IPTC:all",  # any IPTC tag at all means the file has an IPTC record
    "-File:FileType",
)
_ASIF_LISTS = frozenset({"AddedKeywords", "AddedIPTCKeywords"})


@dataclass(frozen=True)
class Provenance:
    """Who wrote this caption, recorded in XMP-asif with every write."""

    tool_version: str
    model_id: str
    model_sha256: str
    """SHA-256 of the model weights; "" when unknown (e.g. the stub model)."""
    prompt_version: str
    run_id: str
    timestamp: str
    """UTC, in exiftool's format: ``YYYY:MM:DD HH:MM:SSZ``."""


@dataclass(frozen=True)
class ExistingMeta:
    """What a file already carries, as far as the writer's rules care.

    Values are always ``str`` (never exiftool's numbers or booleans), and list
    tags are tuples even when they hold a single item.
    """

    descriptions: dict[str, str] = field(default_factory=dict)
    """XMP-dc:Description by language: ``"x-default"`` and every other one present."""
    subjects: tuple[str, ...] = ()
    """XMP-dc:Subject."""
    iptc_present: bool = False
    """The file has an IPTC (IIM) record."""
    iptc_utf8: bool = False
    """IPTC:CodedCharacterSet says UTF-8."""
    iptc_keywords: tuple[str, ...] = ()
    asif: dict[str, Any] = field(default_factory=dict)
    """XMP-asif:* by tag name (no group). AddedKeywords/AddedIPTCKeywords are tuples."""
    original_image_hash: str | None = None
    """XMP-et:OriginalImageHash, or None when the file has none."""
    iptc_supported: bool = True
    """The format has a standard place for IPTC (JPEG, TIFF)."""


@dataclass(frozen=True)
class WritePlan:
    """Exactly what one write will do, decided before anything is written."""

    caption: str | None
    """The caption to write; None leaves every description alone."""
    caption_reason: CaptionReason
    """``empty`` (no description in any language), ``ours`` (every description
    still hashes to As If's CaptionDigest), ``kept-user`` (As If never wrote
    one) or ``kept-edited`` (As If wrote one, but it was edited since or other
    languages were added)."""
    remove_caption_langs: tuple[str, ...]
    """Only for ``ours``: languages of As If's previous caption that the new
    one doesn't overwrite (the old language, when it changed)."""
    add_subjects: tuple[str, ...]
    """Keywords not already in XMP-dc:Subject (in any letter case), in model order."""
    add_iptc: tuple[str, ...]
    """The same for IPTC:Keywords (compat "wide" only)."""
    create_iptc: bool
    """The file has no IPTC yet: create it, marked UTF-8."""
    iptc_note: str | None
    """Why IPTC was skipped or trimmed, in plain words (counts, never values)."""
    write_original_hash: bool
    """The file has no XMP-et:OriginalImageHash yet."""
    image_hash: str
    """The SHA-256 ImageDataHash to record."""
    all_added_subjects: tuple[str, ...]
    """Every keyword As If has added to XMP-dc:Subject, earlier runs included."""
    all_added_iptc: tuple[str, ...]
    """Every keyword As If has added to IPTC:Keywords, earlier runs included."""
    wrote_original_hash: bool
    """As If wrote the OriginalImageHash, in this run or an earlier one."""
    caption_digest: str | None
    """SHA-256 (hex) of the caption written; None when the caption is left alone."""
    caption_lang: str | None
    """Language of the caption written; None when the caption is left alone."""


@dataclass(frozen=True)
class WriteRecord:
    """What a write actually added, for the undo journal."""

    caption_written: bool
    caption_reason: CaptionReason
    subjects_added: tuple[str, ...]
    iptc_added: tuple[str, ...]
    original_hash_written: bool
    iptc_note: str | None


# -- reading -----------------------------------------------------------------


def read_existing(et: ExifTool, path: Path) -> ExistingMeta:
    """Read the tags the writer's rules depend on."""
    rows = et.execute_json("-G1", "-a", "-api", _READ_FILTER, *_READ_TAGS, _file_arg(path))
    if not rows:
        raise ExifToolError("exiftool returned nothing for the file")

    descriptions: dict[str, str] = {}
    subjects: tuple[str, ...] = ()
    iptc_present = iptc_utf8 = False
    iptc_keywords: tuple[str, ...] = ()
    asif: dict[str, Any] = {}
    original_hash: str | None = None
    file_type = ""
    for key, raw in rows[0].items():
        if key == "SourceFile":
            continue
        group, _, name = key.partition(":")
        value = _unmark(raw, key)
        if group == "XMP-dc" and name == "Description":
            descriptions["x-default"] = _one(value)
        elif group == "XMP-dc" and name.startswith("Description-"):
            lang = name.removeprefix("Description-")  # may be "" for xml:lang=""
            descriptions["x-default" if lang.casefold() == "x-default" else lang] = _one(value)
        elif group == "XMP-dc" and name == "Subject":
            subjects = _many(value)
        elif group == "XMP-asif":
            asif[name] = _many(value) if name in _ASIF_LISTS else value
        elif group == "XMP-et" and name == "OriginalImageHash":
            original_hash = _one(value)
        elif group == "File" and name == "FileType":
            file_type = _one(value)
        elif _IPTC_GROUP.fullmatch(group):
            iptc_present = True
            if name == "Keywords":
                iptc_keywords += _many(value)
            elif name == "CodedCharacterSet":
                iptc_utf8 = _one(value).strip().upper() == "UTF8"

    return ExistingMeta(
        descriptions=descriptions,
        subjects=subjects,
        iptc_present=iptc_present,
        iptc_utf8=iptc_utf8,
        iptc_keywords=iptc_keywords,
        asif=asif,
        original_image_hash=original_hash,
        iptc_supported=file_type in _IPTC_FILE_TYPES,
    )


def _unmark(raw: Any, key: str) -> str | tuple[str, ...]:
    """Undo ``_READ_FILTER``. A value without the mark means the filter didn't
    run, so the value may have been mangled: refuse rather than guess."""
    if isinstance(raw, list):
        return tuple(_unmark_one(item, key) for item in raw)
    return _unmark_one(raw, key)


def _unmark_one(raw: Any, key: str) -> str:
    if isinstance(raw, str) and raw.startswith(_MARK):
        return raw[len(_MARK) :]
    raise ExifToolError(f"exiftool returned {key} in an unexpected form")


def _one(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ", ".join(value)  # exiftool's own way of showing a list as one value
    return str(value)


def _many(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(str(v) for v in value)
    return (str(value),)


# -- planning (pure) ---------------------------------------------------------


def plan_write(
    existing: ExistingMeta,
    result: CaptionResult,
    *,
    lang: str,
    compat: Literal["xmp", "wide"],
    image_hash: str,
) -> WritePlan:
    """Decide what to write, following the additive-only rule.

    Pure: the same inputs always give the same plan. The caption and keywords
    are cleaned again here, so the recorded digest is of the exact text
    written. Raises ``ValueError`` for a malformed language, image hash or
    ``compat``, or a caption that cleans down to nothing.
    """
    _check_lang(lang)
    _check_hash(image_hash)
    if compat not in ("xmp", "wide"):
        raise ValueError(f"compat must be 'xmp' or 'wide', not {compat!r}")
    caption = _clean(result.caption, CAPTION_MAX)
    if not caption:
        raise ValueError("the caption is empty after cleaning")
    keywords = _clean_unique(result.keywords, KEYWORD_MAX, fold=True)

    reason, remove = _caption_rule(existing, lang)
    write_caption = reason in ("empty", "ours")

    have = {_key(s) for s in existing.subjects}
    add_subjects = tuple(k for k in keywords if _key(k) not in have)
    add_iptc, create_iptc, iptc_note = _iptc_rule(existing, keywords, compat)
    write_hash = existing.original_image_hash is None

    prev_subjects = _many(existing.asif.get("AddedKeywords"))
    prev_iptc = _many(existing.asif.get("AddedIPTCKeywords"))
    prev_wrote = _one(existing.asif.get("WroteOriginalImageHash")).strip().lower() in ("true", "1")
    return WritePlan(
        caption=caption if write_caption else None,
        caption_reason=reason,
        remove_caption_langs=remove,
        add_subjects=add_subjects,
        add_iptc=add_iptc,
        create_iptc=create_iptc,
        iptc_note=iptc_note,
        write_original_hash=write_hash,
        image_hash=image_hash,
        # Exact-string dedupe: if the owner removed "Dog" and As If later added
        # "dog", undo must know about both spellings.
        all_added_subjects=_clean_unique(prev_subjects + add_subjects, _RECORD_MAX, fold=False),
        all_added_iptc=_clean_unique(prev_iptc + add_iptc, _RECORD_MAX, fold=False),
        wrote_original_hash=prev_wrote or write_hash,
        caption_digest=_digest(caption) if write_caption else None,
        caption_lang=lang if write_caption else None,
    )


def _caption_rule(existing: ExistingMeta, lang: str) -> tuple[CaptionReason, tuple[str, ...]]:
    # A blank entry holds nothing to protect, so it doesn't count as a caption.
    present = {code: text for code, text in existing.descriptions.items() if text.strip()}
    if not present:
        return "empty", ()
    digest = _one(existing.asif.get("CaptionDigest"))
    if not digest:
        return "kept-user", ()
    # As If writes x-default plus one language. Anything else was added by
    # someone else; a malformed recorded language is treated the same way.
    old_lang = _one(existing.asif.get("CaptionLang"))
    ours = {"x-default"} | ({old_lang.casefold()} if _is_lang(old_lang) else set())
    for code, text in present.items():
        if code.casefold() not in ours or _digest(text) != digest:
            return "kept-edited", ()
    keep = {"x-default", lang.casefold()}
    return "ours", tuple(code for code in present if code.casefold() not in keep)


def _iptc_rule(
    existing: ExistingMeta, keywords: tuple[str, ...], compat: str
) -> tuple[tuple[str, ...], bool, str | None]:
    """IPTC keywords to add, whether to create the record, and why not, if not."""
    if compat != "wide":
        return (), False, None
    if not existing.iptc_supported:
        return (), False, "IPTC skipped: this file format has no standard place for it"
    if existing.iptc_present and not existing.iptc_utf8:
        # Old apps wrote Latin-1 IPTC without saying so. Adding UTF-8 to it, or
        # relabelling it as UTF-8, would garble the owner's keywords.
        return (), False, "IPTC left alone: the file's IPTC isn't marked as UTF-8"
    have = {_key(k) for k in existing.iptc_keywords}
    new = [k for k in keywords if _key(k) not in have]
    fits = tuple(k for k in new if len(k.encode("utf-8")) <= IPTC_KEYWORD_MAX_BYTES)
    note = None
    if skipped := len(new) - len(fits):
        plural = "keyword is" if skipped == 1 else "keywords are"
        note = f"{skipped} {plural} too long for IPTC (64 bytes max), so XMP only"
    return fits, bool(fits) and not existing.iptc_present, note


def build_args(plan: WritePlan, prov: Provenance, *, lang: str) -> list[str]:
    """The exact exiftool arguments for ``plan``, without the file name.

    Pure. Every value goes through ``clean_text`` again (defence in depth),
    and every argument through ``check_arg``. Values always follow the first
    ``=`` of an assignment, so to exiftool they are data, never options.
    """
    _check_lang(lang)
    args = ["-overwrite_original", "-P"]

    if plan.caption is not None:
        caption = _clean(plan.caption, CAPTION_MAX)
        if not caption:
            # An empty assignment means "delete" to exiftool.
            raise ValueError("refusing to write an empty caption")
        args += [
            f"-XMP-dc:Description-x-default={caption}",
            f"-XMP-dc:Description-{lang}={caption}",
        ]
        for old in plan.remove_caption_langs:
            _check_lang(old)
            if old.casefold() == lang.casefold():
                raise ValueError("a caption language can't be both written and removed")
            args.append(f"-XMP-dc:Description-{old}=")
    elif plan.remove_caption_langs:
        raise ValueError("caption languages are only removed when a new caption replaces them")

    args += [f"-XMP-dc:Subject+={k}" for k in _written_subjects(plan)]

    iptc = _written_iptc(plan)
    if iptc:
        if plan.create_iptc:
            args += ["-charset", "iptc=UTF8", "-IPTC:CodedCharacterSet=UTF8"]
        args += [f"-IPTC:Keywords+={k}" for k in iptc]

    if plan.write_original_hash:
        _check_hash(plan.image_hash)
        args += [
            f"-XMP-et:OriginalImageHash={plan.image_hash}",
            "-XMP-et:OriginalImageHashType=SHA256",
        ]

    if not _TIMESTAMP.fullmatch(prov.timestamp):
        raise ValueError("the provenance timestamp must be UTC 'YYYY:MM:DD HH:MM:SSZ'")
    args += [
        f"-XMP-asif:ToolVersion={_clean(prov.tool_version, _PROV_MAX)}",
        f"-XMP-asif:ModelId={_clean(prov.model_id, _PROV_MAX)}",
        f"-XMP-asif:ModelSHA256={_clean(prov.model_sha256, _PROV_MAX)}",
        f"-XMP-asif:PromptVersion={_clean(prov.prompt_version, _PROV_MAX)}",
        f"-XMP-asif:RunId={_clean(prov.run_id, _PROV_MAX)}",
        f"-XMP-asif:Timestamp={prov.timestamp}",
    ]
    if plan.caption_digest is not None:
        _check_hash(plan.caption_digest)
        args.append(f"-XMP-asif:CaptionDigest={plan.caption_digest}")
    if plan.caption_lang is not None:
        _check_lang(plan.caption_lang)
        args.append(f"-XMP-asif:CaptionLang={plan.caption_lang}")
    args += _list_args("AddedKeywords", _written_record(plan.all_added_subjects))
    args += _list_args("AddedIPTCKeywords", _written_record(plan.all_added_iptc))
    args.append(f"-XMP-asif:WroteOriginalImageHash={plan.wrote_original_hash}")

    for arg in args:
        check_arg(arg)
    return args


def _list_args(tag: str, values: tuple[str, ...]) -> list[str]:
    # Several assignments in one command set exactly these items; a single
    # empty one clears the list.
    if not values:
        return [f"-XMP-asif:{tag}="]
    return [f"-XMP-asif:{tag}={v}" for v in values]


def _written_subjects(plan: WritePlan) -> tuple[str, ...]:
    return _clean_unique(plan.add_subjects, KEYWORD_MAX, fold=True)


def _written_iptc(plan: WritePlan) -> tuple[str, ...]:
    iptc = _clean_unique(plan.add_iptc, KEYWORD_MAX, fold=True)
    if any(len(k.encode("utf-8")) > IPTC_KEYWORD_MAX_BYTES for k in iptc):
        raise ValueError("an IPTC keyword is longer than 64 bytes")
    return iptc


def _written_record(values: tuple[str, ...]) -> tuple[str, ...]:
    return _clean_unique(values, _RECORD_MAX, fold=False)


# -- writing -----------------------------------------------------------------


def apply_write(
    et: ExifTool, path: Path, plan: WritePlan, prov: Provenance, *, lang: str
) -> WriteRecord:
    """Write ``plan`` into ``path`` in place, then read it back.

    The caller hands this a RAM copy, never an original. The file's access
    and modification times are kept to the nanosecond. Raises
    ``ExifToolError`` if exiftool fails, warns about anything, doesn't
    report the file as updated, or the result doesn't read back as planned.
    """
    args = build_args(plan, prov, lang=lang)
    try:
        before = path.stat()
    except OSError as exc:
        raise ExifToolError(f"can't read the file to write to: {exc.strerror}") from exc
    result = et.execute(*args, _file_arg(path))
    if result.warnings:
        raise ExifToolError(
            f"exiftool warned about the write ({len(result.warnings)} warning(s))", result.stderr
        )
    if not re.search(r"^\s*1 image files updated\s*$", result.stdout, re.MULTILINE):
        raise ExifToolError("exiftool didn't report the file as updated", result.stderr)

    _verify(read_existing(et, path), plan, lang)
    # -P keeps the times only to the second; put back the exact ones.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    return WriteRecord(
        caption_written=plan.caption is not None,
        caption_reason=plan.caption_reason,
        subjects_added=_written_subjects(plan),
        iptc_added=_written_iptc(plan),
        original_hash_written=plan.write_original_hash,
        iptc_note=plan.iptc_note,
    )


def _verify(after: ExistingMeta, plan: WritePlan, lang: str) -> None:
    """Check the file now holds what the plan wrote, so the undo record is true."""
    wrong: list[str] = []
    if plan.caption is not None:
        caption = _clean(plan.caption, CAPTION_MAX)
        # exiftool normalises the case of language tags ("pt-br" -> "pt-BR").
        langs = {code.casefold(): text for code, text in after.descriptions.items()}
        stale = any(old.casefold() in langs for old in plan.remove_caption_langs)
        if langs.get("x-default") != caption or langs.get(lang.casefold()) != caption or stale:
            wrong.append("XMP-dc:Description")
        if _one(after.asif.get("CaptionDigest")) != plan.caption_digest:
            wrong.append("XMP-asif:CaptionDigest")
    if not set(_written_subjects(plan)) <= set(after.subjects):
        wrong.append("XMP-dc:Subject")
    if not set(_written_iptc(plan)) <= set(after.iptc_keywords):
        wrong.append("IPTC:Keywords")
    if plan.write_original_hash and after.original_image_hash != plan.image_hash:
        wrong.append("XMP-et:OriginalImageHash")
    for tag, values in (
        ("AddedKeywords", plan.all_added_subjects),
        ("AddedIPTCKeywords", plan.all_added_iptc),
    ):
        if _many(after.asif.get(tag)) != _written_record(values):
            wrong.append(f"XMP-asif:{tag}")
    if wrong:
        raise ExifToolError(f"the write didn't read back as planned: {', '.join(wrong)}")


# -- helpers -----------------------------------------------------------------


def _file_arg(path: Path) -> str:
    # Absolute, so a file named like an option ("-o") can't be read as one.
    return str(path.absolute())


def _clean(value: str, max_len: int) -> str:
    return clean_text(value.translate(_XML_FORBIDDEN), max_len)


def _clean_unique(values: Iterable[str], max_len: int, *, fold: bool) -> tuple[str, ...]:
    """Cleaned, non-empty values, first occurrence of each (case-insensitive if ``fold``)."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        word = _clean(value, max_len)
        key = _key(word) if fold else word
        if word and key not in seen:
            seen.add(key)
            out.append(word)
    return tuple(out)


def _key(word: str) -> str:
    """Comparison key for keywords: "Café", "café" and a decomposed "café" are one."""
    return " ".join(unicodedata.normalize("NFC", word).split()).casefold()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_lang(lang: str) -> bool:
    # "all" is refused too: exiftool turns it into a wildcard inside tag names.
    return bool(_LANG.fullmatch(lang)) and "all" not in lang.lower().split("-")


def _check_lang(lang: str) -> None:
    if not _is_lang(lang):
        raise ValueError("the caption language isn't a BCP-47 tag like 'en' or 'pt-BR'")


def _check_hash(value: str) -> None:
    if not _HEX64.fullmatch(value):
        raise ValueError("expected a lowercase hex SHA-256")
