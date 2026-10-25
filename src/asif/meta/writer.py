"""The additive-only metadata writer (§7, §8 of docs/DESIGN.md).

As If only writes to fields that are empty or already carry As If's own
provenance. It never removes or rewrites your data. Everything it adds is
recorded inside the file (XMP-asif) so ``asif undo`` (M6) can strip exactly
that and nothing else, even without the vault.

What goes where:

- caption -> ``XMP-dc:Description`` (``x-default`` + the caption language),
  only when the photo has no caption yet, or when the one it has is As
  If's and nobody has edited it since (it still hashes to
  ``XMP-asif:CaptionDigest``). A caption in the older fields (IPTC
  Caption-Abstract, EXIF ImageDescription) counts: an XMP one next to it
  would hide it in apps that read XMP first. Camera filler such as
  "OLYMPUS DIGITAL CAMERA" doesn't;
- keywords -> appended to ``XMP-dc:Subject`` (and ``IPTC:Keywords`` with
  ``compat = "wide"``), skipping any already in either, in any letter case.
  When the photo's own keywords are only in IPTC, they are copied into the
  new XMP list first (recorded as copied, never as As If's), so apps that
  read XMP first still show them. A ``Photoshop:IPTCDigest`` that matched
  the IPTC (or the file had no IPTC) is updated along with it, so apps
  following the MWG guidelines keep showing the XMP; where exiftool can't
  do that (TIFF, a digest outside a JPEG's own Photoshop block, or more
  than one), the IPTC is left alone;
- ``XMP-et:OriginalImageHash`` once, never overwritten;
- provenance (``XMP-asif:*``), rewritten every time: tool, model, prompt,
  run, time, and exactly what As If added, including an IPTC record it
  created.

exiftool stamps its name (``XMP-x:XMPToolkit``) on every XMP packet it
writes; a stamp the photo already had is kept as it was.

Planning is pure (``plan_write`` -> ``build_args``), so the rules are tested
exhaustively without exiftool; only ``read_existing`` and ``apply_write``
touch a file.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import secrets
import stat
import unicodedata
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from asif.infer.schema import CAPTION_MAX, KEYWORD_MAX, CaptionResult, clean_text
from asif.meta.exiftool import ExifTool, ExifToolError, check_arg
from asif.workdir import wipe_file

CaptionReason = Literal["empty", "ours", "kept-user", "kept-edited"]

IPTC_KEYWORD_MAX_BYTES = 64
"""IPTC (IIM) allows at most 64 bytes per keyword."""

# Formats with a standard home for IPTC. exiftool writes a non-standard IPTC
# block into PNG (with a warning) and silently writes nothing into WebP, so
# "wide" falls back to XMP-only there.
_IPTC_FILE_TYPES = frozenset({"JPEG", "MPO", "TIFF"})
# Formats where exiftool's Photoshop:IPTCDigest=new matches the IPTC it
# writes. In a TIFF it pads the IPTC to whole 4-byte words after taking the
# digest, so the digest it stores is never the one readers compute.
_IPTC_DIGEST_FILE_TYPES = frozenset({"JPEG", "MPO"})
# Where that digest belongs in a JPEG, as exiftool names the place with
# "-api SavePath -G5" ("-G5:4" adds "Copy1" and so on to further copies).
# exiftool updates (or adds) the one there and leaves any other as it was.
_IPTC_DIGEST_HOME = "JPEG-APP13-Photoshop:IPTCDigest"

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
    "-XMP-x:XMPToolkit",
    "-IPTC:all",  # any IPTC tag at all means the file has an IPTC record
    "-Photoshop:IPTCDigest",
    "-File:CurrentIPTCDigest",
    "-IFD0:ImageDescription",
    "-File:FileType",
)
_ASIF_LISTS = frozenset({"AddedKeywords", "AddedIPTCKeywords", "CopiedKeywords"})

# What cameras put in EXIF ImageDescription when nobody wrote a caption,
# matched against the whole text (whitespace collapsed, any letter case).
# Filler missing from here costs a photo its AI caption (reported as
# kept-user); a pattern that is too loose would hide a real caption, so
# each one spells out the firmware's own wording.
_CAMERA_FILLER = re.compile(
    r"""
    # DIGITAL CAMERA, OLYMPUS DIGITAL CAMERA, LEICA DIGITAL CAMERA, SANYO
    # DIGITAL CAMERA, JENOPTIK DIGITAL CAMERA, MEDION DIGITAL CAMERA: only
    # these brands ("Vintage digital camera" is a caption)
    (?:(?:olympus|leica|sanyo|jenoptik|medion)\s+)?digital\s+(?:still\s+)?camera
    # KODAK Digital Still Camera, KODAK CX7530 ZOOM DIGITAL CAMERA, KODAK
    # EASYSHARE C813 ZOOM DIGITAL CAMERA, KONICA MINOLTA DIGITAL CAMERA,
    # SAMSUNG DIGITAL CAMERA: at most one model name, and it
    # has a digit ("Kodak Brownie, my grandfather's first camera" is a caption)
    | (?:agfa|kodak|konica|minolta|pentax|samsung)(?:\s+(?:minolta|easyshare))?
      (?:\s+[a-z-]*\d[\w-]*)?(?:\s+zoom)?\s+digital\s+(?:still\s+)?camera
    # SAMSUNG CAMERA PICTURES
    | samsung\s+camera\s+pictures?
    # SONY DSC
    | (?:sony\s+)?dsc
    # Exif_JPEG_PICTURE, EXIF_JPEG_422 ("Exif JPEG test" is a caption)
    | exif[\s_]jpeg(?:[\s_](?:picture|\d+))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# exiftool copies the xml:lang of a keyword list's first item onto every
# item it appends, and none of its output formats shows these qualifiers, so
# they are read from the raw XMP packet.
_DC_SUBJECT = "{http://purl.org/dc/elements/1.1/}subject"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_DTD = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)", re.IGNORECASE)

# Why XMP keywords were skipped. Plain words, never the keywords themselves.
_NOTE_LANG_TAGGED = (
    "XMP keywords skipped: the photo's keywords are marked with a language, "
    "and As If's would wrongly get the same mark"
)
_NOTE_IPTC_ONLY = (
    "XMP keywords skipped: they would hide the photo's own IPTC keywords in many "
    "apps, and those aren't marked as UTF-8, so they can't be copied over safely"
)
# exiftool warns while writing a file whose IPTCDigest no longer matches its
# IPTC (an app changed the IPTC without updating it). That was already so,
# and stays so: As If never touches a stale digest, and the protected gate
# checks it. Only this exact warning is let through; any other still
# refuses the write.
_STALE_DIGEST_WARNING = r"^IPTCDigest is not current\. XMP may be out of sync$"
_NOTE_IPTC_DIGEST = (
    "IPTC skipped: the file has a checksum (IPTCDigest) that keeps apps showing "
    "its XMP, and exiftool can't update it in this format"
)
_NOTE_IPTC_DIGEST_PLACE = (
    "IPTC skipped: the file has more than one IPTC checksum (IPTCDigest), or one "
    "in an unusual place, and exiftool can keep only the usual one in step"
)


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
    source_sha256: str = ""
    """SHA-256 of the whole original this copy was made from; "" when not known.

    Lets a later run tell As If's finished copy from one made before the
    original was edited (the image-data hash alone can't see a fixed date)."""


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
    legacy_captions: tuple[str, ...] = ()
    """IPTC:Caption-Abstract and EXIF ImageDescription: the older homes of a caption."""
    subject_langs: bool = False
    """An XMP-dc:Subject item carries an xml:lang tag (or the packet couldn't
    be read to tell), which exiftool would copy onto As If's keywords."""
    xmp_toolkit: str | None = None
    """XMP-x:XMPToolkit, or None when the file has none."""
    iptc_digest: str | None = None
    """Photoshop:IPTCDigest: the MD5 of the IPTC the XMP was last in step
    with. Apps following the MWG guidelines show the XMP only while it
    matches the IPTC (or there is none). None when the file has none."""
    current_iptc_digest: str | None = None
    """File:CurrentIPTCDigest, exiftool's MD5 of the IPTC as it is; None without IPTC."""
    iptc_digest_writable: bool = True
    """exiftool can keep Photoshop:IPTCDigest in step with IPTC it writes (JPEG)."""
    iptc_digest_in_place: bool = True
    """The file has no Photoshop:IPTCDigest, or just one, and it is in a
    JPEG's own Photoshop block (APP13). Some scanners put a second Photoshop
    block in EXIF: exiftool would leave a digest there as it was, and apps
    may go by it."""


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
    copy_subjects: tuple[str, ...] = ()
    """The photo's own IPTC:Keywords, copied into XMP-dc:Subject ahead of
    ``add_subjects`` when it has no XMP keywords yet. Never As If's."""
    all_copied_subjects: tuple[str, ...] = ()
    """Every keyword As If has copied that way, earlier runs included."""
    subject_note: str | None = None
    """Why XMP keywords were skipped, in plain words (never values)."""
    created_iptc: bool = False
    """As If created the IPTC record, in this run or an earlier one."""
    keep_xmp_toolkit: str | None = None
    """The photo's XMP-x:XMPToolkit, kept as it is (exiftool would stamp its
    own); None when it has none."""
    update_iptc_digest: bool = False
    """Set Photoshop:IPTCDigest to the IPTC as written: it was in step (or
    there was no IPTC), so apps following the MWG guidelines showed the XMP,
    and must go on doing so rather than switch to the IPTC."""


@dataclass(frozen=True)
class WriteRecord:
    """What a write actually added, for the undo journal."""

    caption_written: bool
    caption_reason: CaptionReason
    subjects_added: tuple[str, ...]
    iptc_added: tuple[str, ...]
    original_hash_written: bool
    iptc_note: str | None
    subjects_copied: tuple[str, ...] = ()
    """The photo's own IPTC keywords copied into XMP-dc:Subject."""
    iptc_created: bool = False
    """This write created the IPTC record."""
    subject_note: str | None = None
    """Why XMP keywords were skipped, in plain words (never values)."""


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
    toolkit: str | None = None
    iptc_digest: str | None = None
    current_digest: str | None = None
    legacy: tuple[str, ...] = ()
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
        elif group == "XMP-x" and name == "XMPToolkit":
            toolkit = _one(value)
        elif group == "IFD0" and name == "ImageDescription":
            legacy += (_one(value),)
        elif group == "File" and name == "FileType":
            file_type = _one(value)
        elif group == "Photoshop" and name == "IPTCDigest":
            iptc_digest = _one(value)
        elif group == "File" and name == "CurrentIPTCDigest":
            current_digest = _one(value)
        elif _IPTC_GROUP.fullmatch(group):
            iptc_present = True
            if name == "Keywords":
                iptc_keywords += _many(value)
            elif name == "CodedCharacterSet":
                iptc_utf8 = _one(value).strip().upper() == "UTF8"
            elif name == "Caption-Abstract":
                legacy += (_one(value),)

    # Only a list that has items can lend a language tag to new ones.
    langs = bool(subjects) and _subject_langs(et.execute("-b", "-XMP", _file_arg(path)).stdout)
    # The read above shows only one copy of the digest; another says where each is.
    in_place = iptc_digest is None or _iptc_digest_in_place(et, path)
    return ExistingMeta(
        descriptions=descriptions,
        subjects=subjects,
        iptc_present=iptc_present,
        iptc_utf8=iptc_utf8,
        iptc_keywords=iptc_keywords,
        asif=asif,
        original_image_hash=original_hash,
        iptc_supported=file_type in _IPTC_FILE_TYPES,
        legacy_captions=legacy,
        subject_langs=langs,
        xmp_toolkit=toolkit,
        iptc_digest=iptc_digest,
        current_iptc_digest=current_digest,
        iptc_digest_writable=file_type in _IPTC_DIGEST_FILE_TYPES,
        iptc_digest_in_place=in_place,
    )


def _iptc_digest_in_place(et: ExifTool, path: Path) -> bool:
    """Is the file's only Photoshop:IPTCDigest in a JPEG's own Photoshop block?

    Only the tag names are read: they say where each copy is.
    """
    rows = et.execute_json(
        "-a", "-api", "SavePath", "-G5:4", "-Photoshop:IPTCDigest", _file_arg(path)
    )
    places = [key for key in (rows[0] if rows else {}) if key != "SourceFile"]
    return places == [_IPTC_DIGEST_HOME]


def _subject_langs(packet: str) -> bool:
    """Does an XMP-dc:Subject item carry an xml:lang tag?

    Anything that can't be read with certainty counts as yes: the cost is
    skipping keywords, never mislabelling them. (A tag on an enclosing
    element makes exiftool refuse the write with a warning, so only the
    list itself matters.)
    """
    # Some writers pad the packet with NULs, which XML doesn't allow and
    # exiftool passes on; structure.read_xmp accepts them the same way.
    packet = packet.rstrip("\0")
    if not packet.strip():
        return False
    if _DTD.search(packet):
        # XMP never needs one, and a DTD can make a parser expand entities
        # without bound. Without one, expat knows only the five XML entities.
        return True
    try:
        root = ElementTree.fromstring(packet.encode("utf-8"))  # noqa: S314 (no DTD, see above)
    except (ElementTree.ParseError, LookupError, ValueError):
        return True
    return any(
        item.get(_XML_LANG) is not None
        for subject in root.iter(_DC_SUBJECT)
        for item in subject.iter()
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


def _flag(value: Any) -> bool:
    """An XMP boolean As If recorded ("True"/"False")."""
    return _one(value).strip().lower() in ("true", "1")


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

    copy_subjects, add_subjects, subject_note = _subject_rule(existing, keywords)
    add_iptc, create_iptc, update_digest, iptc_note = _iptc_rule(existing, keywords, compat)
    write_hash = existing.original_image_hash is None

    prev_subjects = _many(existing.asif.get("AddedKeywords"))
    prev_copied = _many(existing.asif.get("CopiedKeywords"))
    prev_iptc = _many(existing.asif.get("AddedIPTCKeywords"))
    prev_wrote = _flag(existing.asif.get("WroteOriginalImageHash"))
    prev_created = _flag(existing.asif.get("CreatedIPTC"))
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
        copy_subjects=copy_subjects,
        all_copied_subjects=_clean_unique(prev_copied + copy_subjects, _RECORD_MAX, fold=False),
        subject_note=subject_note,
        # Sticky: on the next run the record exists, so create_iptc is False.
        created_iptc=prev_created or create_iptc,
        keep_xmp_toolkit=existing.xmp_toolkit or None,
        update_iptc_digest=update_digest,
    )


def _caption_rule(existing: ExistingMeta, lang: str) -> tuple[CaptionReason, tuple[str, ...]]:
    # A blank entry holds nothing to protect, so it doesn't count as a caption.
    present = {code: text for code, text in existing.descriptions.items() if text.strip()}
    # IPTC Caption-Abstract and EXIF ImageDescription hold the same caption
    # for older apps. An XMP caption next to one would hide it in every app
    # that reads XMP first.
    legacy = [text for text in existing.legacy_captions if _is_caption(text)]
    if not present:
        return ("kept-user" if legacy else "empty"), ()
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
    # As If never writes the older fields. Apps that keep them in step may
    # copy its caption there; any other text in them is someone's edit.
    if any(_digest(text) != digest for text in legacy):
        return "kept-edited", ()
    keep = {"x-default", lang.casefold()}
    return "ours", tuple(code for code in present if code.casefold() not in keep)


def _is_caption(text: str) -> bool:
    """Is an older caption field more than blank padding or camera filler?"""
    words = " ".join(text.replace("\0", " ").split())
    return bool(words) and not _CAMERA_FILLER.fullmatch(words)


def _subject_rule(
    existing: ExistingMeta, keywords: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """The photo's IPTC keywords to copy into XMP-dc:Subject first, the
    keywords to add after them, and why none, if none."""
    # A keyword the photo has in IPTC is the owner's, even if XMP lacks it.
    have = {_key(k) for k in (*existing.subjects, *existing.iptc_keywords)}
    add = tuple(k for k in keywords if _key(k) not in have)
    if not add:
        return (), (), None
    if existing.subject_langs:
        return (), (), _NOTE_LANG_TAGGED
    if existing.subjects or not existing.iptc_keywords:
        return (), add, None
    # The owner's keywords are only in IPTC. Apps that read XMP first would
    # show As If's alone and hide theirs, so the new XMP list starts with a
    # copy of theirs, as those apps do themselves when they save keywords.
    if not existing.iptc_utf8 and not all(k.isascii() for k in existing.iptc_keywords):
        # IPTC not marked as UTF-8 may be Latin-1 or unmarked UTF-8; a wrong
        # guess would copy the owner's keywords garbled.
        return (), (), _NOTE_IPTC_ONLY
    return _clean_unique(existing.iptc_keywords, _RECORD_MAX, fold=False), add, None


def _iptc_rule(
    existing: ExistingMeta, keywords: tuple[str, ...], compat: str
) -> tuple[tuple[str, ...], bool, bool, str | None]:
    """IPTC keywords to add, whether to create the record, whether to update
    Photoshop:IPTCDigest, and why not, if not."""
    if compat != "wide":
        return (), False, False, None
    if not existing.iptc_supported:
        return (), False, False, "IPTC skipped: this file format has no standard place for it"
    if existing.iptc_present and not existing.iptc_utf8:
        # Old apps wrote Latin-1 IPTC without saying so. Adding UTF-8 to it, or
        # relabelling it as UTF-8, would garble the owner's keywords.
        return (), False, False, "IPTC left alone: the file's IPTC isn't marked as UTF-8"
    if existing.iptc_digest is not None and not existing.iptc_digest_writable:
        # Changing the IPTC without its digest could make apps switch from
        # the owner's XMP to the IPTC.
        return (), False, False, _NOTE_IPTC_DIGEST
    if not existing.iptc_digest_in_place:
        # The same, for a digest exiftool leaves as it was: a second copy,
        # or one in a Photoshop block of a scanner's EXIF.
        return (), False, False, _NOTE_IPTC_DIGEST_PLACE
    have = {_key(k) for k in existing.iptc_keywords}
    new = [k for k in keywords if _key(k) not in have]
    fits = tuple(k for k in new if len(k.encode("utf-8")) <= IPTC_KEYWORD_MAX_BYTES)
    note = None
    if skipped := len(new) - len(fits):
        plural = "keyword is" if skipped == 1 else "keywords are"
        note = f"{skipped} {plural} too long for IPTC (64 bytes max), so XMP only"
    # Apps following the MWG guidelines show the XMP while the digest matches
    # the IPTC (or there is no IPTC), and the IPTC once it doesn't. Shown
    # before, the XMP stays shown: the digest follows the new IPTC. Already
    # stale, the IPTC is shown either way, so the digest is left as it is.
    in_step = existing.current_iptc_digest in (None, existing.iptc_digest)
    update = bool(fits) and existing.iptc_digest is not None and in_step
    return fits, bool(fits) and not existing.iptc_present, update, note


def build_args(plan: WritePlan, prov: Provenance, *, lang: str) -> list[str]:
    """The exact exiftool arguments for ``plan``, without the file name.

    Pure. Every value goes through ``clean_text`` again (defence in depth),
    and every argument through ``check_arg``. Values always follow the first
    ``=`` of an assignment, so to exiftool they are data, never options.
    """
    _check_lang(lang)
    args = ["-overwrite_original", "-P", "-api", f"NoWarning={_STALE_DIGEST_WARNING}"]

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

    args += [f"-XMP-dc:Subject+={k}" for k in (*_written_copies(plan), *_written_subjects(plan))]
    if plan.keep_xmp_toolkit is not None:
        # Copied inside exiftool, so the value itself never becomes an argument.
        args.append("-XMP-x:XMPToolkit<XMP-x:XMPToolkit")

    iptc = _written_iptc(plan)
    if iptc:
        if plan.create_iptc:
            args += ["-charset", "iptc=UTF8", "-IPTC:CodedCharacterSet=UTF8"]
        args += [f"-IPTC:Keywords+={k}" for k in iptc]
        if plan.update_iptc_digest:
            # exiftool takes the MD5 of the IPTC it writes, so no value of
            # the photo's becomes an argument. Not "-=old" as well: with no
            # IPTC before there is no old digest, and nothing would change.
            args.append("-Photoshop:IPTCDigest=new")

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
    if prov.source_sha256:
        _check_hash(prov.source_sha256)
        args.append(f"-XMP-asif:SourceSHA256={prov.source_sha256}")
    if plan.caption_digest is not None:
        _check_hash(plan.caption_digest)
        args.append(f"-XMP-asif:CaptionDigest={plan.caption_digest}")
    if plan.caption_lang is not None:
        _check_lang(plan.caption_lang)
        args.append(f"-XMP-asif:CaptionLang={plan.caption_lang}")
    args += _list_args("AddedKeywords", _written_record(plan.all_added_subjects))
    args += _list_args("CopiedKeywords", _written_record(plan.all_copied_subjects))
    args += _list_args("AddedIPTCKeywords", _written_record(plan.all_added_iptc))
    args.append(f"-XMP-asif:WroteOriginalImageHash={plan.wrote_original_hash}")
    args.append(f"-XMP-asif:CreatedIPTC={plan.created_iptc}")

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


def _written_copies(plan: WritePlan) -> tuple[str, ...]:
    # The owner's own spellings, every one of them: no case folding.
    return _clean_unique(plan.copy_subjects, _RECORD_MAX, fold=False)


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
    and modification times are kept to the nanosecond, and the copy of the
    photo the write replaces is overwritten with zeros. Raises
    ``ExifToolError`` if the file isn't a regular file, exiftool fails,
    warns about anything or doesn't report the file as updated, or the
    result doesn't read back as planned.
    """
    args = build_args(plan, prov, lang=lang)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ExifToolError(f"can't read the file to write to: {exc.strerror}") from exc
    if not stat.S_ISREG(before.st_mode):
        # A link could lead the write, and the wipe after it, to another file.
        raise ExifToolError("refusing to write to anything but a regular file")
    with _wiping_replaced_inode(path):
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
        subjects_copied=_written_copies(plan),
        iptc_created=plan.create_iptc and bool(_written_iptc(plan)),
        subject_note=plan.subject_note,
    )


@contextlib.contextmanager
def _wiping_replaced_inode(path: Path) -> Iterator[None]:
    """Zero the file ``path`` names now, once a write has replaced it.

    exiftool never writes in place: it writes a new file and renames it over
    ``path``. The old inode, the whole photo as it was, is unlinked without
    being overwritten, and wiping ``path`` later only reaches the new one.
    A second name for it (or, where hard links aren't possible, as on FAT,
    an open descriptor) keeps it reachable so it can be zeroed. Nothing is
    zeroed while anything else still names it: a write that didn't replace
    it leaves the caller's file as it was.

    exiftool's half-written new file (``<name>_exiftool_tmp``) is wiped too:
    exiftool removes it itself, unless it was killed mid-write (a timeout).
    """
    name = path.with_name(f"{secrets.token_hex(8)}.pre-write")
    link: Path | None = name
    fd: int | None = None
    try:
        os.link(path, name)
    except OSError:
        link = None
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        except OSError as exc:
            raise ExifToolError(f"can't open the file to write to: {exc.strerror}") from exc
    try:
        yield
    finally:
        # Nothing writes it any more: exiftool finished, failed (and removed
        # it) or was killed and waited for. After a write it was renamed
        # over ``path``, so nothing is there.
        with contextlib.suppress(OSError):  # the inode below still gets its turn
            wipe_file(path.with_name(path.name + "_exiftool_tmp"))
        if link is not None:
            try:
                only_ours = link.stat().st_nlink == 1
            except OSError:
                only_ours = False
            if only_ours:
                wipe_file(link)
            else:
                link.unlink(missing_ok=True)
        elif fd is not None:
            try:
                if os.fstat(fd).st_nlink == 0:
                    _zero_fd(fd)
            finally:
                os.close(fd)


def _zero_fd(fd: int) -> None:
    """Overwrite an open file with zeros: ``wipe_file`` for a file with no name left."""
    size = os.fstat(fd).st_size
    chunk = bytes(min(size, 1 << 20))
    offset = 0
    with contextlib.suppress(OSError):  # best effort, as in wipe_file
        while offset < size:
            written = os.pwrite(fd, chunk[: size - offset], offset)
            if written <= 0:
                break
            offset += written
        os.fsync(fd)


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
    subjects = set(_written_copies(plan)) | set(_written_subjects(plan))
    # The exiftool reads behind every other check can't see language tags.
    if not subjects <= set(after.subjects) or (subjects and after.subject_langs):
        wrong.append("XMP-dc:Subject")
    if not set(_written_iptc(plan)) <= set(after.iptc_keywords):
        wrong.append("IPTC:Keywords")
    if (
        plan.update_iptc_digest
        and _written_iptc(plan)
        and (after.iptc_digest is None or after.iptc_digest != after.current_iptc_digest)
    ):
        wrong.append("Photoshop:IPTCDigest")
    if plan.write_original_hash and after.original_image_hash != plan.image_hash:
        wrong.append("XMP-et:OriginalImageHash")
    if plan.keep_xmp_toolkit is not None and after.xmp_toolkit != plan.keep_xmp_toolkit:
        wrong.append("XMP-x:XMPToolkit")
    for tag, values in (
        ("AddedKeywords", plan.all_added_subjects),
        ("CopiedKeywords", plan.all_copied_subjects),
        ("AddedIPTCKeywords", plan.all_added_iptc),
    ):
        if _many(after.asif.get(tag)) != _written_record(values):
            wrong.append(f"XMP-asif:{tag}")
    if _flag(after.asif.get("CreatedIPTC")) != plan.created_iptc:
        wrong.append("XMP-asif:CreatedIPTC")
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
