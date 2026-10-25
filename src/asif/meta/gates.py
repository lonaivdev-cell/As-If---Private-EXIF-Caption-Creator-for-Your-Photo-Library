"""The integrity gates (§7 of docs/DESIGN.md). Every one must pass before a
processed copy is allowed to replace anything.

Before writing, ``snapshot`` records the RAM copy: its image-data hash, a
full tag dump, the XMP's own text and (for JPEG, PNG and WebP) its bytes.
It also says when a write to the file *can't* be proven safe
(``Snapshot.unprovable``): such a file is refused, never written. After
writing, ``check_written`` runs, in order:

1. ``equality``: exiftool's SHA-256 ``ImageDataHash`` is unchanged.
2. ``protected``: every tag outside the ones As If writes is unchanged,
   and the ones it writes only gained values (additive-only). With compat
   "wide", a ``Photoshop:IPTCDigest`` in step with the IPTC stays in step,
   so apps keep showing the XMP they showed.
3. ``trailer``: a JPEG's bytes after the primary image are unchanged
   (motion-photo video, MPF images), and must be provably so.
4. ``structure``: ``asif.meta.structure.compare``, byte for byte, and
   every value of the original XMP still there, read without exiftool.
5. ``size``: the file grew by less than 64 KiB.

``check_source_unchanged`` is the design's gate 4 (``remote-unchanged``).

Gate details name tags, segments and byte counts, never values: a value can
be a caption, a keyword or GPS coordinates.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import mmap
import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from asif.meta import structure
from asif.meta.exiftool import ExifTool, ExifToolError
from asif.meta.structure import StructureError, detect_format, jpeg_trailer

MAX_GROWTH = 64 * 1024
"""The size gate: a write may add at most this many bytes, minus one."""

_HASH_CHUNK = 1 << 20

# Formats whose bytes are kept in the snapshot for the trailer and structure
# gates. TIFFs can be huge, and exiftool rebuilds them, so their bytes can't
# match anyway: they are checked through a memory map instead.
_KEEP_BYTES = frozenset({"jpeg", "png", "webp"})

# The tag dump. Deliberately stricter than a plain ``-json -G1 -a -b``:
#   -n        raw values. Print conversion rounds: a GPS latitude moved by
#             1e-7 degrees prints exactly the same.
#   -G1:4     with plain -G1, exiftool's JSON silently drops the second of
#             two same-named tags in one group (two COM segments, say).
#             Family 4 tells the copies apart; ``_dump_tags`` renumbers them.
#   -u        unknown tags too (a scanner's private TIFF tags, say): without
#             it they aren't compared at all.
#   -b        binary values come back base64-encoded, compared byte for byte.
#   -ICC_Profile -MakerNotes
#             the blocks themselves; -all:all only lists the tags decoded
#             from them. exiftool rebuilds a maker note when it reads it, so
#             this is its decoding, not the raw bytes: the structure gate
#             compares those (JPEG, PNG, WebP), and a TIFF with a maker note
#             is refused (``Snapshot.unprovable``).
_DUMP_ARGS = ("-json", "-n", "-G1:4", "-a", "-u", "-b", "-all:all", "-ICC_Profile", "-MakerNotes")
# File-system dates, exiftool's own version and values derived from other tags.
_SKIP_GROUPS = frozenset({"System", "ExifTool", "Composite"})

# -- what a write may change --------------------------------------------------

# The tags As If writes (§8). Matched on group and name, any copy.
_OURS = frozenset(
    {
        "XMP-dc:Subject",
        "XMP-et:OriginalImageHash",
        "XMP-et:OriginalImageHashType",
        # exiftool stamps its own name and version on every XMP packet it writes.
        "XMP-x:XMPToolkit",
        # The MD5 of the part of the XMP that didn't fit one JPEG segment,
        # added when a write makes the packet spill over. The tags in that
        # part are still compared one by one.
        "XMP-xmpNote:HasExtendedXMP",
    }
)
_OURS_GROUPS = frozenset({"XMP-asif"})

# compat = "wide": IPTC bookkeeping exiftool maintains, found by writing
# keywords into every fixture (tests/test_gates.py repeats that check).
_IPTC = frozenset(
    {
        "IPTC:Keywords",
        # Set to UTF-8 when As If creates the IPTC record.
        "IPTC:CodedCharacterSet",
        # exiftool adds these when it creates IPTC records 1 (envelope) and 2.
        "IPTC:EnvelopeRecordVersion",
        "IPTC:ApplicationRecordVersion",
        # exiftool's MD5 of the current IPTC block; changes with the keywords.
        "File:CurrentIPTCDigest",
    }
)
# compat = "wide": Photoshop:IPTCDigest records the IPTC the XMP was last in
# step with. Apps following the MWG guidelines show the XMP while it matches
# the IPTC (or there is no IPTC), and the IPTC once it doesn't, so the writer
# keeps it in step with the keywords it adds. It may change only from in
# step (or no IPTC) to in step, and it must not go stale under new IPTC:
# those apps would switch from the owner's XMP to the IPTC.
_IPTC_DIGEST = "Photoshop:IPTCDigest"
_CURRENT_IPTC_DIGEST = "File:CurrentIPTCDigest"

# Format bookkeeping that legitimately changes, by exact key (copies included).
_FORMAT_BOOKKEEPING: dict[str, frozenset[str]] = {
    "webp": frozenset(
        {
            # XMP needs the extended format: a simple WebP gains a VP8X header,
            # which exiftool reports as a different file type and new flags.
            # The structure gate checks every flag except XMP.
            "File:FileType",
            "RIFF:WebP_Flags",
            # With a VP8X header exiftool reports the size twice: the canvas
            # first, then the bitstream's. The first copy is still compared,
            # and the structure gate checks the canvas against the bitstream.
            "RIFF:ImageWidth#2",
            "RIFF:ImageHeight#2",
        }
    ),
}
# TIFF: exiftool rewrites the whole file and moves the data of every IFD
# (every page, reduced-size preview and thumbnail). Where it sits may
# change, never what: exiftool's ImageDataHash covers the strips and tiles
# of every TIFF IFD outside maker notes (a byte flipped in page 2 changes
# it) and the JPEG images it calls OtherImage and JpgFromRaw, the byte
# counts are still compared, and so is the thumbnail itself
# (IFDn:ThumbnailImage, as binary). Not for JPEG: there the hash doesn't
# cover EXIF's IFD1, and nothing in a JPEG's EXIF may move anyway.
_TIFF_MOVED = re.compile(
    r"IFD\d+:(?:StripOffsets|TileOffsets|ThumbnailOffset|OtherImageStart|JpgFromRawStart)"
    r"|SubIFD\d*:(?:StripOffsets|TileOffsets|OtherImageStart|JpgFromRawStart)"
)
# Every name exiftool gives where a TIFF's strips, tiles or JPEG images
# start (tags 0x111, 0x144 and 0x201, named by IFD, compression and file
# type). It moves all of them; one ``_TIFF_MOVED`` doesn't cover (an IFD0
# "PreviewImageStart", say, whose image the hash doesn't cover) would fail
# the protected gate after the model's work, so ``snapshot`` refuses it.
_TIFF_OFFSET = re.compile(
    r"[\w-]+:(?:StripOffsets|TileOffsets|ThumbnailOffset|PreviewImageStart"
    r"|OtherImageStart|JpgFromRawStart)"
)
# MPF (JPEG): exiftool reports where each extra image starts as an absolute
# file offset, so it moves when the XMP in front of it grows. The stored,
# relative offset must not change: the structure gate compares the MPF
# segment byte for byte, and the trailer gate the images themselves.
_MPF_IMAGE_START = re.compile(r"MPImage\d+:MPImageStart")
# XMP dcterms:provenance, as exiftool names it with the usual prefix (any
# language of it): C2PA's link to Content Credentials kept outside the file.
_C2PA_LINK = re.compile(r"XMP-dcterms:Provenance(?:-[\w-]+)?")

# Allowed doesn't mean anything goes: the fields As If writes are
# additive-only (§7), and the protected gate checks that itself rather than
# trusting the writer.
# - Keywords: every value there before is still there after, as often.
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "xmp": ("XMP-dc:Subject",),
    "wide": ("XMP-dc:Subject", "IPTC:Keywords"),  # with "xmp", IPTC can't change at all
}
# - Set once, then never changed: As If may add these, but a value already
#   there stays exactly as it was. Relabelling Latin-1 IPTC as UTF-8, say,
#   would garble the owner's keywords.
_WRITE_ONCE = frozenset({"XMP-et:OriginalImageHash", "XMP-et:OriginalImageHashType"})
_WRITE_ONCE_IPTC = frozenset(
    {"IPTC:CodedCharacterSet", "IPTC:EnvelopeRecordVersion", "IPTC:ApplicationRecordVersion"}
)
# - Captions: one may change only if it was As If's own, i.e. it hashes to
#   the XMP-asif:CaptionDigest recorded *before* the write.
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _is_caption(base: str) -> bool:
    # x-default is plain "Description", other languages "Description-pt-BR".
    return base == "XMP-dc:Description" or base.startswith("XMP-dc:Description-")


def _allowed(key: str, fmt: str, compat: Literal["xmp", "wide"]) -> bool:
    base = key.partition("#")[0]
    group = base.partition(":")[0]
    if base in _OURS or group in _OURS_GROUPS or _is_caption(base):
        return True
    if compat == "wide" and base in _IPTC:
        return True
    if key in _FORMAT_BOOKKEEPING.get(fmt, ()):
        return True
    if fmt == "tiff" and _TIFF_MOVED.fullmatch(base):
        return True
    return fmt == "jpeg" and _MPF_IMAGE_START.fullmatch(key) is not None


def _ours_or_blank(text: object, digest: object) -> bool:
    """A caption As If may replace: As If's own (it hashes to ``digest``), or blank.

    A blank entry holds nothing to protect, which is the writer's rule too.
    A non-string (exiftool's JSON turns "2019" into 2019) is never As If's.
    """
    if not isinstance(text, str):
        return False
    if not text.strip():
        return True
    return (
        isinstance(digest, str)
        and _HEX64.fullmatch(digest) is not None
        and hashlib.sha256(text.encode("utf-8")).hexdigest() == digest
    )


def _owned(tags: dict[str, Any], compat: Literal["xmp", "wide"]) -> dict[str, Any]:
    """Values in fields As If writes that are not As If's to change: the
    owner's captions, and write-once fields already set."""
    digest = tags.get("XMP-asif:CaptionDigest")
    once = (_WRITE_ONCE | _WRITE_ONCE_IPTC) if compat == "wide" else _WRITE_ONCE
    return {
        key: value
        for key, value in tags.items()
        if key.partition("#")[0] in once
        or (_is_caption(key.partition("#")[0]) and not _ours_or_blank(value, digest))
    }


def _xmp_shown(tags: dict[str, Any]) -> bool:
    """Do apps following the MWG guidelines show the XMP rather than the IPTC?
    They do while there is no IPTC, no IPTCDigest, or one that matches the IPTC."""
    current, digest = tags.get(_CURRENT_IPTC_DIGEST), tags.get(_IPTC_DIGEST)
    return current is None or digest is None or digest == current


def _digest_kept_in_step(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Was Photoshop:IPTCDigest in step with the IPTC (or there was none), and is it now?"""
    now = after.get(_CURRENT_IPTC_DIGEST)
    return _xmp_shown(before) and now is not None and after.get(_IPTC_DIGEST) == now


def _list_values(tags: dict[str, Any], name: str) -> Counter[str]:
    """Every value of a list tag, all copies together, with how often it occurs."""
    out: Counter[str] = Counter()
    for key, value in tags.items():
        if key.partition("#")[0] == name:
            # One item comes back as a scalar. JSON text keeps 1, "1" and true apart.
            for item in value if isinstance(value, list) else [value]:
                out[json.dumps(item)] += 1
    return out


# -- records ------------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    size: int
    mtime_ns: int
    sha256: str
    """Of the whole file."""


@dataclass(frozen=True)
class Snapshot:
    """The RAM copy as it was *before* writing."""

    fmt: str
    """``detect_format`` of the bytes, or ``"other"``."""
    image_hash: str
    """SHA-256 ``ImageDataHash``; "" for a file exiftool can't hash, which is
    then ``unprovable``."""
    size: int
    sha256: str
    # Kept out of repr: tag values can be GPS coordinates, and data is the photo.
    tags: dict[str, Any] = field(repr=False)
    """The full tag dump, keyed ``Group:Tag`` (``Group:Tag#2`` for a second copy)."""
    data: bytes | None = field(repr=False)
    """The whole file for JPEG, PNG and WebP; None otherwise (TIFFs can be huge)."""
    xmp: Counter[str] = field(repr=False)
    """The XMP's own values, read without exiftool (``structure.read_xmp``)."""
    unprovable: str | None
    """Why no write to this file can be proven safe, in plain words, or None.

    Set for a file the gates can't check (an unknown format, a structure
    that can't be parsed, a TIFF exiftool can't hash or whose preview it
    would move) and for one exiftool can't write without losing or
    rewriting something: a TIFF with a maker note or with bytes nothing
    points to, XMP with markup inside a value, metadata exiftool warns it
    can't read in full, Content Credentials, Apple's iDOT PNG chunk. It is
    a limit of the format or of exiftool, not a sign of a bug, so the
    caller should skip the file before writing. Only a file whose structure
    can't be parsed gets a reason starting "its structure can't be
    checked". The structure gate fails for it regardless."""


@dataclass(frozen=True)
class GateResult:
    name: str
    ok: bool
    detail: str = ""
    """Tag names, segment names or byte counts. Never values."""


# -- helpers ------------------------------------------------------------------


def fingerprint(path: Path) -> Fingerprint:
    """Size, modification time and SHA-256 of a file, all through one handle."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        st = os.fstat(fh.fileno())
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return Fingerprint(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=digest.hexdigest())


def _dump_tags(et: ExifTool, path: Path) -> tuple[dict[str, Any], list[str]]:
    """Every tag exiftool can read, minus file-system and derived ones, and
    exiftool's warnings about reading the file.

    Keys are ``Group:Tag``. A second copy of a tag in the same group becomes
    ``Group:Tag#2`` (and so on, in file order). exiftool's own copy numbers
    count across *all* groups, so adding ``XMP-dc:Description`` would turn an
    untouched ``PNG:Description`` into ``PNG:Copy1:Description``; numbering
    within the group keeps the keys stable.
    """
    out = et.execute(*_DUMP_ARGS, str(path)).stdout
    try:
        # Pairs, not dicts: a JSON object with a repeated key must not lose one.
        rows = json.loads(out, object_pairs_hook=lambda pairs: pairs)
    except ValueError as exc:
        raise ExifToolError("exiftool's tag dump isn't valid JSON") from exc
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise ExifToolError("unexpected tag dump from exiftool")
    tags: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in rows[0]:
        if key == "ExifTool:Error":
            # exiftool reports unreadable files here (stderr stays quiet with -json).
            raise ExifToolError("exiftool can't read this file", str(value))
        parts = key.split(":")
        if len(parts) >= 2 and parts[0] == "ExifTool" and parts[-1] == "Warning":
            warnings.append(str(value))
        if len(parts) < 2 or parts[0] in _SKIP_GROUPS:
            continue  # SourceFile, System:*, ExifTool:*, Composite:*
        base = f"{parts[0]}:{parts[-1]}"
        name, n = base, 1
        while name in tags:
            n += 1
            name = f"{base}#{n}"
        tags[name] = value
    return tags, warnings


@contextlib.contextmanager
def _mapped(path: Path) -> Iterator[mmap.mmap]:
    """The file, memory-mapped read-only: a large TIFF isn't copied into memory."""
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as view:
        yield view


def snapshot(et: ExifTool, path: Path) -> Snapshot:
    """Record ``path`` before writing. Raises ``ExifToolError`` if exiftool can't read it."""
    tags, warnings = _dump_tags(et, path)
    with path.open("rb") as fh:
        fmt = detect_format(fh.read(16)) or "other"
    unhashable: str | None = None
    try:
        image_hash = et.image_data_hash(path)
    except ExifToolError:
        # A valid file exiftool can't hash: a TIFF whose single image is
        # JPEG-compressed strips, which it takes for a preview. Anything
        # else it can't hash may well be corrupt.
        if not (
            fmt == "tiff"
            and tags.get("IFD0:Compression") in (6, 7)
            and "IFD0:PreviewImageStart" in tags
            and "IFD0:StripOffsets" not in tags
        ):
            raise
        image_hash = ""
        unhashable = "it's a JPEG-compressed TIFF, whose image data exiftool can't hash"
    data = path.read_bytes() if fmt in _KEEP_BYTES else None
    xmp, unprovable = _inspect(path, fmt, data)
    unprovable = unprovable or unhashable or _exiftool_limit(fmt, tags, warnings)
    if data is None:
        fp = fingerprint(path)
        return Snapshot(fmt, image_hash, fp.size, fp.sha256, tags, None, xmp, unprovable)
    digest = hashlib.sha256(data).hexdigest()
    return Snapshot(fmt, image_hash, len(data), digest, tags, data, xmp, unprovable)


def _exiftool_limit(fmt: str, tags: dict[str, Any], warnings: list[str]) -> str | None:
    """Why exiftool can't write the file unharmed, as its tag dump shows; None if it can."""
    if any(warning.startswith("[Minor]") for warning in warnings):
        # exiftool's mark for a part it skipped while reading (items past
        # the 1000th of an XMP list, a huge array): the protected gate
        # compares its reading, so it can't see that part.
        return (
            "exiftool reads only part of its metadata, so As If can't prove a write keeps the rest"
        )
    if any(k.partition("#")[0] == "JUMBF:JUMDLabel" and v == "c2pa" for k, v in tags.items()):
        return structure.C2PA_PROBLEM  # read by exiftool, in case the bytes check misses it
    if any(_C2PA_LINK.fullmatch(k.partition("#")[0]) for k in tags):
        return structure.C2PA_LINK_PROBLEM  # the same, for the XMP check
    if fmt == "tiff" and any(
        _TIFF_OFFSET.fullmatch(base) and not _TIFF_MOVED.fullmatch(base)
        for base in (key.partition("#")[0] for key in tags)
    ):
        return (
            "it's a TIFF with an embedded preview that exiftool would move, which As If can't check"
        )
    return None


def _inspect(path: Path, fmt: str, data: bytes | None) -> tuple[Counter[str], str | None]:
    """The XMP's own values, and why no write can be proven safe (None if one can)."""
    if fmt not in _KEEP_BYTES and fmt != "tiff":
        return Counter(), "As If can't check this file type byte by byte"
    try:
        if data is not None:
            text = structure.read_xmp(data)
            return text.values, text.problem or structure.write_hazard(data)
        with _mapped(path) as view:
            layout = structure.tiff_layout(view)
            text = structure.read_xmp(view)
    except StructureError as exc:
        return Counter(), f"its structure can't be checked: {exc}"
    if layout.maker_notes:
        # exiftool moves the note, rewrites the offsets inside it and drops
        # any bytes nothing in it points to: "never touched" can't hold.
        return text.values, "it's a TIFF with maker notes, which exiftool would rewrite"
    if layout.unreferenced:
        unreferenced = f"the TIFF holds {layout.unreferenced} bytes nothing points to"
        return text.values, f"{unreferenced}, which exiftool would drop"
    return text.values, text.problem


def _names(label: str, keys: Iterable[str], limit: int = 8) -> str:
    keys = sorted(keys)
    shown = ", ".join(keys[:limit])
    more = f" (+{len(keys) - limit} more)" if len(keys) > limit else ""
    return f"{label}: {shown}{more}"


# -- the gates ----------------------------------------------------------------


def _equality(et: ExifTool, before: Snapshot, after: Path) -> GateResult:
    try:
        image_hash = et.image_data_hash(after)
    except ExifToolError:
        return GateResult("equality", False, "exiftool can't compute the image-data hash")
    if image_hash != before.image_hash:
        return GateResult("equality", False, "the image data changed")
    return GateResult("equality", True)


def _protected(
    et: ExifTool, before: Snapshot, after: Path, compat: Literal["xmp", "wide"]
) -> GateResult:
    try:
        tags, _warnings = _dump_tags(et, after)
    except ExifToolError:
        return GateResult("protected", False, "exiftool can't read the written file")
    # The owner's values in fields As If writes are compared like any other tag.
    owned = _owned(before.tags, compat)
    old = {k: v for k, v in before.tags.items() if not _allowed(k, before.fmt, compat)} | owned
    new = {k: v for k, v in tags.items() if k in owned or not _allowed(k, before.fmt, compat)}
    changed = [k for k in old.keys() & new.keys() if old[k] != new[k]]
    out_of_step: list[str] = []
    if compat == "wide":
        if _IPTC_DIGEST in changed and _digest_kept_in_step(before.tags, tags):
            changed.remove(_IPTC_DIGEST)
        # An unchanged digest goes stale too when the IPTC changes under it.
        if _xmp_shown(before.tags) and not _xmp_shown(tags):
            out_of_step.append(_IPTC_DIGEST)
    added, removed = new.keys() - old.keys(), old.keys() - new.keys()
    lost = [
        name
        for name in _KEYWORDS[compat]
        if _list_values(before.tags, name) - _list_values(tags, name)
    ]
    parts = [
        _names(label, keys)
        for label, keys in (
            ("changed", changed),
            ("added", added),
            ("removed", removed),
            ("lost values", lost),
            ("out of step with the IPTC", out_of_step),
        )
        if keys
    ]
    return GateResult("protected", not parts, "; ".join(parts))


def _trailer(before: Snapshot, after_data: bytes | None) -> GateResult:
    if before.fmt != "jpeg":
        return GateResult("trailer", True, "n/a")
    if before.data is None or after_data is None:
        return GateResult("trailer", False, "can't prove the trailer is intact: file not read")
    try:
        old, new = jpeg_trailer(before.data), jpeg_trailer(after_data)
    except StructureError:
        return GateResult(
            "trailer", False, "can't prove the trailer is intact: JPEG can't be parsed"
        )
    if old != new:
        return GateResult(
            "trailer", False, f"the trailer changed ({len(old)} bytes before, {len(new)} after)"
        )
    return GateResult("trailer", True, f"{len(old)} bytes, identical" if old else "no trailer")


def _structure(
    before: Snapshot, after: Path, after_data: bytes | None, compat: Literal["xmp", "wide"]
) -> GateResult:
    cant = "can't prove the structure is intact"
    if before.unprovable is not None:
        return GateResult("structure", False, f"can't prove a write is safe: {before.unprovable}")
    diffs: list[str] = []
    text: structure.XmpText | None = None
    if before.fmt == "tiff":
        # exiftool rebuilds a TIFF, so its bytes can't match; the snapshot
        # proved every byte is something exiftool carries over. That leaves
        # the XMP's own text.
        try:
            with _mapped(after) as view:
                if detect_format(view[:16]) != "tiff":
                    return GateResult("structure", False, "the written file isn't a TIFF any more")
                text = structure.read_xmp(view)
        except StructureError:  # before ValueError: it is one
            text = None
        except (OSError, ValueError):  # ValueError: an empty file can't be mapped
            return GateResult("structure", False, f"{cant}: file not read")
    else:
        if before.data is None or after_data is None:
            return GateResult("structure", False, f"{cant}: file not read")
        try:
            found = structure.compare(before.data, after_data, allow_iptc=compat == "wide")
        except StructureError:
            return GateResult("structure", False, f"{cant}: can't parse it")
        if found is None:
            # The snapshot says JPEG/PNG/WebP but the bytes aren't: never assume.
            return GateResult("structure", False, f"{cant}: unknown format")
        diffs = found
        with contextlib.suppress(StructureError):
            text = structure.read_xmp(after_data)
    if text is None:
        if not diffs:  # a file that changed format can't be read as before: said already
            diffs.append("the written XMP can't be read")
    elif lost := _xmp_lost(before, text.values):
        diffs.append(f"{lost} of the original XMP's values went missing")
    if diffs:
        more = f" (+{len(diffs) - 3} more)" if len(diffs) > 3 else ""
        return GateResult("structure", False, "; ".join(diffs[:3]) + more)
    return GateResult("structure", True)


def _xmp_lost(before: Snapshot, after: Counter[str]) -> int:
    """How many of the original XMP's values are gone, As If's own old caption aside.

    exiftool reads the XMP into tags and writes it back from them, so a
    value its parser misses (text beside markup, say) vanishes without a
    warning, and the protected gate, which uses the same parser, can't see
    it. This compares the packets' own text.
    """
    digest = before.tags.get("XMP-asif:CaptionDigest")
    missing = before.xmp - after
    return sum(n for value, n in missing.items() if not _ours_or_blank(value, digest))


def _size(before: Snapshot, after: Path) -> GateResult:
    try:
        growth = after.stat().st_size - before.size
    except OSError:
        return GateResult("size", False, "can't read the written file")
    if growth >= MAX_GROWTH:
        return GateResult("size", False, f"grew by {growth} bytes (limit {MAX_GROWTH - 1})")
    return GateResult("size", True, f"grew by {growth} bytes")


def check_written(
    et: ExifTool, before: Snapshot, after: Path, *, compat: Literal["xmp", "wide"]
) -> list[GateResult]:
    """Run every gate on the written file ``after``; all of them, even after a failure.

    Never raises for a bad or unreadable file: a gate that can't prove its
    point fails.
    """
    after_data: bytes | None = None
    if before.fmt in _KEEP_BYTES:
        # Unreadable: after_data stays None, and the trailer and structure gates fail.
        with contextlib.suppress(OSError):
            after_data = after.read_bytes()
    return [
        _equality(et, before, after),
        _protected(et, before, after, compat),
        _trailer(before, after_data),
        _structure(before, after, after_data, compat),
        _size(before, after),
    ]


def check_source_unchanged(then: Fingerprint, now: Fingerprint) -> GateResult:
    """Gate 4 (``remote-unchanged``): size, mtime and SHA-256 all equal.

    For the local adapter the "remote" is the source file.
    """
    changed = [
        what
        for what, same in (
            ("size", then.size == now.size),
            ("modification time", then.mtime_ns == now.mtime_ns),
            ("content", then.sha256 == now.sha256),
        )
        if not same
    ]
    if changed:
        return GateResult(
            "remote-unchanged", False, "changed since it was read: " + ", ".join(changed)
        )
    return GateResult("remote-unchanged", True)


def all_ok(results: Iterable[GateResult]) -> bool:
    """True when every gate passed. No gates at all proves nothing, so that's False."""
    results = list(results)
    return bool(results) and all(r.ok for r in results)
