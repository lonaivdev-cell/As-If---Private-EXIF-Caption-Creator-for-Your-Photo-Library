"""The integrity gates (§7 of docs/DESIGN.md). Every one must pass before a
processed copy is allowed to replace anything.

Before writing, ``snapshot`` records the RAM copy: its image-data hash, a
full tag dump and (for JPEG, PNG and WebP) its bytes. After writing,
``check_written`` runs, in order:

1. ``equality``: exiftool's SHA-256 ``ImageDataHash`` is unchanged.
2. ``protected``: every tag outside the ones As If writes is unchanged.
3. ``trailer``: a JPEG's bytes after the primary image are unchanged
   (motion-photo video, MPF images), and must be provably so.
4. ``structure``: ``asif.meta.structure.compare``, byte for byte.
5. ``size``: the file grew by less than 64 KiB.

``check_source_unchanged`` is the design's gate 4 (``remote-unchanged``).

Gate details name tags, segments and byte counts, never values: a value can
be a caption, a keyword or GPS coordinates.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from collections.abc import Iterable
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
# gates. TIFFs can be huge and have no structure check, so they aren't.
_KEEP_BYTES = frozenset({"jpeg", "png", "webp"})

# The tag dump. Deliberately stricter than a plain ``-json -G1 -a -b``:
#   -n        raw values. Print conversion rounds: a GPS latitude moved by
#             1e-7 degrees prints exactly the same.
#   -G1:4     with plain -G1, exiftool's JSON silently drops the second of
#             two same-named tags in one group (two COM segments, say).
#             Family 4 tells the copies apart; ``_dump_tags`` renumbers them.
#   -b        binary values come back base64-encoded, compared byte for byte.
#   -ICC_Profile -MakerNotes
#             the raw blocks; -all:all only lists the tags decoded from them.
_DUMP_ARGS = ("-json", "-n", "-G1:4", "-a", "-b", "-all:all", "-ICC_Profile", "-MakerNotes")
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
    "tiff": frozenset(
        {
            # exiftool rewrites a TIFF and moves its image data. The equality
            # gate hashes the data these point at; the byte counts are compared.
            "IFD0:StripOffsets",
            "IFD0:TileOffsets",
        }
    ),
}
# MPF (JPEG): exiftool reports where each extra image starts as an absolute
# file offset, so it moves when the XMP in front of it grows. The stored,
# relative offset must not change: the structure gate compares the MPF
# segment byte for byte, and the trailer gate the images themselves.
_MPF_IMAGE_START = re.compile(r"MPImage\d+:MPImageStart")


def _allowed(key: str, fmt: str, compat: Literal["xmp", "wide"]) -> bool:
    base = key.partition("#")[0]
    group = base.partition(":")[0]
    if base in _OURS or group in _OURS_GROUPS:
        return True
    # Captions: x-default is plain "Description", other languages "Description-pt-BR".
    if base == "XMP-dc:Description" or base.startswith("XMP-dc:Description-"):
        return True
    if compat == "wide" and base in _IPTC:
        return True
    if key in _FORMAT_BOOKKEEPING.get(fmt, ()):
        return True
    return fmt == "jpeg" and _MPF_IMAGE_START.fullmatch(key) is not None


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
    """SHA-256 ``ImageDataHash``."""
    size: int
    sha256: str
    # Kept out of repr: tag values can be GPS coordinates, and data is the photo.
    tags: dict[str, Any] = field(repr=False)
    """The full tag dump, keyed ``Group:Tag`` (``Group:Tag#2`` for a second copy)."""
    data: bytes | None = field(repr=False)
    """The whole file for JPEG, PNG and WebP; None otherwise (TIFFs can be huge)."""


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


def _dump_tags(et: ExifTool, path: Path) -> dict[str, Any]:
    """Every tag exiftool can read, minus file-system and derived ones.

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
    for key, value in rows[0]:
        if key == "ExifTool:Error":
            # exiftool reports unreadable files here (stderr stays quiet with -json).
            raise ExifToolError("exiftool can't read this file", str(value))
        parts = key.split(":")
        if len(parts) < 2 or parts[0] in _SKIP_GROUPS:
            continue  # SourceFile, System:*, ExifTool:*, Composite:*
        base = f"{parts[0]}:{parts[-1]}"
        name, n = base, 1
        while name in tags:
            n += 1
            name = f"{base}#{n}"
        tags[name] = value
    return tags


def snapshot(et: ExifTool, path: Path) -> Snapshot:
    """Record ``path`` before writing. Raises ``ExifToolError`` if exiftool can't read it."""
    image_hash = et.image_data_hash(path)
    tags = _dump_tags(et, path)
    with path.open("rb") as fh:
        fmt = detect_format(fh.read(16)) or "other"
    if fmt not in _KEEP_BYTES:
        fp = fingerprint(path)
        return Snapshot(fmt, image_hash, fp.size, fp.sha256, tags, None)
    data = path.read_bytes()
    return Snapshot(fmt, image_hash, len(data), hashlib.sha256(data).hexdigest(), tags, data)


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
        tags = _dump_tags(et, after)
    except ExifToolError:
        return GateResult("protected", False, "exiftool can't read the written file")
    old = {k: v for k, v in before.tags.items() if not _allowed(k, before.fmt, compat)}
    new = {k: v for k, v in tags.items() if not _allowed(k, before.fmt, compat)}
    changed = [k for k in old.keys() & new.keys() if old[k] != new[k]]
    added, removed = new.keys() - old.keys(), old.keys() - new.keys()
    parts = [
        _names(label, keys)
        for label, keys in (("changed", changed), ("added", added), ("removed", removed))
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
    before: Snapshot, after_data: bytes | None, compat: Literal["xmp", "wide"]
) -> GateResult:
    if before.fmt not in _KEEP_BYTES:
        return GateResult("structure", True, "n/a")
    if before.data is None or after_data is None:
        return GateResult("structure", False, "can't prove the structure is intact: file not read")
    try:
        diffs = structure.compare(before.data, after_data, allow_iptc=compat == "wide")
    except StructureError:
        return GateResult("structure", False, "can't prove the structure is intact: can't parse it")
    if diffs is None:
        # The snapshot says JPEG/PNG/WebP but the bytes aren't: never assume.
        return GateResult("structure", False, "can't prove the structure is intact: unknown format")
    if diffs:
        more = f" (+{len(diffs) - 3} more)" if len(diffs) > 3 else ""
        return GateResult("structure", False, "; ".join(diffs[:3]) + more)
    return GateResult("structure", True)


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
        _structure(before, after_data, compat),
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
