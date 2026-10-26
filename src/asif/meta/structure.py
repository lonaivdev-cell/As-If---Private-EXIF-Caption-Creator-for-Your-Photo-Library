"""Byte-level structure checks for JPEG, PNG and WebP (and TIFF's layout).
Pure Python, no exiftool.

These back up exiftool's ``ImageDataHash`` with something stronger: apart
from the XMP packet (and IPTC, when As If writes it), every byte of the file
must be identical before and after a write, in the same order. They also
find the JPEG trailer, where Samsung and Google "motion photos" and MPF
multi-picture files (Ultra HDR gain maps, depth maps) keep their extra data.

exiftool rebuilds a TIFF on every write, so its bytes can't match. Instead,
``tiff_layout`` proves exiftool *can* carry everything over: every byte is
reachable from an IFD it rewrites. ``read_xmp`` reads the XMP packet's own
text with an XML parser, so a value exiftool's parser misses can't vanish
unnoticed. ``write_hazard`` finds what a write would break even though
those bytes stay the same: Content Credentials, Apple's iDOT chunk.

Every function takes whole files as bytes and is pure. Messages name
segments, chunks and byte counts, never file contents: these files are
somebody's photos.
"""

from __future__ import annotations

import hashlib
import heapq
import re
import struct
import zlib
from collections import Counter
from collections.abc import Buffer, Iterable
from dataclasses import dataclass, field
from typing import Literal
from xml.parsers import expat

Format = Literal["jpeg", "png", "webp", "tiff"]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class StructureError(ValueError):
    """The bytes can't be parsed (truncated, corrupt, or not this format)."""


def detect_format(data: bytes) -> Format | None:
    """The file format by magic bytes only (file names can lie)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == _PNG_SIGNATURE:
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    # Classic TIFF ("*") and BigTIFF ("+"), either byte order.
    if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    return None


def is_bigtiff(data: bytes) -> bool:
    """BigTIFF (either byte order), which exiftool reads but can't write."""
    return data[:4] in (b"II+\x00", b"MM\x00+")


_FORMAT_NAMES = {"jpeg": "JPEG", "png": "PNG", "webp": "WebP", "tiff": "TIFF", None: "unknown"}


def _count_diff(kind: str, before: int, after: int) -> str:
    return f"{kind} differs ({before} bytes before, {after} after)"


def _diff_sequences(
    kind: str, before: list[tuple[str, bytes]], after: list[tuple[str, bytes]]
) -> list[str]:
    """Compare two ordered lists of (name, raw bytes) items, describing differences."""
    if len(before) != len(after):
        msg = f"{kind} count differs ({len(before)} before, {len(after)} after)"
        for i, (b, a) in enumerate(zip(before, after, strict=False)):
            if b != a:
                return [f"{msg}; first difference at {kind} {i + 1} ({b[0]} before, {a[0]} after)"]
        return [msg]
    diffs = []
    for i, ((b_name, b_raw), (a_name, a_raw)) in enumerate(zip(before, after, strict=True)):
        if b_name != a_name:
            diffs.append(f"{kind} {i + 1} is {b_name} before, {a_name} after")
        elif b_raw != a_raw:
            diffs.append(_count_diff(f"{kind} {i + 1} ({b_name})", len(b_raw), len(a_raw)))
    return diffs


# -- JPEG ----------------------------------------------------------------------

_XMP_STD = b"http://ns.adobe.com/xap/1.0/\x00"
_XMP_EXT = b"http://ns.adobe.com/xmp/extension/\x00"
_PHOTOSHOP = b"Photoshop 3.0\x00"
_APP1, _APP13, _SOS, _EOI = 0xE1, 0xED, 0xDA, 0xD9
# Markers that stand alone, with no length field: TEM and RST0-RST7.
_STANDALONE = frozenset({0x01, *range(0xD0, 0xD8)})
_MARKER_NAMES = {
    0xC4: "DHT",
    0xC8: "JPG",
    0xCC: "DAC",
    0xD8: "SOI",
    0xD9: "EOI",
    0xDA: "SOS",
    0xDB: "DQT",
    0xDC: "DNL",
    0xDD: "DRI",
    0xDE: "DHP",
    0xDF: "EXP",
    0xFE: "COM",
    0x01: "TEM",
}


def _marker_name(marker: int) -> str:
    if 0xE0 <= marker <= 0xEF:
        return f"APP{marker - 0xE0}"
    if 0xD0 <= marker <= 0xD7:
        return f"RST{marker - 0xD0}"
    if 0xC0 <= marker <= 0xCF and marker not in _MARKER_NAMES:
        return f"SOF{marker - 0xC0}"
    return _MARKER_NAMES.get(marker, f"marker 0x{marker:02X}")


@dataclass(frozen=True)
class _JpegSegment:
    marker: int
    raw: bytes
    """Everything from the segment's first 0xFF (fill bytes included) to its end."""
    payload: bytes
    """The bytes after the length field (empty for stand-alone markers)."""
    payload_at: int
    """Where ``payload`` starts in the file."""


@dataclass(frozen=True)
class _JpegLayout:
    header: list[_JpegSegment]
    """Segments between SOI and the first SOS, in file order."""
    scan_start: int
    """Offset of the first SOS marker (or of EOI if the file has no scan)."""
    eoi: int
    """Offset just past the EOI marker that ends the primary image."""


# Inside a scan, 0xFF is followed by 0x00 (a stuffed data byte), RST0-RST7 or
# more 0xFF (fill); anything else is a real marker. One regex search keeps the
# scan in C: about 25 ms for 30 MB of entropy data, and no slow worst case.
_NEXT_MARKER = re.compile(rb"\xff[\x01-\xcf\xd8-\xfe]")


def _skip_entropy_data(data: bytes, pos: int) -> int:
    """Offset of the first real marker at or after ``pos`` (past any fill bytes)."""
    match = _NEXT_MARKER.search(data, pos)
    if match is None:
        raise StructureError("JPEG image data runs to the end of the file (no EOI marker)")
    return match.start()


def _jpeg_layout(data: bytes) -> _JpegLayout:
    if data[:2] != b"\xff\xd8":
        raise StructureError("no JPEG start-of-image marker")
    end = len(data)
    header: list[_JpegSegment] = []
    scan_start: int | None = None
    pos = 2
    while True:
        start = pos
        if pos >= end:
            raise StructureError("JPEG has no end-of-image marker")
        if data[pos] != 0xFF:
            raise StructureError(f"JPEG has stray bytes where a marker should be (offset {pos})")
        while pos < end and data[pos] == 0xFF:
            pos += 1  # fill bytes may precede any marker
        if pos >= end:
            raise StructureError("JPEG has no end-of-image marker")
        marker = data[pos]
        pos += 1
        if marker == _EOI:
            if scan_start is None:
                scan_start = start
            return _JpegLayout(header, scan_start, pos)
        if marker == 0x00 or marker == 0xD8:
            raise StructureError(f"JPEG has an invalid marker at offset {start}")
        if marker in _STANDALONE:
            if scan_start is None:
                header.append(_JpegSegment(marker, data[start:pos], b"", pos))
            continue
        if pos + 2 > end:
            raise StructureError(f"JPEG {_marker_name(marker)} segment runs past the end")
        length = int.from_bytes(data[pos : pos + 2], "big")
        if length < 2 or pos + length > end:
            raise StructureError(f"JPEG {_marker_name(marker)} segment runs past the end")
        seg_end = pos + length
        if marker == _SOS and scan_start is None:
            scan_start = start
        if scan_start is None:
            header.append(
                _JpegSegment(marker, data[start:seg_end], data[pos + 2 : seg_end], pos + 2)
            )
        pos = seg_end
        if marker == _SOS:
            pos = _skip_entropy_data(data, pos)


def jpeg_eoi_offset(data: bytes) -> int:
    """Offset just past the EOI marker that ends the *primary* image.

    Walks the marker segments from SOI. Inside the entropy-coded data after
    each SOS it skips ``FF00`` stuffing, ``FFD0``-``FFD7`` restart markers
    and ``FF`` fill bytes, so progressive files (several SOS/DHT between
    scans) work too. Anything after this offset is the trailer.

    Raises ``StructureError`` when there is no SOI, a segment runs past the
    end, or no EOI is found.
    """
    return _jpeg_layout(data).eoi


def jpeg_trailer(data: bytes) -> bytes:
    """The bytes after the primary image's EOI (motion-photo video, MPF images...)."""
    return data[jpeg_eoi_offset(data) :]


def jpeg_segments(data: bytes, marker: int, prefix: bytes) -> list[tuple[int, bytes]]:
    """Every segment ahead of the image data with this ``marker`` and a payload
    starting with ``prefix``, in file order: where its payload (the bytes
    after the length field) starts in the file, and the payload.

    Raises ``StructureError`` when the JPEG can't be parsed.
    """
    return [
        (seg.payload_at, seg.payload)
        for seg in _jpeg_layout(data).header
        if seg.marker == marker and seg.payload.startswith(prefix)
    ]


def _ignorable_jpeg_segment(seg: _JpegSegment, allow_iptc: bool) -> bool:
    if seg.marker == _APP1 and seg.payload.startswith((_XMP_STD, _XMP_EXT)):
        return True
    return allow_iptc and seg.marker == _APP13 and seg.payload.startswith(_PHOTOSHOP)


def compare_jpeg(before: bytes, after: bytes, *, allow_iptc: bool) -> list[str]:
    """Differences outside the parts As If may change; ``[]`` means none.

    Ignored: APP1 segments holding standard or extended XMP and, when
    ``allow_iptc``, APP13 ``Photoshop 3.0`` segments (where IPTC lives).
    Every other segment from SOI to the first SOS must match byte-for-byte
    and in order; the bytes from the first SOS to EOI must be identical, and
    so must the trailers. Raises ``StructureError`` if either side can't be
    parsed.
    """

    def kept(layout: _JpegLayout) -> list[tuple[str, bytes]]:
        return [
            (_marker_name(seg.marker), seg.raw)
            for seg in layout.header
            if not _ignorable_jpeg_segment(seg, allow_iptc)
        ]

    lb, la = _jpeg_layout(before), _jpeg_layout(after)
    diffs = _diff_sequences("JPEG segment", kept(lb), kept(la))
    scan_b, scan_a = before[lb.scan_start : lb.eoi], after[la.scan_start : la.eoi]
    if scan_b != scan_a:
        diffs.append(
            _count_diff("JPEG image data (first scan to end of image)", len(scan_b), len(scan_a))
        )
    trailer_b, trailer_a = before[lb.eoi :], after[la.eoi :]
    if trailer_b != trailer_a:
        diffs.append(
            _count_diff("JPEG trailer (after end of image)", len(trailer_b), len(trailer_a))
        )
    return diffs


# -- PNG -----------------------------------------------------------------------

_PNG_TEXT_CHUNKS = frozenset({b"iTXt", b"tEXt", b"zTXt"})
_PNG_XMP_KEYWORD = b"XML:com.adobe.xmp"
# ImageMagick's hex-encoded copy of the XMP, which exiftool reads and rewrites too.
_PNG_XMP_RAW_PROFILE = b"Raw profile type xmp"
# exiftool's (non-standard) home for IPTC in PNG, used for compat = "wide".
_PNG_IPTC_KEYWORD = b"Raw profile type iptc"
# Apple's offsets into the image data, for decoding it in parallel.
_PNG_IDOT = b"iDOT"


def _chunk_name(fourcc: bytes) -> str:
    """A chunk type for messages; hex if it isn't the usual four letters."""
    if fourcc.isascii() and fourcc.decode("ascii").replace(" ", "").isalnum():
        return fourcc.decode("ascii").rstrip()
    return "0x" + fourcc.hex()


def _png_chunks(data: bytes) -> tuple[list[tuple[bytes, bytes, bytes]], bytes]:
    """``([(type, data, raw chunk), ...], bytes after IEND)``.

    Every chunk's CRC is checked: exiftool refuses to write a PNG with a bad
    one (a sign of corruption), and it always writes good ones itself.
    """
    if data[:8] != _PNG_SIGNATURE:
        raise StructureError("no PNG signature")
    chunks: list[tuple[bytes, bytes, bytes]] = []
    pos, end = 8, len(data)
    while True:
        if pos + 8 > end:
            raise StructureError("PNG has no IEND chunk")
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8]
        chunk_end = pos + 12 + length  # length, type, data, CRC
        if chunk_end > end:
            raise StructureError(f"PNG {_chunk_name(ctype)} chunk runs past the end")
        cdata = data[pos + 8 : pos + 8 + length]
        crc = int.from_bytes(data[chunk_end - 4 : chunk_end], "big")
        if zlib.crc32(cdata, zlib.crc32(ctype)) != crc:
            raise StructureError(f"PNG {_chunk_name(ctype)} chunk has a bad CRC")
        chunks.append((ctype, cdata, data[pos:chunk_end]))
        pos = chunk_end
        if ctype == b"IEND":
            return chunks, data[pos:]


def _idot_distances(chunks: list[tuple[bytes, bytes, bytes]]) -> list[list[int]]:
    """For each ``iDOT`` chunk, how far each ``IDAT`` chunk starts from it.

    Apple's iDOT chunk records where the image data continues, counted
    from the iDOT chunk's own start, so these must never change.
    """
    starts: list[int] = []
    pos = 8
    for _ctype, _cdata, raw in chunks:
        starts.append(pos)
        pos += len(raw)
    idats = [at for at, (ctype, _d, _r) in zip(starts, chunks, strict=True) if ctype == b"IDAT"]
    return [
        [idat - at for idat in idats]
        for at, (ctype, _d, _r) in zip(starts, chunks, strict=True)
        if ctype == _PNG_IDOT
    ]


def compare_png(before: bytes, after: bytes, *, allow_iptc: bool = False) -> list[str]:
    """Chunk-for-chunk (type, data and CRC) in order; ``[]`` means no differences.

    Ignores ``iTXt``/``tEXt``/``zTXt`` chunks whose keyword is
    ``XML:com.adobe.xmp`` and, when ``allow_iptc``, ``Raw profile type
    iptc`` (where exiftool puts IPTC in a PNG). Apple's ``iDOT`` chunk
    must still point at the same image data: no chunk may appear, vanish
    or change size between it and the image data. Bytes after ``IEND``
    must match too. Raises ``StructureError`` if either side can't be
    parsed.
    """
    ignored = {_PNG_XMP_KEYWORD, _PNG_IPTC_KEYWORD} if allow_iptc else {_PNG_XMP_KEYWORD}

    def kept(chunks: list[tuple[bytes, bytes, bytes]]) -> list[tuple[str, bytes]]:
        return [
            (_chunk_name(ctype), raw)
            for ctype, cdata, raw in chunks
            if not (ctype in _PNG_TEXT_CHUNKS and cdata.split(b"\0", 1)[0] in ignored)
        ]

    chunks_b, tail_b = _png_chunks(before)
    chunks_a, tail_a = _png_chunks(after)
    diffs = _diff_sequences("PNG chunk", kept(chunks_b), kept(chunks_a))
    if _idot_distances(chunks_b) != _idot_distances(chunks_a):
        diffs.append("PNG iDOT chunk no longer points at the same image data")
    if tail_b != tail_a:
        diffs.append(_count_diff("PNG data after IEND", len(tail_b), len(tail_a)))
    return diffs


# -- WebP ----------------------------------------------------------------------

_VP8X_XMP = 0x04
_VP8X_ALPHA = 0x10


def _webp_chunks(data: bytes) -> tuple[list[tuple[bytes, bytes, bytes]], bytes]:
    """``([(fourcc, data, raw chunk incl. padding), ...], bytes after the RIFF data)``."""
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise StructureError("not a RIFF/WebP file")
    riff_end = 8 + int.from_bytes(data[4:8], "little")
    if riff_end > len(data) or riff_end < 12:
        raise StructureError("WebP RIFF size doesn't match the file")
    chunks: list[tuple[bytes, bytes, bytes]] = []
    pos = 12
    while pos < riff_end:
        if pos + 8 > riff_end:
            raise StructureError("WebP chunk header runs past the end")
        fourcc = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        chunk_end = pos + 8 + size + (size & 1)  # odd sizes carry a pad byte
        if pos + 8 + size > riff_end:
            raise StructureError(f"WebP {_chunk_name(fourcc)} chunk runs past the end")
        chunk_end = min(chunk_end, riff_end)
        chunks.append((fourcc, data[pos + 8 : pos + 8 + size], data[pos:chunk_end]))
        pos = chunk_end
    return chunks, data[riff_end:]


def _webp_bitstream(chunks: list[tuple[bytes, bytes, bytes]]) -> tuple[int, int, bool]:
    """``(width, height, has_alpha)`` from the image bitstream itself."""
    for fourcc, cdata, _raw in chunks:
        if fourcc == b"VP8L":
            if len(cdata) < 5 or cdata[0] != 0x2F:
                raise StructureError("WebP VP8L header is truncated or invalid")
            bits = int.from_bytes(cdata[1:5], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, bool(bits >> 28 & 1)
        if fourcc == b"VP8 ":
            if len(cdata) < 10 or cdata[3:6] != b"\x9d\x01\x2a":
                raise StructureError("WebP VP8 header is truncated or invalid")
            width, height = struct.unpack("<HH", cdata[6:10])
            return width & 0x3FFF, height & 0x3FFF, False
    raise StructureError("WebP has no VP8/VP8L image data")


def _vp8x(chunks: list[tuple[bytes, bytes, bytes]]) -> bytes | None:
    for fourcc, cdata, _raw in chunks:
        if fourcc == b"VP8X":
            if len(cdata) < 10:
                raise StructureError("WebP VP8X header is truncated")
            return cdata
    return None


def _canvas(vp8x: bytes) -> tuple[int, int]:
    return int.from_bytes(vp8x[4:7], "little") + 1, int.from_bytes(vp8x[7:10], "little") + 1


def compare_webp(before: bytes, after: bytes) -> list[str]:
    """RIFF chunk-for-chunk in order; ``[]`` means no differences.

    ``XMP `` chunks are ignored and the ``VP8X`` header chunk is checked on
    its own: when both sides have one, everything in it except the XMP flag
    (0x04) must match, canvas size included. A file that had no ``VP8X``
    may gain one carrying only the XMP flag (plus the alpha flag when the
    image is VP8L with alpha, or an ``ALPH`` chunk exists) and a canvas the
    size of the image. Bytes after the RIFF data must match too. Raises
    ``StructureError`` if either side can't be parsed.
    """
    chunks_b, tail_b = _webp_chunks(before)
    chunks_a, tail_a = _webp_chunks(after)
    diffs: list[str] = []

    vp8x_b, vp8x_a = _vp8x(chunks_b), _vp8x(chunks_a)
    if vp8x_b is not None and vp8x_a is None:
        diffs.append("WebP VP8X header was removed")
    elif vp8x_b is not None and vp8x_a is not None:
        if _canvas(vp8x_b) != _canvas(vp8x_a):
            diffs.append("WebP VP8X canvas size differs")
        flags_b, flags_a = vp8x_b[0] | _VP8X_XMP, vp8x_a[0] | _VP8X_XMP
        # Reserved bytes (and anything past the canvas) count as flags here.
        if flags_b != flags_a or vp8x_b[1:4] + vp8x_b[10:] != vp8x_a[1:4] + vp8x_a[10:]:
            diffs.append("WebP VP8X flags differ (other than the XMP flag)")
    elif vp8x_a is not None:
        width, height, alpha = _webp_bitstream(chunks_a)
        allowed = _VP8X_XMP
        if alpha or any(fourcc == b"ALPH" for fourcc, _d, _r in chunks_a):
            allowed |= _VP8X_ALPHA
        if vp8x_a[0] & ~allowed or vp8x_a[1:4] != b"\0\0\0" or vp8x_a[10:]:
            diffs.append("WebP gained a VP8X header with flags other than XMP/alpha")
        if _canvas(vp8x_a) != (width, height):
            diffs.append("WebP gained a VP8X header whose canvas doesn't match the image")

    def kept(chunks: list[tuple[bytes, bytes, bytes]]) -> list[tuple[str, bytes]]:
        return [(_chunk_name(f), raw) for f, _d, raw in chunks if f not in (b"XMP ", b"VP8X")]

    diffs += _diff_sequences("WebP chunk", kept(chunks_b), kept(chunks_a))
    if tail_b != tail_a:
        diffs.append(_count_diff("WebP data after the RIFF data", len(tail_b), len(tail_a)))
    return diffs


# -- TIFF ----------------------------------------------------------------------

# Bytes per value of each field type (TIFF 6.0, plus 13 for IFD pointers).
_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
# Pointers to the IFDs exiftool follows and rebuilds: SubIFDs, Exif, GPS, Interop.
_TIFF_SUB_IFDS = frozenset({0x014A, 0x8769, 0x8825, 0xA005})
# (offsets, byte counts) of data outside the IFDs: strips, tiles, a JPEG thumbnail.
_TIFF_DATA = ((0x0111, 0x0117), (0x0144, 0x0145), (0x0201, 0x0202))
# MakerNote, and DNGPrivateData (which holds one). exiftool rebuilds both
# when it moves them, rewriting the offsets inside and dropping any bytes
# nothing in them points to.
_TIFF_MAKER_NOTES = frozenset({0x927C, 0xC634})
_TIFF_XMP = 0x02BC
# TIFF aligns values to even offsets and some writers to 4 bytes: a gap of
# up to 3 bytes between two referenced parts is padding, whatever it holds.
_TIFF_PADDING = 3
# A longer gap holds nothing worth keeping if it is all zeros, apart from at
# most one stale 8-byte TIFF header: Pillow's multi-page writer leaves each
# appended page's own header (byte order, 42, an offset) in the file. Gaps
# longer than this aren't padding, whatever they hold.
_TIFF_JUNK_MAX = 64 * 1024
# Far more strips or tiles than any real image has: past this, checking
# would cost more memory than the file is worth, so refuse instead.
_TIFF_MAX_PARTS = 1_000_000


@dataclass(frozen=True)
class TiffLayout:
    unreferenced: int
    """Bytes nothing points to from an IFD exiftool rewrites, padding and
    zero filler aside. exiftool rebuilds the file from its IFDs, so it would
    drop them: a vendor block behind a private offset tag, say."""
    maker_notes: bool
    """An IFD holds a maker note, which exiftool rebuilds when it moves it."""
    xmp: tuple[bytes, ...] = field(repr=False)
    """The XMP packet (tag 700) of every IFD that has one."""
    shared: int = 0
    """Bytes the strips, tiles and JPEG blocks (tags 0x201/0x202) claim more
    than once: old-style JPEG TIFFs point a page's JPEG block at its strips,
    say. exiftool writes each claim out on its own, so these bytes would be
    written again for every extra claim."""


def tiff_layout(data: Buffer) -> TiffLayout:
    """Walk every IFD exiftool rewrites and account for each byte of a TIFF.

    Covered: the header, the IFD chain from IFD0 and every SubIFD, Exif,
    GPS and Interop IFD, every value stored outside its IFD entry, and the
    strips, tiles and JPEG thumbnail they point to. Every other tag's value
    is just data, never an offset: exiftool copies the number, not what it
    points to. Takes ``bytes`` or an ``mmap``, so a large TIFF isn't copied.
    ``shared`` counts the bytes those strips, tiles and JPEG blocks share.
    Raises ``StructureError`` for BigTIFF, an IFD loop, an unknown field
    type, or anything that points past the end of the file.
    """
    with memoryview(data) as view:
        return _tiff_walk(view)


def _tiff_walk(view: memoryview) -> TiffLayout:
    size = len(view)
    magic = bytes(view[:4])
    if magic in (b"II+\x00", b"MM\x00+"):
        raise StructureError("BigTIFF, which exiftool can't write")
    if magic not in (b"II*\x00", b"MM\x00*") or size < 8:
        raise StructureError("not a TIFF file")
    order = "<" if magic[:2] == b"II" else ">"
    spans: list[tuple[int, int]] = [(0, 8)]
    data_spans: list[tuple[int, int]] = []  # strips, tiles and JPEG blocks
    xmp: list[bytes] = []
    maker_notes = False

    def claim(start: int, length: int, what: str, into: list[tuple[int, int]] = spans) -> None:
        if start + length > size:
            raise StructureError(f"TIFF {what} runs past the end of the file")
        into.append((start, start + length))

    def ints(typ: int, count: int, at: int) -> tuple[int, ...]:
        if typ not in (3, 4, 13):  # SHORT, LONG, IFD
            raise StructureError(f"TIFF offsets stored as field type {typ}")
        code = "H" if typ == 3 else "I"
        return struct.unpack_from(f"{order}{count}{code}", view, at)

    # (offset, part of the IFD0 -> IFD1 -> ... chain). Only that chain's
    # next-IFD links are followed: exiftool ignores them in sub-IFDs.
    todo: list[tuple[int, bool]] = [(struct.unpack_from(f"{order}I", view, 4)[0], True)]
    seen: set[int] = set()
    while todo:
        at, main = todo.pop()
        if at == 0:
            continue
        if at in seen:
            raise StructureError("TIFF IFDs form a loop")
        seen.add(at)
        claim(at, 2, "IFD")
        count = struct.unpack_from(f"{order}H", view, at)[0]
        claim(at, 2 + 12 * count + 4, "IFD")
        entries: dict[int, tuple[int, int, int]] = {}
        for i in range(count):
            pos = at + 2 + 12 * i
            tag, typ, n = struct.unpack_from(f"{order}HHI", view, pos)
            width = _TIFF_TYPE_SIZE.get(typ)
            if width is None:
                raise StructureError(f"TIFF tag 0x{tag:04X} has an unknown field type ({typ})")
            where = pos + 8
            if width * n > 4:
                where = struct.unpack_from(f"{order}I", view, pos + 8)[0]
                claim(where, width * n, f"tag 0x{tag:04X}")
            entries[tag] = (typ, n, where)
            if tag in _TIFF_MAKER_NOTES:
                maker_notes = True
            elif tag == _TIFF_XMP:
                xmp.append(bytes(view[where : where + width * n]))
            elif tag in _TIFF_SUB_IFDS and typ in (4, 13):
                todo.extend((off, False) for off in ints(typ, n, where))
        if main:
            todo.append((struct.unpack_from(f"{order}I", view, at + 2 + 12 * count)[0], True))
        for off_tag, len_tag in _TIFF_DATA:
            if off_tag not in entries and len_tag not in entries:
                continue
            if off_tag not in entries or len_tag not in entries:
                raise StructureError(f"TIFF tag 0x{off_tag:04X} or 0x{len_tag:04X} is missing")
            offsets, lengths = ints(*entries[off_tag]), ints(*entries[len_tag])
            if len(offsets) != len(lengths) or len(offsets) > _TIFF_MAX_PARTS:
                raise StructureError(f"TIFF tag 0x{off_tag:04X} doesn't match 0x{len_tag:04X}")
            for start, length in zip(offsets, lengths, strict=True):
                claim(start, length, "image data", data_spans)

    spans.sort()
    data_spans.sort()  # in place: there can be a million of them
    unreferenced = covered = 0
    for start, end in heapq.merge(spans, data_spans, [(size, size)]):
        if start > covered and not _tiff_filler(view[covered:start], magic):
            unreferenced += start - covered
        covered = max(covered, end)
    return TiffLayout(unreferenced, maker_notes, tuple(xmp), _overlap(data_spans))


def _overlap(spans: list[tuple[int, int]]) -> int:
    """How many bytes sorted ``(start, end)`` spans cover more than once, in all."""
    union = covered = 0
    for start, end in spans:
        if end > covered:
            union += end - max(start, covered)
            covered = end
    return sum(end - start for start, end in spans) - union


def _tiff_filler(gap: memoryview, magic: bytes) -> bool:
    """Bytes nothing points to that hold nothing: see ``_TIFF_PADDING``."""
    if len(gap) <= _TIFF_PADDING:
        return True
    if len(gap) > _TIFF_JUNK_MAX:
        return False
    raw = gap.tobytes()
    header = raw.find(magic)
    if header >= 0:
        raw = raw[:header] + raw[header + 8 :]
    return not raw.strip(b"\0")


# -- XMP text ------------------------------------------------------------------

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_META = "adobe:ns:meta/"
_XMP_NOTE = "http://ns.adobe.com/xmp/note/"
# As If's own namespace (asif_ns.config): its record is rewritten every time.
_ASIF_NS = "http://ns.as-if.invalid/1.0/"
_HAS_EXTENDED = f"{_XMP_NOTE} HasExtendedXMP"
# Where C2PA puts the link to Content Credentials kept outside the file.
# Matched by namespace, so any prefix counts.
_DCTERMS_PROVENANCE = "http://purl.org/dc/terms/ provenance"
# RDF syntax and bookkeeping exiftool rewrites on every write, so not values:
# rdf:about/parseType/nodeID, xml:lang (it fixes the case), the toolkit name
# (x:xmptk, and the x:xaptk of Photoshop 7-era XMP, which exiftool reads as
# the same tag and writes back as x:xmptk; the writer checks it was kept)
# and the GUID linking the extended XMP (an MD5 of that part).
_XMP_NOT_VALUES = frozenset(
    {
        f"{_RDF} about",
        "about",  # very old XMP, unprefixed
        f"{_RDF} parseType",
        f"{_RDF} nodeID",
        "http://www.w3.org/XML/1998/namespace lang",
        f"{_META} xmptk",
        f"{_META} xaptk",
        _HAS_EXTENDED,
    }
)
# No real XMP packet comes near this; a compressed one that does is refused.
_XMP_MAX = 16 * 1024 * 1024
# The trailer that ends an XMP packet. Adobe's apps (Photoshop, Bridge,
# Lightroom) rewrite a packet in place, and when the new one is shorter,
# what is left of the old one can follow the trailer.
_XPACKET_END = re.compile(rb"<\?xpacket\s+end\s*=\s*(['\"])[rw]\1\s*\?>")
# Leftovers that hold nothing: whitespace, stray angle brackets and slashes,
# and bare tags (no attributes, no text). Anything else may be a value.
_EMPTY_LEFTOVERS = re.compile(rb"(?:[\s<>/]|</?[A-Za-z_][\w.:-]*\s*/?>)*")


def leftovers(packet: bytes) -> tuple[bytes, bytes] | None:
    """``(the packet up to its trailer, what follows)`` if something other
    than whitespace follows the trailer; None if nothing does, or there is
    no trailer."""
    end = _XPACKET_END.search(packet)
    if end is None or not packet[end.end() :].strip():
        return None
    return packet[: end.end()], packet[end.end() :]


def empty_leftovers(tail: bytes) -> bool:
    """Do leftovers after an XMP packet's trailer hold nothing (no text, no
    attribute values)? Then no reader of any kind can find a value there."""
    return _EMPTY_LEFTOVERS.fullmatch(tail) is not None


LEFTOVERS_PROBLEM = (
    "its XMP is followed by leftovers of an older XMP packet, which exiftool can't carry over"
)
"""Why a file whose XMP packet is followed by what may be older values is
never written: exiftool reads them as part of the packet, then drops them or
refuses the write."""


@dataclass(frozen=True)
class XmpText:
    """What every XMP packet in a file says, read with an XML parser."""

    values: Counter[tuple[bytes, str]] = field(repr=False)
    """Every non-blank text and attribute value, stripped, with the property
    it belongs to (``xmp_property``), as ``(property, value)``, and how
    often it occurs. Excluded: RDF syntax, xml:lang, the toolkit name, the
    extended-XMP link, and everything in As If's own namespace."""
    problem: str | None = None
    """Why a write can't keep this XMP, or the Content Credentials it links
    to, intact, in plain words; None if it can."""


class _XmpReader:
    """Collects the values of XMP packets, through expat. No DTD, so no entity tricks."""

    def __init__(self, values: Counter[tuple[bytes, str]], links: set[str]) -> None:
        self.values, self.links = values, links
        self.markup = False
        self.c2pa_link = False
        self.leftovers = False
        """A packet was followed by leftovers that may hold values."""
        # Per open element: ([skipped, has children, has text, parseType Literal],
        # text so far, the property its text belongs to).
        self.stack: list[tuple[list[bool], list[str], bytes]] = []
        self.cdata: list[str] | None = None
        """The text of the CDATA section being read, if any."""

    def parse(self, packet: bytes) -> None:
        """Read one packet. Leftovers after its trailer that make it ill-formed
        are left out: only the packet itself is read, and ``leftovers`` says
        whether they might hold values (which exiftool can't carry over)."""
        # Some writers pad the packet with NULs, which XML doesn't allow.
        packet = packet.rstrip(b"\0")
        try:
            part = self._read(packet)
        except StructureError:
            split = leftovers(packet)
            if split is None:
                raise
            head, tail = split
            part = self._read(head)  # an ill-formed packet still raises
            self.leftovers = self.leftovers or not empty_leftovers(tail)
        self.values.update(part.values)
        self.links |= part.links
        self.markup = self.markup or part.markup
        self.c2pa_link = self.c2pa_link or part.c2pa_link

    @staticmethod
    def _read(packet: bytes) -> _XmpReader:
        """A reader that has read ``packet`` alone, so a failed read leaves nothing behind."""
        reader = _XmpReader(Counter(), set())
        reader._parse(packet)
        return reader

    def _parse(self, packet: bytes) -> None:
        parser = expat.ParserCreate(namespace_separator=" ")
        parser.StartDoctypeDeclHandler = self._no_dtd
        parser.EntityDeclHandler = self._no_dtd
        parser.StartElementHandler = self._start
        parser.EndElementHandler = self._end
        parser.CharacterDataHandler = self._text
        parser.StartCdataSectionHandler = self._start_cdata
        parser.EndCdataSectionHandler = self._end_cdata
        try:
            parser.Parse(packet, True)
        except expat.ExpatError as exc:
            raise StructureError("the XMP isn't well-formed XML") from exc

    def _no_dtd(self, *_args: object) -> None:
        raise StructureError("the XMP has a DTD, which XMP doesn't allow")

    def _flush(self) -> None:
        flags, text, prop = self.stack[-1]
        value = "".join(text).strip()
        text.clear()
        if value:
            flags[2] = True
            if not flags[0]:
                self.values[prop, value] += 1

    def _start(self, name: str, attrs: dict[str, str]) -> None:
        skip = name.startswith(_ASIF_NS + " ") or name == _HAS_EXTENDED
        if name == _DCTERMS_PROVENANCE or _DCTERMS_PROVENANCE in attrs:
            self.c2pa_link = True
        prop = _TOP
        if self.stack:
            self._flush()
            self.stack[-1][0][1] = True
            skip = skip or self.stack[-1][0][0]
            prop = self.stack[-1][2]
        prop = _property(prop, name)
        for key, value in attrs.items():
            if key == _HAS_EXTENDED:
                self.links.add(value.strip())
            elif value.strip() and not (
                skip or key in _XMP_NOT_VALUES or key.startswith(_ASIF_NS + " ")
            ):
                # The attribute form of a property or struct field, or
                # (rdf:resource) the value of the element it is on.
                self.values[_property(prop, key), value.strip()] += 1
        literal = attrs.get(f"{_RDF} parseType") == "Literal"
        self.stack.append(([skip, False, False, literal], [], prop))

    def _text(self, data: str) -> None:
        if self.stack:
            self.stack[-1][1].append(data)
        if self.cdata is not None:
            self.cdata.append(data)

    def _start_cdata(self) -> None:
        self.cdata = []

    def _end_cdata(self) -> None:
        text, self.cdata = "".join(self.cdata or ()), None
        # exiftool's parser doesn't know CDATA: "<b>bold</b>" inside one
        # becomes a structure, losing the text around it, and a lone "<b>"
        # makes it refuse the write. Any "<" counts, to be safe.
        if "<" in text and not (self.stack and self.stack[-1][0][0]):
            self.markup = True

    def _end(self, name: str) -> None:
        flags, text, _prop = self.stack[-1]
        if name == _HAS_EXTENDED:
            self.links.add("".join(text).strip())
        self._flush()
        skipped, children, has_text, literal = flags
        # Text beside child elements (mixed content), or markup inside a
        # parseType="Literal" value: exiftool keeps the elements as a
        # structure and silently drops the text around them.
        if children and (has_text or literal) and not skipped:
            self.markup = True
        self.stack.pop()


_TOP = bytes(16)
"""The property of a value outside any property (``xmp_property()``)."""


def xmp_property(*path: str) -> bytes:
    """The key ``read_xmp`` files a value under: its property, then the
    field inside it, and so on, each named by namespace URI, a space and
    local name (``"http://purl.org/dc/elements/1.1/ description"``).

    A property is named by its namespace URI and local name, so when
    exiftool "fixes" a namespace (Windows' MicrosoftPhoto URI) every value
    in it moves to another property. RDF's own syntax (rdf:Description,
    rdf:Bag, rdf:li, rdf:value, rdf:resource) and the x:xmpmeta wrapper
    don't count: a property reads the same written as an element or an
    attribute, in one rdf:Description or several, and a list item belongs
    to its list. Anything else inside a property is a field of it, so a
    struct's fields move with a renamed struct, as list items do with a
    renamed list.

    The key is a fixed-size digest of the whole path, not the path itself:
    every open element and every value carries one, so a path that grew
    with each level of nesting (or with the length of a crafted name) would
    make memory and time grow with the square of the XMP's size.
    """
    prop = _TOP
    for name in path:
        prop = _property(prop, name)
    return prop


def _property(outer: bytes, name: str) -> bytes:
    """The key of element or attribute ``name`` inside ``outer`` (the key of
    the element it is in; ``_TOP`` at the top). See ``xmp_property``."""
    if name.startswith((f"{_RDF} ", f"{_META} ")):
        return outer
    # ``outer`` is always 16 bytes, so no two paths hash the same input.
    return hashlib.blake2b(outer + name.encode("utf-8", "surrogatepass"), digest_size=16).digest()


def read_xmp(data: Buffer) -> XmpText:
    """The text of every XMP packet in a JPEG, PNG, WebP or TIFF file.

    Read with an XML parser rather than exiftool's, so a value exiftool
    can't read (and so wouldn't write back) still counts. ``problem`` names
    what exiftool would lose (markup inside a value, JPEG extended XMP the
    main packet doesn't link to, leftovers of an older packet after the
    trailer that may hold values) or what a write would break (Content
    Credentials kept outside the file). Leftovers that hold nothing (a stray
    ">") are no problem: no XMP reader sees them. Raises ``StructureError``
    when the file can't be parsed, isn't one of those formats, or an XMP
    packet isn't well-formed XML (or has a DTD).
    """
    extended: dict[bytes, list[tuple[int, int, bytes]]] = {}
    raw_profile = False
    with memoryview(data) as view:
        fmt = detect_format(bytes(view[:16]))
        if fmt == "tiff":
            packets = list(tiff_layout(view).xmp)
        elif fmt in ("jpeg", "png", "webp"):
            whole = data if isinstance(data, bytes) else view.tobytes()
            if fmt == "jpeg":
                packets = _jpeg_xmp(whole, extended)
            elif fmt == "png":
                packets, raw_profile = _png_xmp(whole)
            else:
                packets = [cdata for f, cdata, _r in _webp_chunks(whole)[0] if f == b"XMP "]
        else:
            raise StructureError("not a JPEG, PNG, WebP or TIFF file")

    values: Counter[tuple[bytes, str]] = Counter()
    links: set[str] = set()
    reader = _XmpReader(values, links)
    for packet in packets:
        reader.parse(packet)
    for parts in extended.values():
        reader.parse(_join_extended(parts))
    problem = None
    if reader.markup:
        problem = "its XMP has markup inside a value, which exiftool can't carry over"
    elif any(guid.decode("ascii", "replace") not in links for guid in extended):
        # exiftool only reads extended XMP the main packet links to (by its
        # MD5), and drops the rest on the next write.
        problem = "it has extended XMP its main XMP doesn't link to, which exiftool would drop"
    elif reader.leftovers:
        problem = LEFTOVERS_PROBLEM
    elif raw_profile:
        # exiftool rewrites that chunk along with the real XMP, so the
        # structure gate could never pass.
        problem = "its XMP is in ImageMagick's raw-profile form, which exiftool would rewrite"
    elif reader.c2pa_link:
        problem = C2PA_LINK_PROBLEM
    return XmpText(values, problem)


def _jpeg_xmp(data: bytes, extended: dict[bytes, list[tuple[int, int, bytes]]]) -> list[bytes]:
    packets = []
    for seg in _jpeg_layout(data).header:
        if seg.marker != _APP1:
            continue
        if seg.payload.startswith(_XMP_STD):
            packets.append(seg.payload[len(_XMP_STD) :])
        elif seg.payload.startswith(_XMP_EXT):
            # GUID (32 hex digits), full length, this part's offset, then the part.
            body = seg.payload[len(_XMP_EXT) :]
            if len(body) < 40:
                raise StructureError("JPEG extended XMP segment is truncated")
            total, offset = struct.unpack(">II", body[32:40])
            extended.setdefault(body[:32], []).append((offset, total, body[40:]))
    return packets


def _join_extended(parts: Iterable[tuple[int, int, bytes]]) -> bytes:
    out = bytearray()
    totals = set()
    for offset, total, chunk in sorted(parts):
        if offset != len(out):
            raise StructureError("JPEG extended XMP is incomplete")
        out += chunk
        totals.add(total)
    if totals != {len(out)}:
        raise StructureError("JPEG extended XMP is incomplete")
    return bytes(out)


def _png_xmp(data: bytes) -> tuple[list[bytes], bool]:
    """The PNG's XMP packets, and whether it also has ImageMagick's raw-profile XMP."""
    packets = []
    raw_profile = False
    for ctype, cdata, _raw in _png_chunks(data)[0]:
        if ctype not in _PNG_TEXT_CHUNKS:
            continue
        keyword, _, rest = cdata.partition(b"\0")
        raw_profile = raw_profile or keyword == _PNG_XMP_RAW_PROFILE
        if keyword != _PNG_XMP_KEYWORD:
            continue
        if ctype == b"tEXt":
            packets.append(rest)
        elif ctype == b"zTXt":
            packets.append(_inflate(rest[1:]))
        else:  # iTXt: compressed flag, method, language, translated keyword, text
            compressed = rest[:1] == b"\x01"
            text = rest[2:].split(b"\0", 2)
            if len(text) != 3:
                raise StructureError("PNG iTXt chunk is truncated")
            packets.append(_inflate(text[2]) if compressed else text[2])
    return packets, raw_profile


def _inflate(data: bytes) -> bytes:
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(data, _XMP_MAX + 1)
    except zlib.error as exc:
        raise StructureError("PNG compressed XMP can't be decompressed") from exc
    if len(out) > _XMP_MAX:
        raise StructureError("PNG compressed XMP is implausibly large")
    return out


# -- what a write would break unseen -----------------------------------------

_APP11 = 0xEB
# C2PA's manifest store: a JUMBF superbox ("jumb") whose description box
# ("jumd") has a type UUID starting with "c2pa" and the label "c2pa". In a
# JPEG it fills APP11 segments ("JP", box instance and sequence number
# first); PNG and WebP have a chunk of their own for it.
_C2PA_PNG_CHUNK = b"caBX"
_C2PA_WEBP_CHUNK = b"C2PA"


def _c2pa_box(box: bytes) -> bool:
    """Is this JUMBF superbox a C2PA manifest store?"""
    if box[4:8] != b"jumb" or box[12:16] != b"jumd":
        return False
    return box[16:20] == b"c2pa" or box[33:38] == b"c2pa\0"  # type UUID, or label


C2PA_PROBLEM = "it carries Content Credentials (C2PA), which any change to the file breaks"
"""Why a file with a C2PA manifest is never written, in plain words."""
C2PA_LINK_PROBLEM = (
    "its XMP may link to Content Credentials (C2PA) kept outside the file "
    "(dcterms:provenance), which any change to the file would break"
)
"""Why a file whose XMP links to a C2PA manifest stored elsewhere is never
written. That manifest's signature covers the whole file, XMP included.
Any dcterms:provenance counts: the spec allows any URI, and skipping a file
is the safe mistake."""


def write_hazard(data: bytes) -> str | None:
    """What adding XMP to this JPEG, PNG or WebP would break although every
    byte the structure check compares stays the same; None if nothing.

    - Content Credentials (C2PA): their signature covers every byte of the
      file outside the manifest, so any write invalidates them.
    - Apple's ``iDOT`` PNG chunk, with no XMP before it: exiftool puts new
      XMP right before the image data, after iDOT, which shifts the data
      iDOT points to. (``compare_png`` checks that too.)

    In plain words, never values. Raises ``StructureError`` when the file
    can't be parsed or isn't one of those formats.
    """
    fmt = detect_format(data)
    if fmt == "jpeg":
        for seg in _jpeg_layout(data).header:
            if seg.marker == _APP11 and seg.payload[:2] == b"JP" and _c2pa_box(seg.payload[8:]):
                return C2PA_PROBLEM
        return None
    if fmt == "webp":
        chunks = _webp_chunks(data)[0]
        return C2PA_PROBLEM if any(f == _C2PA_WEBP_CHUNK for f, _d, _r in chunks) else None
    if fmt != "png":
        raise StructureError("not a JPEG, PNG or WebP file")
    chunks = _png_chunks(data)[0]
    if any(ctype == _C2PA_PNG_CHUNK for ctype, _d, _r in chunks):
        return C2PA_PROBLEM
    for ctype, cdata, _raw in chunks:
        if ctype in _PNG_TEXT_CHUNKS and cdata.split(b"\0", 1)[0] == _PNG_XMP_KEYWORD:
            return None  # exiftool rewrites XMP where it is: here, ahead of any iDOT
        if ctype == _PNG_IDOT:
            return "it has Apple's iDOT chunk, whose offsets new XMP would break"
    return None


# -- dispatch ------------------------------------------------------------------


def compare(before: bytes, after: bytes, *, allow_iptc: bool) -> list[str] | None:
    """Structure differences, dispatching on ``detect_format(before)``.

    ``None`` means there's no structure check for this format (TIFF, or not
    recognised). A format change is itself a difference. Raises
    ``StructureError`` if either side can't be parsed.
    """
    fmt_b, fmt_a = detect_format(before), detect_format(after)
    if fmt_b != fmt_a:
        return [
            f"file format changed ({_FORMAT_NAMES[fmt_b]} before, {_FORMAT_NAMES[fmt_a]} after)"
        ]
    if fmt_b == "jpeg":
        return compare_jpeg(before, after, allow_iptc=allow_iptc)
    if fmt_b == "png":
        return compare_png(before, after, allow_iptc=allow_iptc)
    if fmt_b == "webp":
        return compare_webp(before, after)
    return None
