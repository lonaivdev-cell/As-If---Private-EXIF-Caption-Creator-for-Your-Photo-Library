"""Byte-level structure checks for JPEG, PNG and WebP. Pure Python, no exiftool.

These back up exiftool's ``ImageDataHash`` with something stronger: apart
from the XMP packet (and IPTC, when As If writes it), every byte of the file
must be identical before and after a write, in the same order. They also
find the JPEG trailer, where Samsung and Google "motion photos" and MPF
multi-picture files (Ultra HDR gain maps, depth maps) keep their extra data.

Every function takes whole files as ``bytes`` and is pure. Messages name
segments, chunks and byte counts, never file contents: these files are
somebody's photos.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Literal

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
                header.append(_JpegSegment(marker, data[start:pos], b""))
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
            header.append(_JpegSegment(marker, data[start:seg_end], data[pos + 2 : seg_end]))
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
# exiftool's (non-standard) home for IPTC in PNG, used for compat = "wide".
_PNG_IPTC_KEYWORD = b"Raw profile type iptc"


def _chunk_name(fourcc: bytes) -> str:
    """A chunk type for messages; hex if it isn't the usual four letters."""
    if fourcc.isascii() and fourcc.decode("ascii").replace(" ", "").isalnum():
        return fourcc.decode("ascii").rstrip()
    return "0x" + fourcc.hex()


def _png_chunks(data: bytes) -> tuple[list[tuple[bytes, bytes, bytes]], bytes]:
    """``([(type, data, raw chunk), ...], bytes after IEND)``."""
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
        chunks.append((ctype, data[pos + 8 : pos + 8 + length], data[pos:chunk_end]))
        pos = chunk_end
        if ctype == b"IEND":
            return chunks, data[pos:]


def compare_png(before: bytes, after: bytes, *, allow_iptc: bool = False) -> list[str]:
    """Chunk-for-chunk (type, data and CRC) in order; ``[]`` means no differences.

    Ignores ``iTXt``/``tEXt``/``zTXt`` chunks whose keyword is
    ``XML:com.adobe.xmp`` and, when ``allow_iptc``, ``Raw profile type
    iptc`` (where exiftool puts IPTC in a PNG). Bytes after ``IEND`` must
    match too. Raises ``StructureError`` if either side can't be parsed.
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
