"""Byte-level structure checks, mostly on hand-built files.

Hand-built bytes make each JPEG/PNG/WebP/TIFF corner case explicit; a few
tests also run on Pillow output and the generated fixtures to prove the
parsers agree with real encoders.
"""

from __future__ import annotations

import hashlib
import io
import mmap
import struct
import zlib
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from asif.meta.structure import (
    C2PA_PROBLEM,
    StructureError,
    compare,
    compare_jpeg,
    compare_png,
    compare_webp,
    detect_format,
    jpeg_eoi_offset,
    jpeg_trailer,
    read_xmp,
    tiff_layout,
    write_hazard,
)

SECRET = b"GPS -22.951916 my secret caption"

# -- builders ------------------------------------------------------------------

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def seg(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


JFIF = seg(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
EXIF = seg(0xE1, b"Exif\x00\x00MM\x00*\x00\x00\x00\x08" + SECRET)
XMP = seg(0xE1, b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>" + SECRET + b"</x:xmpmeta>")
XMP_EXT = seg(0xE1, b"http://ns.adobe.com/xmp/extension/\x00" + b"0" * 40 + b"<more/>")
IPTC = seg(0xED, b"Photoshop 3.0\x008BIM\x04\x04\x00\x00\x00\x00\x00\x05" + SECRET)
DQT0 = seg(0xDB, b"\x00" + bytes(range(1, 65)))
DQT1 = seg(0xDB, b"\x01" + bytes(range(2, 66)))
SOF0 = seg(0xC0, b"\x08\x00\x10\x00\x10\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01")
DHT = seg(0xC4, b"\x00" + b"\x01" + b"\x00" * 15 + b"\x00")
SOS = seg(0xDA, b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00")
HEADER = (JFIF, EXIF, DQT0, DQT1, SOF0, DHT)


def jpeg(*segments: bytes, scan: bytes = b"\x12\x34\x56", trailer: bytes = b"") -> bytes:
    return SOI + b"".join(segments or HEADER) + SOS + scan + EOI + trailer


def chunk(ctype: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)


PNG_SIG = b"\x89PNG\r\n\x1a\n"
IHDR = chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
IDAT = chunk(b"IDAT", zlib.compress(b"\x00" + b"\x10" * 48))
IEND = chunk(b"IEND", b"")
GAMA = chunk(b"gAMA", struct.pack(">I", 45455))
PNG_XMP = chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + SECRET)


def png(*chunks: bytes, tail: bytes = b"") -> bytes:
    return PNG_SIG + IHDR + b"".join(chunks) + IDAT + IEND + tail


# Apple's iDOT: where the image data continues, counted from iDOT's own start.
IDOT = chunk(b"iDOT", struct.pack(">7I", 2, 0, 8, 40, 8, 8, 40 + len(IDAT)))


def rchunk(fourcc: bytes, data: bytes) -> bytes:
    pad = b"\x00" if len(data) % 2 else b""
    return fourcc + struct.pack("<I", len(data)) + data + pad


def riff(*chunks: bytes, tail: bytes = b"") -> bytes:
    body = b"WEBP" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body + tail


def vp8x(flags: int, width: int = 640, height: int = 480, reserved: bytes = b"\0\0\0") -> bytes:
    data = bytes([flags]) + reserved
    data += (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return rchunk(b"VP8X", data)


# A VP8 key frame header for 640x480 followed by some (fake) partition data.
VP8 = rchunk(b"VP8 ", b"\x30\x7f\x00\x9d\x01\x2a" + struct.pack("<HH", 640, 480) + b"\x11" * 21)


def vp8l(width: int, height: int, alpha: bool) -> bytes:
    bits = (width - 1) | (height - 1) << 14 | int(alpha) << 28
    return rchunk(b"VP8L", b"\x2f" + struct.pack("<I", bits) + b"\x22" * 10)


WEBP_XMP = rchunk(b"XMP ", b"<x:xmpmeta>" + SECRET + b"</x:xmpmeta>")
ICCP = rchunk(b"ICCP", b"\x00\x00\x02\x0clcms" + b"\x01" * 20)
ALPH = rchunk(b"ALPH", b"\x00" + b"\x07" * 9)

# (tag, field type, count, value): the value's bytes, or the offset of a blob already placed.
Entry = tuple[int, int, int, bytes | int]


class Tiff:
    """A classic TIFF, laid out in the order its parts are added.

    Build bottom-up: image data and sub-IFDs first, so their offsets are
    known when the IFD pointing to them is written; IFD0 last.
    """

    def __init__(self, order: str = "<") -> None:
        self.order = order
        self.buf = bytearray((b"II*\x00" if order == "<" else b"MM\x00*") + b"\x00" * 4)

    def here(self) -> int:
        """Where the next part goes (word-aligned, as TIFF wants)."""
        if len(self.buf) % 2:
            self.buf += b"\x00"
        return len(self.buf)

    def blob(self, data: bytes) -> int:
        at = self.here()
        self.buf += data
        return at

    def ifd(self, *entries: Entry, next_ifd: int = 0) -> int:
        at = self.here()
        table = bytearray(struct.pack(f"{self.order}H", len(entries)))
        values = bytearray()
        values_at = at + 2 + 12 * len(entries) + 4
        for tag, typ, count, value in sorted(entries, key=lambda e: e[0]):
            head = struct.pack(f"{self.order}HHI", tag, typ, count)
            if isinstance(value, int):
                table += head + struct.pack(f"{self.order}I", value)
            elif len(value) <= 4:
                table += head + value.ljust(4, b"\x00")
            else:
                table += head + struct.pack(f"{self.order}I", values_at + len(values))
                values += value + b"\x00" * (len(value) % 2)
        table += struct.pack(f"{self.order}I", next_ifd)
        self.buf += table + values
        return at

    def done(self, ifd0: int) -> bytes:
        self.buf[4:8] = struct.pack(f"{self.order}I", ifd0)
        return bytes(self.buf)

    def short(self, tag: int, *values: int) -> Entry:
        return (tag, 3, len(values), struct.pack(f"{self.order}{len(values)}H", *values))

    def long(self, tag: int, *values: int) -> Entry:
        return (tag, 4, len(values), struct.pack(f"{self.order}{len(values)}I", *values))

    def page(self, width: int, height: int, *extra: Entry, seed: int = 0, next_ifd: int = 0) -> int:
        """An uncompressed RGB image in one strip, plus ``extra`` entries; its IFD's offset."""
        pixels = bytes((seed + i) % 251 for i in range(width * height * 3))
        strip = self.blob(pixels)
        return self.ifd(
            self.long(256, width),
            self.long(257, height),
            self.short(258, 8, 8, 8),
            self.short(259, 1),  # no compression
            self.short(262, 2),  # RGB
            self.long(273, strip),
            self.short(277, 3),
            self.long(278, height),
            self.long(279, len(pixels)),
            *extra,
            next_ifd=next_ifd,
        )


def ascii_entry(tag: int, text: str) -> Entry:
    raw = text.encode() + b"\x00"
    return (tag, 2, len(raw), raw)


def undefined(tag: int, raw: bytes) -> Entry:
    return (tag, 7, len(raw), raw)


def canon_maker_note(at: int, trailing: bytes = b"") -> bytes:
    """A Canon-style maker note to be placed at TIFF offset ``at``: an IFD whose
    out-of-line values use absolute offsets (as Canon's do), then ``trailing``
    bytes nothing points to."""
    image_type = b"Canon EOS TEST\x00\x00"
    firmware = b"Firmware 1.0.0\x00\x00"
    values_at = at + 2 + 12 * 3 + 4
    return (
        struct.pack("<H", 3)
        + struct.pack("<HHII", 0x0006, 2, len(image_type), values_at)
        + struct.pack("<HHII", 0x0007, 2, len(firmware), values_at + len(image_type))
        + struct.pack("<HHII", 0x0010, 4, 1, 0x80000001)  # ModelID, inline
        + struct.pack("<I", 0)
        + image_type
        + firmware
        + trailing
    )


def assert_private(messages: list[str]) -> None:
    for msg in messages:
        assert "secret" not in msg and "22.95" not in msg, msg


# -- detect_format -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\xff\xd8\xff\xe0rest", "jpeg"),
        (PNG_SIG + b"rest", "png"),
        (b"RIFF\x10\x00\x00\x00WEBPVP8 ", "webp"),
        (b"II*\x00\x08\x00\x00\x00", "tiff"),
        (b"MM\x00*\x00\x00\x00\x08", "tiff"),
        (b"II+\x00\x08\x00\x00\x00", "tiff"),  # BigTIFF
        (b"RIFF\x10\x00\x00\x00WAVEfmt ", None),
        (b"\x00\x00\x00\x18ftypheic", None),
        (b"\xff\xd8", None),  # too short to be sure
        (b"", None),
    ],
)
def test_detect_format_by_magic_bytes(data: bytes, expected: str | None) -> None:
    assert detect_format(data) == expected


# -- JPEG walking ------------------------------------------------------------------


def test_eoi_of_a_simple_jpeg_is_its_end() -> None:
    data = jpeg()
    assert jpeg_eoi_offset(data) == len(data)
    assert jpeg_trailer(data) == b""


def test_trailer_is_everything_after_the_primary_eoi() -> None:
    # A second JPEG (MPF) and a fake MP4 after the primary image.
    trailer = jpeg(scan=b"\x99") + b"\x00\x00\x00\x18ftypmp42" + EOI + b"\xff\xd9\xff"
    data = jpeg(trailer=trailer)
    assert jpeg_eoi_offset(data) == len(data) - len(trailer)
    assert jpeg_trailer(data) == trailer


@pytest.mark.parametrize(
    "scan",
    [
        pytest.param(b"\x01\xff\x00\x02\xff\x00\xd9\x03", id="ff00-stuffing"),
        pytest.param(b"\x01\xff\xd0\x02\xff\xd7\x03\xff\xd3", id="restart-markers"),
        pytest.param(b"\x01\x02\xff\xff\xff", id="fill-bytes-before-eoi"),
        pytest.param(b"", id="empty-scan"),
        pytest.param(b"\xff\x00" * 1000, id="all-stuffing"),
    ],
)
def test_eoi_inside_entropy_coded_data(scan: bytes) -> None:
    data = jpeg(scan=scan, trailer=b"TRAILER")
    assert jpeg_trailer(data) == b"TRAILER"


def test_progressive_scans_with_tables_between_them() -> None:
    data = (
        SOI
        + b"".join(HEADER)
        + SOS
        + b"\x01\xff\x00\x02"
        + DHT
        + SOS
        + b"\x03\xff\xd1\x04"
        + seg(0xDD, b"\x00\x04")  # DRI between scans
        + SOS
        + b"\x05"
        + EOI
        + b"TAIL"
    )
    assert jpeg_trailer(data) == b"TAIL"


def test_marker_like_bytes_inside_segments_are_skipped() -> None:
    # An EXIF thumbnail is a whole JPEG inside APP1: its EOI must not end the walk.
    thumb_app1 = seg(0xE1, b"Exif\x00\x00" + jpeg(scan=b"\x01"))
    data = jpeg(JFIF, thumb_app1, DQT0, SOF0, DHT, trailer=b"T")
    assert jpeg_trailer(data) == b"T"


def test_fill_bytes_before_header_markers() -> None:
    data = SOI + b"\xff\xff" + JFIF[0:] + b"\xff" + DQT0 + SOF0 + DHT + SOS + b"\x01" + EOI
    assert jpeg_eoi_offset(data) == len(data)


def test_file_without_a_scan_still_ends_at_eoi() -> None:
    data = SOI + DQT0 + EOI + b"x"
    assert jpeg_trailer(data) == b"x"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\x89PNG\r\n\x1a\n", id="not-a-jpeg"),
        pytest.param(SOI, id="soi-only"),
        pytest.param(jpeg()[:-2], id="missing-eoi"),
        pytest.param(jpeg()[:-1], id="eoi-cut-in-half"),
        pytest.param(SOI + JFIF[:-3], id="segment-runs-past-the-end"),
        pytest.param(SOI + b"\xff\xe1\x40", id="length-field-cut"),
        pytest.param(SOI + b"\xff\xe1\x00\x01" + EOI, id="length-below-two"),
        pytest.param(SOI + b"\xff\xe1\x7f\xff" + EOI, id="length-past-the-end"),
        pytest.param(SOI + JFIF + b"junk" + EOI, id="stray-bytes-between-segments"),
        pytest.param(SOI + JFIF + SOS + b"\x01\x02\x03", id="scan-runs-to-the-end"),
        pytest.param(SOI + JFIF + SOS + b"\x01\xff", id="scan-ends-on-ff"),
        pytest.param(SOI + SOI + EOI, id="nested-soi"),
        pytest.param(SOI + JFIF + b"\xff\xff\xff", id="fill-bytes-then-eof"),
    ],
)
def test_unparseable_jpegs_raise(data: bytes) -> None:
    with pytest.raises(StructureError) as exc:
        jpeg_eoi_offset(data)
    assert_private([str(exc.value)])


def _pillow_jpeg(**kw: object) -> bytes:
    im = Image.linear_gradient("L").convert("RGB").resize((96, 64))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90, **kw)
    return buf.getvalue()


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="baseline"),
        pytest.param({"progressive": True}, id="progressive"),
        pytest.param({"restart_marker_blocks": 1}, id="restart-markers"),
        pytest.param({"progressive": True, "restart_marker_blocks": 2}, id="both"),
    ],
)
def test_real_encoder_output(options: dict[str, object]) -> None:
    data = _pillow_jpeg(**options)
    assert jpeg_eoi_offset(data) == len(data)
    with_trailer = data + data  # e.g. an MPF secondary image
    assert jpeg_trailer(with_trailer) == data


def test_fixture_jpegs(fixtures_dir: Path) -> None:
    for name in ("plain.jpg", "exif.jpg", "progressive.jpg", "rotated.jpg"):
        data = (fixtures_dir / name).read_bytes()
        assert jpeg_trailer(data) == b"", name
    motion = (fixtures_dir / "motion.jpg").read_bytes()
    assert jpeg_trailer(motion).endswith(b"MotionPhoto_Data")
    mpo = (fixtures_dir / "mpo.jpg").read_bytes()
    assert jpeg_trailer(mpo).startswith(b"\xff\xd8\xff")
    with pytest.raises(StructureError):
        jpeg_trailer((fixtures_dir / "truncated.jpg").read_bytes())
    with pytest.raises(StructureError):
        jpeg_trailer((fixtures_dir / "corrupt.jpg").read_bytes())


# -- compare_jpeg ------------------------------------------------------------------


def test_identical_jpegs() -> None:
    assert compare_jpeg(jpeg(), jpeg(), allow_iptc=False) == []


@pytest.mark.parametrize(
    "after",
    [
        pytest.param(jpeg(JFIF, EXIF, XMP, DQT0, DQT1, SOF0, DHT), id="xmp-added"),
        pytest.param(jpeg(JFIF, EXIF, XMP, XMP_EXT, XMP_EXT, DQT0, DQT1, SOF0, DHT), id="ext-xmp"),
        pytest.param(jpeg(JFIF, XMP, EXIF, DQT0, DQT1, SOF0, DHT), id="xmp-before-exif"),
    ],
)
def test_xmp_segments_are_ignored(after: bytes) -> None:
    assert compare_jpeg(jpeg(), after, allow_iptc=False) == []
    assert compare_jpeg(after, jpeg(), allow_iptc=False) == []


def test_changed_xmp_is_ignored() -> None:
    before = jpeg(JFIF, EXIF, XMP, DQT0, DQT1, SOF0, DHT)
    other_xmp = seg(0xE1, b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta>new</x:xmpmeta>")
    after = jpeg(JFIF, EXIF, other_xmp, DQT0, DQT1, SOF0, DHT)
    assert compare_jpeg(before, after, allow_iptc=False) == []


def test_xmp_lookalike_is_not_ignored() -> None:
    # No NUL after the namespace: not an XMP segment, so it must match.
    fake = seg(0xE1, b"http://ns.adobe.com/xap/1.0/" + SECRET)
    diffs = compare_jpeg(jpeg(), jpeg(JFIF, EXIF, fake, DQT0, DQT1, SOF0, DHT), allow_iptc=False)
    assert diffs and "APP1" in diffs[0]
    assert_private(diffs)


def test_iptc_only_ignored_when_allowed() -> None:
    after = jpeg(JFIF, EXIF, IPTC, DQT0, DQT1, SOF0, DHT)
    assert compare_jpeg(jpeg(), after, allow_iptc=True) == []
    diffs = compare_jpeg(jpeg(), after, allow_iptc=False)
    assert diffs == [
        "JPEG segment count differs (6 before, 7 after); "
        "first difference at JPEG segment 3 (DQT before, APP13 after)"
    ]


def test_app13_that_is_not_photoshop_is_compared() -> None:
    other = seg(0xED, b"Adobe_CM\x00" + SECRET)
    after = jpeg(JFIF, EXIF, other, DQT0, DQT1, SOF0, DHT)
    assert compare_jpeg(jpeg(), after, allow_iptc=True) != []


def test_changed_exif_segment() -> None:
    edited = seg(0xE1, b"Exif\x00\x00MM\x00*\x00\x00\x00\x08" + SECRET.replace(b"9", b"8"))
    diffs = compare_jpeg(jpeg(), jpeg(JFIF, edited, DQT0, DQT1, SOF0, DHT), allow_iptc=True)
    assert diffs == [
        f"JPEG segment 2 (APP1) differs ({len(EXIF)} bytes before, {len(edited)} after)"
    ]
    assert_private(diffs)


def test_inserted_app_segment() -> None:
    app15 = seg(0xEF, b"vendor junk")
    diffs = compare_jpeg(jpeg(), jpeg(JFIF, app15, EXIF, DQT0, DQT1, SOF0, DHT), allow_iptc=True)
    assert len(diffs) == 1 and "count differs" in diffs[0] and "APP15" in diffs[0]


def test_reordered_segments() -> None:
    diffs = compare_jpeg(jpeg(), jpeg(JFIF, EXIF, DQT1, DQT0, SOF0, DHT), allow_iptc=False)
    assert diffs == [
        f"JPEG segment 3 (DQT) differs ({len(DQT0)} bytes before, {len(DQT1)} after)",
        f"JPEG segment 4 (DQT) differs ({len(DQT1)} bytes before, {len(DQT0)} after)",
    ]
    diffs = compare_jpeg(jpeg(), jpeg(EXIF, JFIF, DQT0, DQT1, SOF0, DHT), allow_iptc=False)
    assert diffs[0] == "JPEG segment 1 is APP0 before, APP1 after"


def test_removed_segment() -> None:
    diffs = compare_jpeg(jpeg(), jpeg(JFIF, DQT0, DQT1, SOF0, DHT), allow_iptc=False)
    assert len(diffs) == 1 and "6 before, 5 after" in diffs[0]


def test_fill_bytes_count_as_content() -> None:
    padded = SOI + b"\xff" + b"".join(HEADER) + SOS + b"\x12\x34\x56" + EOI
    assert compare_jpeg(jpeg(), padded, allow_iptc=False) != []


def test_changed_image_data() -> None:
    diffs = compare_jpeg(jpeg(), jpeg(scan=b"\x12\x34\x57"), allow_iptc=False)
    n = len(SOS) + 3 + len(EOI)  # from the first SOS marker to EOI
    assert diffs == [
        f"JPEG image data (first scan to end of image) differs ({n} bytes before, {n} after)"
    ]


def test_changed_tables_between_progressive_scans() -> None:
    def prog(dht_byte: bytes) -> bytes:
        dht = seg(0xC4, b"\x10" + dht_byte * 16 + b"\x00")
        return SOI + b"".join(HEADER) + SOS + b"\x01" + dht + SOS + b"\x02" + EOI

    assert compare_jpeg(prog(b"\x00"), prog(b"\x00"), allow_iptc=False) == []
    assert compare_jpeg(prog(b"\x00"), prog(b"\x01"), allow_iptc=False) != []


@pytest.mark.parametrize(
    ("trailer_after", "message"),
    [
        (b"MOTION-PHOTO-VIDEO-DATA!", "(23 bytes before, 24 after)"),
        (b"MOTION-PHOTO-VIDEO-DAT", "(23 bytes before, 22 after)"),
        (b"MOTION-PHOTO-VIDEO-DATB", "(23 bytes before, 23 after)"),
        (b"", "(23 bytes before, 0 after)"),
    ],
)
def test_trailer_changes(trailer_after: bytes, message: str) -> None:
    before = jpeg(trailer=b"MOTION-PHOTO-VIDEO-DATA")
    diffs = compare_jpeg(before, jpeg(trailer=trailer_after), allow_iptc=True)
    assert diffs == [f"JPEG trailer (after end of image) differs {message}"]


def test_compare_jpeg_raises_on_unparseable_side() -> None:
    with pytest.raises(StructureError):
        compare_jpeg(jpeg(), jpeg()[:-2], allow_iptc=False)
    with pytest.raises(StructureError):
        compare_jpeg(b"garbage", jpeg(), allow_iptc=False)


# -- compare_png -------------------------------------------------------------------


def test_identical_pngs() -> None:
    assert compare_png(png(GAMA), png(GAMA)) == []


@pytest.mark.parametrize("ctype", [b"iTXt", b"tEXt", b"zTXt"])
def test_png_xmp_chunks_are_ignored(ctype: bytes) -> None:
    xmp = chunk(ctype, b"XML:com.adobe.xmp\x00" + SECRET)
    assert compare_png(png(GAMA), png(GAMA, xmp)) == []
    assert compare_png(png(xmp, GAMA), png(GAMA, PNG_XMP)) == []


def test_png_xmp_after_idat_is_ignored() -> None:
    before = PNG_SIG + IHDR + IDAT + IEND
    after = PNG_SIG + IHDR + IDAT + PNG_XMP + IEND
    assert compare_png(before, after) == []


def test_png_iptc_only_ignored_when_allowed() -> None:
    iptc = chunk(b"zTXt", b"Raw profile type iptc\x00\x00" + zlib.compress(SECRET))
    assert compare_png(png(GAMA), png(iptc, GAMA, PNG_XMP), allow_iptc=True) == []
    diffs = compare_png(png(GAMA), png(iptc, GAMA, PNG_XMP))
    assert diffs and "zTXt" in diffs[0]
    assert_private(diffs)


def test_png_other_text_chunks_are_compared() -> None:
    desc = chunk(b"tEXt", b"Description\x00" + SECRET)
    edited = chunk(b"tEXt", b"Description\x00" + SECRET.upper())
    diffs = compare_png(png(desc), png(edited))
    assert diffs == [f"PNG chunk 2 (tEXt) differs ({len(desc)} bytes before, {len(edited)} after)"]
    assert_private(diffs)


def test_png_ancillary_chunk_changed_added_reordered() -> None:
    gama2 = chunk(b"gAMA", struct.pack(">I", 50000))
    phys = chunk(b"pHYs", struct.pack(">IIB", 2835, 2835, 1))
    assert compare_png(png(GAMA), png(gama2)) == [
        f"PNG chunk 2 (gAMA) differs ({len(GAMA)} bytes before, {len(gama2)} after)"
    ]
    assert "count differs" in compare_png(png(GAMA), png(GAMA, phys))[0]
    assert len(compare_png(png(GAMA, phys), png(phys, GAMA))) == 2


def test_png_crc_is_compared() -> None:
    bad_crc = GAMA[:-1] + bytes([GAMA[-1] ^ 1])
    assert compare_png(png(GAMA), png(bad_crc)) != []


def test_png_bytes_after_iend() -> None:
    assert compare_png(png(tail=b"extra"), png(tail=b"extra")) == []
    for tail_b, tail_a in [(b"", b"x"), (b"x", b""), (b"x", b"y")]:
        diffs = compare_png(png(tail=tail_b), png(tail=tail_a))
        assert diffs == [
            f"PNG data after IEND differs ({len(tail_b)} bytes before, {len(tail_a)} after)"
        ]


def test_png_idot_must_keep_pointing_at_the_image_data() -> None:
    # Two IDATs after iDOT: nothing may appear or change size between them.
    bigger = chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + SECRET * 3)
    moved = ["PNG iDOT chunk no longer points at the same image data"]
    assert compare_png(png(IDOT, IDAT), png(IDOT, PNG_XMP, IDAT)) == moved
    assert compare_png(png(IDOT, PNG_XMP, IDAT), png(IDOT, bigger, IDAT)) == moved
    # XMP ahead of iDOT, or after the image data, may change: iDOT still points right.
    assert compare_png(png(PNG_XMP, IDOT, IDAT), png(bigger, IDOT, IDAT)) == []
    assert compare_png(png(IDOT, IDAT), PNG_SIG + IHDR + IDOT + IDAT + IDAT + PNG_XMP + IEND) == []


def test_png_image_data_changed() -> None:
    other_idat = chunk(b"IDAT", zlib.compress(b"\x00" + b"\x11" * 48))
    after = PNG_SIG + IHDR + other_idat + IEND
    assert compare_png(png(), after) != []


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\xff\xd8\xff\xe0", id="not-a-png"),
        pytest.param(PNG_SIG + IHDR + IDAT, id="missing-iend"),
        pytest.param(PNG_SIG + IHDR + IDAT[:-3], id="chunk-runs-past-the-end"),
        pytest.param(PNG_SIG + IHDR + b"\x00\x00", id="chunk-header-cut"),
    ],
)
def test_unparseable_pngs_raise(data: bytes) -> None:
    with pytest.raises(StructureError):
        compare_png(data, png())
    with pytest.raises(StructureError):
        compare_png(png(), data)


def test_png_fixtures_parse(fixtures_dir: Path) -> None:
    for name in ("plain.png", "alpha.png"):
        data = (fixtures_dir / name).read_bytes()
        assert compare_png(data, data) == []


# -- compare_webp ------------------------------------------------------------------


def test_identical_webp() -> None:
    assert compare_webp(riff(VP8), riff(VP8)) == []


def test_simple_webp_gains_vp8x_with_xmp() -> None:
    assert compare_webp(riff(VP8), riff(vp8x(0x04), VP8, WEBP_XMP)) == []


def test_xmp_chunk_changed_or_removed_is_ignored() -> None:
    before = riff(vp8x(0x04), VP8, WEBP_XMP)
    other = rchunk(b"XMP ", b"<x:xmpmeta>new</x:xmpmeta>")
    assert compare_webp(before, riff(vp8x(0x04), VP8, other)) == []
    assert compare_webp(before, riff(vp8x(0x00), VP8)) == []


def test_lossless_with_alpha_may_gain_the_alpha_flag() -> None:
    image = vp8l(320, 240, alpha=True)
    assert compare_webp(riff(image), riff(vp8x(0x04, 320, 240), image, WEBP_XMP)) == []
    assert compare_webp(riff(image), riff(vp8x(0x14, 320, 240), image, WEBP_XMP)) == []


def test_alph_chunk_allows_the_alpha_flag() -> None:
    # Not valid without VP8X in practice, but the rule is about what may be gained.
    assert compare_webp(riff(ALPH, VP8), riff(vp8x(0x14), ALPH, VP8, WEBP_XMP)) == []


def test_alpha_flag_without_alpha_is_a_difference() -> None:
    diffs = compare_webp(riff(VP8), riff(vp8x(0x14), VP8, WEBP_XMP))
    assert diffs == ["WebP gained a VP8X header with flags other than XMP/alpha"]
    opaque = vp8l(320, 240, alpha=False)
    assert compare_webp(riff(opaque), riff(vp8x(0x14, 320, 240), opaque)) != []


@pytest.mark.parametrize("flag", [0x02, 0x08, 0x20, 0x01, 0x40, 0x80])
def test_gained_vp8x_with_other_flags(flag: int) -> None:
    assert compare_webp(riff(VP8), riff(vp8x(0x04 | flag), VP8)) != []


def test_gained_vp8x_reserved_bytes_must_be_zero() -> None:
    assert compare_webp(riff(VP8), riff(vp8x(0x04, reserved=b"\0\1\0"), VP8)) != []


def test_gained_vp8x_canvas_must_match_the_image() -> None:
    diffs = compare_webp(riff(VP8), riff(vp8x(0x04, 641, 480), VP8))
    assert diffs == ["WebP gained a VP8X header whose canvas doesn't match the image"]
    image = vp8l(320, 240, alpha=False)
    assert compare_webp(riff(image), riff(vp8x(0x04, 320, 241), image)) != []


def test_vp8x_on_both_sides_xmp_flag_may_change() -> None:
    before = riff(vp8x(0x20), ICCP, VP8)
    assert compare_webp(before, riff(vp8x(0x24), ICCP, VP8, WEBP_XMP)) == []


@pytest.mark.parametrize("flag", [0x02, 0x08, 0x10, 0x20])
def test_vp8x_on_both_sides_other_flags_must_match(flag: int) -> None:
    before = riff(vp8x(0x04), VP8, WEBP_XMP)
    diffs = compare_webp(before, riff(vp8x(0x04 | flag), VP8, WEBP_XMP))
    assert diffs == ["WebP VP8X flags differ (other than the XMP flag)"]


def test_vp8x_on_both_sides_canvas_and_reserved_bytes() -> None:
    before = riff(vp8x(0x04), VP8)
    assert compare_webp(before, riff(vp8x(0x04, 640, 481), VP8)) == [
        "WebP VP8X canvas size differs"
    ]
    assert compare_webp(before, riff(vp8x(0x04, reserved=b"\1\0\0"), VP8)) == [
        "WebP VP8X flags differ (other than the XMP flag)"
    ]


def test_vp8x_removed_is_a_difference() -> None:
    diffs = compare_webp(riff(vp8x(0x00), VP8), riff(VP8))
    assert diffs == ["WebP VP8X header was removed"]


def test_webp_non_xmp_chunks_are_compared() -> None:
    exif = rchunk(b"EXIF", b"MM\x00*" + SECRET)
    edited = rchunk(b"EXIF", b"MM\x00*" + SECRET.upper())
    before = riff(vp8x(0x08), VP8, exif)
    diffs = compare_webp(before, riff(vp8x(0x08), VP8, edited))
    assert diffs == [f"WebP chunk 2 (EXIF) differs ({len(exif)} bytes before, {len(edited)} after)"]
    assert_private(diffs)
    assert compare_webp(before, riff(vp8x(0x08), exif, VP8)) != []  # reordered
    assert compare_webp(before, riff(vp8x(0x08), VP8, exif, rchunk(b"ABCD", b"x"))) != []


def test_webp_image_data_changed() -> None:
    other = rchunk(b"VP8 ", VP8[8:-1] + b"\x12")
    assert compare_webp(riff(VP8), riff(other)) != []


def test_webp_odd_sized_chunks_and_padding() -> None:
    odd = rchunk(b"ABCD", b"abc")
    assert len(odd) == 12  # header + 3 bytes + pad
    assert compare_webp(riff(VP8, odd), riff(vp8x(0x04), VP8, odd, WEBP_XMP)) == []
    other_pad = odd[:-1] + b"\x01"
    assert compare_webp(riff(VP8, odd), riff(VP8, other_pad)) != []


def test_webp_bytes_after_riff() -> None:
    assert compare_webp(riff(VP8, tail=b"junk"), riff(VP8, tail=b"junk")) == []
    assert compare_webp(riff(VP8, tail=b"junk"), riff(VP8)) == [
        "WebP data after the RIFF data differs (4 bytes before, 0 after)"
    ]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"RIFF\x04\x00\x00\x00WAVE", id="not-webp"),
        pytest.param(riff(VP8)[:-4], id="riff-size-past-the-end"),
        pytest.param(b"RIFF\x00\x00\x00\x00WEBPVP8 ", id="riff-size-too-small"),
        pytest.param(b"RIFF\x0c\x00\x00\x00WEBPVP8 ", id="chunk-header-cut"),
        pytest.param(b"RIFF\x10\x00\x00\x00WEBPVP8 \xff\x00\x00\x00", id="chunk-past-the-end"),
        pytest.param(riff(rchunk(b"VP8X", b"\x04\x00")), id="short-vp8x"),
    ],
)
def test_unparseable_webp_raises(data: bytes) -> None:
    with pytest.raises(StructureError):
        compare_webp(data, data)


def test_gained_vp8x_needs_a_readable_bitstream() -> None:
    broken = rchunk(b"VP8 ", b"\x00" * 4)
    with pytest.raises(StructureError):
        compare_webp(riff(broken), riff(vp8x(0x04), broken))
    with pytest.raises(StructureError):
        compare_webp(riff(ICCP), riff(vp8x(0x04), ICCP))  # no image at all


def test_webp_fixtures_parse(fixtures_dir: Path) -> None:
    for name in ("lossy.webp", "lossless.webp"):
        data = (fixtures_dir / name).read_bytes()
        assert compare_webp(data, data) == []


# -- write_hazard -------------------------------------------------------------------

C2PA_UUID = b"c2pa" + bytes.fromhex("00110010800000AA00389B71")


def jumbf(label: bytes = b"c2pa", uuid: bytes = C2PA_UUID) -> bytes:
    """A JUMBF superbox with only its description box, as a C2PA manifest store begins."""
    jumd = b"jumd" + uuid + b"\x03" + label + b"\x00"
    jumb = b"jumb" + struct.pack(">I", 4 + len(jumd)) + jumd
    return struct.pack(">I", 4 + len(jumb)) + jumb


def app11(box: bytes) -> bytes:
    """A JPEG APP11 segment holding JUMBF: "JP", box instance 1, sequence number 1."""
    return seg(0xEB, b"JP\x00\x01\x00\x00\x00\x01" + box)


def jpeg_with(*segments: bytes) -> bytes:
    return jpeg(JFIF, *segments, DQT0, SOF0, DHT)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(jpeg_with(app11(jumbf())), id="jpeg"),
        pytest.param(jpeg_with(app11(jumbf(uuid=b"\x00" * 16))), id="jpeg-label-only"),
        pytest.param(jpeg_with(app11(jumbf(label=b"manifests"))), id="jpeg-type-only"),
        pytest.param(png(chunk(b"caBX", jumbf())), id="png"),
        pytest.param(riff(VP8, rchunk(b"C2PA", jumbf())), id="webp"),
    ],
)
def test_content_credentials_are_a_hazard(data: bytes) -> None:
    # Their signature covers every byte outside the manifest: any write breaks it.
    assert write_hazard(data) == C2PA_PROBLEM


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(jpeg(), id="jpeg"),
        pytest.param(jpeg_with(app11(jumbf(b"jpxt", b"\x01" * 16))), id="other-jumbf"),
        pytest.param(jpeg_with(seg(0xEB, b"JP\x00\x01")), id="cut-short-app11"),
        pytest.param(jpeg_with(seg(0xEB, b"HDR_" + SECRET)), id="other-app11"),
        pytest.param(png(GAMA, PNG_XMP), id="png"),
        pytest.param(png(PNG_XMP, IDOT), id="png-xmp-ahead-of-idot"),
        pytest.param(riff(vp8x(0x04), VP8, WEBP_XMP), id="webp"),
    ],
)
def test_no_hazard(data: bytes) -> None:
    assert write_hazard(data) is None


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(png(IDOT), id="no-xmp"),
        pytest.param(png(IDOT, PNG_XMP), id="xmp-after-idot"),
        pytest.param(PNG_SIG + IHDR + IDOT + IDAT + PNG_XMP + IEND, id="xmp-after-the-data"),
    ],
)
def test_apple_idot_without_xmp_ahead_of_it_is_a_hazard(data: bytes) -> None:
    # exiftool puts new XMP right before the image data, after iDOT.
    assert write_hazard(data) == "it has Apple's iDOT chunk, whose offsets new XMP would break"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"II*\x00\x08\x00\x00\x00", id="tiff"),
        pytest.param(b"GIF89a" + SECRET, id="unknown"),
        pytest.param(jpeg()[:-2], id="jpeg-cut"),
        pytest.param(png()[:-4], id="png-cut"),
    ],
)
def test_write_hazard_raises_for_what_it_cant_parse(data: bytes) -> None:
    with pytest.raises(StructureError) as exc:
        write_hazard(data)
    assert_private([str(exc.value)])


# -- compare -----------------------------------------------------------------------


def test_compare_dispatches_by_content() -> None:
    assert compare(jpeg(), jpeg(JFIF, EXIF, XMP, DQT0, DQT1, SOF0, DHT), allow_iptc=False) == []
    assert compare(png(), png(PNG_XMP), allow_iptc=False) == []
    assert compare(riff(VP8), riff(vp8x(0x04), VP8, WEBP_XMP), allow_iptc=False) == []
    iptc = jpeg(JFIF, EXIF, IPTC, DQT0, DQT1, SOF0, DHT)
    assert compare(jpeg(), iptc, allow_iptc=True) == []
    assert compare(jpeg(), iptc, allow_iptc=False) != []


def test_compare_has_no_check_for_tiff_or_unknown() -> None:
    tiff = b"II*\x00\x08\x00\x00\x00" + b"\x00" * 32
    assert compare(tiff, tiff + b"more", allow_iptc=False) is None
    assert compare(b"whatever", b"else", allow_iptc=False) is None


def test_a_format_change_is_a_difference() -> None:
    assert compare(jpeg(), png(), allow_iptc=False) == [
        "file format changed (JPEG before, PNG after)"
    ]
    assert compare(png(), b"garbage", allow_iptc=False) == [
        "file format changed (PNG before, unknown after)"
    ]
    tiff = b"II*\x00\x08\x00\x00\x00"
    assert compare(tiff, jpeg(), allow_iptc=False) == [
        "file format changed (TIFF before, JPEG after)"
    ]


def test_compare_raises_when_a_side_cant_be_parsed() -> None:
    with pytest.raises(StructureError):
        compare(jpeg(), jpeg()[:-2], allow_iptc=False)


# -- tiff_layout -------------------------------------------------------------------


@pytest.mark.parametrize("order", ["<", ">"])
def test_tiff_simple_layout_is_fully_referenced(order: str) -> None:
    t = Tiff(order)
    layout = tiff_layout(t.done(t.page(16, 8, ascii_entry(0x010F, "Maker"))))
    assert layout.unreferenced == 0
    assert not layout.maker_notes
    assert layout.xmp == ()


def test_tiff_every_ifd_exiftool_rewrites_is_followed() -> None:
    t = Tiff()
    page2 = t.page(8, 8, seed=7)  # IFD1, through the next-IFD link
    sub = t.page(4, 4, seed=9)  # a reduced-size SubIFD
    interop = t.ifd(ascii_entry(0x0001, "R98"))
    exif = t.ifd(ascii_entry(0x9003, "2019:07:14 10:20:30"), t.long(0xA005, interop))
    gps = t.ifd((0x0000, 1, 4, b"\x02\x03\x00\x00"), ascii_entry(0x0001, "S"))
    thumb = t.blob(b"\xff\xd8thumbnail\xff\xd9")
    ifd2 = t.ifd(t.long(0x0201, thumb), t.long(0x0202, 13))
    t.buf[page2 + 2 + 12 * 9 : page2 + 2 + 12 * 9 + 4] = struct.pack("<I", ifd2)  # IFD1 -> IFD2
    sub_ptr = (0x014A, 13, 1, struct.pack("<I", sub))  # type IFD
    ifd0 = t.page(16, 16, sub_ptr, t.long(0x8769, exif), t.long(0x8825, gps), next_ifd=page2)
    assert tiff_layout(t.done(ifd0)).unreferenced == 0


def test_tiff_tiles_are_followed() -> None:
    t = Tiff()
    tiles = [t.blob(bytes([i]) * 48) for i in range(4)]
    ifd0 = t.ifd(
        t.long(256, 8), t.long(257, 8), t.long(322, 4), t.long(323, 4),
        t.long(324, *tiles), t.long(325, *[48] * 4),
    )  # fmt: skip
    assert tiff_layout(t.done(ifd0)).unreferenced == 0


def test_tiff_data_behind_a_private_offset_tag_is_unreferenced() -> None:
    # A scanner's private tag holding the offset of its own block: exiftool
    # copies the number, not the block, so the block would be lost.
    t = Tiff()
    vendor = t.blob(b"VENDOR-SCAN-CALIBRATION:" + SECRET * 10)
    data = t.done(t.page(16, 8, t.long(0xC5AB, vendor), t.long(0xC5AC, 344)))
    assert tiff_layout(data).unreferenced == 344
    # The same bytes as the value of a tag (an out-of-line UNDEFINED) are covered.
    t = Tiff()
    data = t.done(t.page(16, 8, undefined(0xC5AB, b"VENDOR-SCAN-CALIBRATION:" + SECRET * 10)))
    assert tiff_layout(data).unreferenced == 0


def _with_gap(gap: bytes) -> bytes:
    t = Tiff()
    strip = t.page(4, 4)  # a page whose IFD comes after a gap
    t.blob(gap)
    return t.done(t.page(4, 4, next_ifd=strip))


@pytest.mark.parametrize(
    ("gap", "unreferenced"),
    [
        pytest.param(b"\x00" * 200, 0, id="zeros"),
        pytest.param(b"\x07\x07", 0, id="two-bytes-of-padding"),
        pytest.param(b"\x00" * 10 + b"II*\x00\x58\x00\x00\x00" + b"\x00" * 4, 0, id="stale-header"),
        pytest.param(b"\x00" * 10 + b"MM\x00*\x00\x00\x00\x08", 18, id="other-byte-order"),
        pytest.param(b"II*\x00\x08\x00\x00\x00II*\x00\x08\x00\x00\x00", 16, id="two-headers"),
        pytest.param(b"\x00" * 30 + b"\x01" + b"\x00" * 30, 62, id="one-byte-of-data"),
        pytest.param(b"\x00" * (64 * 1024 + 2), 64 * 1024 + 2, id="too-long-for-filler"),
    ],
)
def test_tiff_gaps_hold_filler_or_data(gap: bytes, unreferenced: int) -> None:
    assert tiff_layout(_with_gap(gap)).unreferenced == unreferenced


@pytest.mark.parametrize(
    ("tail", "unreferenced"),
    [(b"", 0), (b"\x00\x01\x02", 0), (b"\x00" * 64, 0), (SECRET, len(SECRET))],
)
def test_tiff_bytes_after_the_last_part(tail: bytes, unreferenced: int) -> None:
    t = Tiff()
    data = t.done(t.page(4, 4)) + tail
    assert tiff_layout(data).unreferenced == unreferenced


@pytest.mark.parametrize("tag", [0x927C, 0xC634])  # MakerNote, DNGPrivateData
def test_tiff_maker_notes_are_reported(tag: int) -> None:
    t = Tiff()
    exif = t.ifd(undefined(tag, canon_maker_note(0)))
    layout = tiff_layout(t.done(t.page(4, 4, t.long(0x8769, exif))))
    assert layout.maker_notes
    assert layout.unreferenced == 0  # its bytes are covered; exiftool rebuilds them anyway


def test_tiff_xmp_packets_are_returned() -> None:
    t = Tiff()
    page2 = t.page(4, 4, undefined(0x02BC, b"<page2/>"))
    data = t.done(t.page(4, 4, undefined(0x02BC, b"<x:xmpmeta/>"), next_ifd=page2))
    assert tiff_layout(data).xmp == (b"<x:xmpmeta/>", b"<page2/>")
    assert "xmpmeta" not in repr(tiff_layout(data))


def _loop() -> bytes:
    t = Tiff()
    ifd0 = t.here()
    return t.done(t.ifd(t.long(256, 1), next_ifd=ifd0))


def _past_end() -> bytes:
    t = Tiff()
    return t.done(t.ifd(t.long(273, 10_000), t.long(279, 10)))


def _unknown_type() -> bytes:
    t = Tiff()
    return t.done(t.ifd((0xC000, 99, 1, b"\x00\x00\x00\x00")))


def _mismatched_strips() -> bytes:
    t = Tiff()
    return t.done(t.ifd(t.long(273, 8, 8), t.long(279, 1)))


def _counts_missing() -> bytes:
    t = Tiff()
    return t.done(t.ifd(t.long(273, 8)))


def _offsets_as_text() -> bytes:
    t = Tiff()
    return t.done(t.ifd(ascii_entry(273, "8"), t.long(279, 1)))


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"II+\x00\x08\x00\x00\x00" + b"\x00" * 16, id="bigtiff"),
        pytest.param(b"\xff\xd8\xff\xe0" + b"\x00" * 16, id="not-a-tiff"),
        pytest.param(b"II*\x00", id="header-cut"),
        pytest.param(b"II*\x00\xff\x00\x00\x00", id="ifd-past-the-end"),
        pytest.param(_loop(), id="ifd-loop"),
        pytest.param(_past_end(), id="strip-past-the-end"),
        pytest.param(_unknown_type(), id="unknown-field-type"),
        pytest.param(_mismatched_strips(), id="offsets-and-counts-differ"),
        pytest.param(_counts_missing(), id="byte-counts-missing"),
        pytest.param(_offsets_as_text(), id="offsets-as-text"),
    ],
)
def test_unparseable_tiffs_raise(data: bytes) -> None:
    with pytest.raises(StructureError) as exc:
        tiff_layout(data)
    assert_private([str(exc.value)])


def test_tiff_layout_takes_a_memory_map(tmp_path: Path) -> None:
    t = Tiff()
    path = tmp_path / "photo.tif"
    path.write_bytes(t.done(t.page(4, 4, undefined(0x02BC, xmp_packet("<dc:title>x</dc:title>")))))
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as view:
        assert tiff_layout(view).unreferenced == 0
        assert read_xmp(view).values == Counter({"x": 1})
    # Leaving the block closed the map: nothing kept a view of it. Not even on an error:
    path.write_bytes(_loop())
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as view:
        with pytest.raises(StructureError):
            tiff_layout(view)
        with pytest.raises(StructureError):
            read_xmp(view)


def pillow_tiff(pages: int = 1, **options: object) -> bytes:
    """A Pillow TIFF: ``pages`` pages (different sizes and modes), saved with ``options``."""
    images = [
        Image.new("RGB", (64, 48), (10, 120, 200)),
        Image.new("RGB", (32, 24), (200, 10, 20)),
        Image.new("L", (8, 8), 7),
    ][:pages]
    buf = io.BytesIO()
    images[0].save(buf, "TIFF", save_all=pages > 1, append_images=images[1:], **options)
    return buf.getvalue()


SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pillow_tiff(), id="plain"),
        pytest.param(pillow_tiff(compression="tiff_lzw", icc_profile=SRGB), id="lzw-icc"),
        pytest.param(pillow_tiff(compression="jpeg"), id="jpeg-compressed"),
        # Pillow's multi-page writer leaves each appended page's own TIFF header behind.
        pytest.param(pillow_tiff(2), id="two-pages"),
        pytest.param(pillow_tiff(3, compression="tiff_lzw"), id="three-pages-lzw"),
    ],
)
def test_pillow_tiffs_are_fully_referenced(data: bytes) -> None:
    assert tiff_layout(data).unreferenced == 0


def test_tiff_fixture_is_fully_referenced(fixtures_dir: Path) -> None:
    layout = tiff_layout((fixtures_dir / "plain.tif").read_bytes())
    assert layout.unreferenced == 0
    assert not layout.maker_notes


# -- read_xmp ------------------------------------------------------------------------

_NAMESPACES = (
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
    'xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/" '
    'xmlns:xmpNote="http://ns.adobe.com/xmp/note/" '
    'xmlns:asif="http://ns.as-if.invalid/1.0/"'
)


def xmp_packet(body: str, attrs: str = "") -> bytes:
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Some Toolkit 1.0">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="uuid:1234" {_NAMESPACES} {attrs}>{body}'
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    ).encode()


def jpeg_xmp(*segments: bytes) -> bytes:
    return jpeg(JFIF, *segments, DQT0, SOF0, DHT)


def std_xmp(packet: bytes) -> bytes:
    return seg(0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + packet)


def ext_xmp(guid: bytes, packet: bytes, parts: int = 2) -> list[bytes]:
    size = -(-len(packet) // parts)
    return [
        seg(
            0xE1,
            b"http://ns.adobe.com/xmp/extension/\x00"
            + guid
            + struct.pack(">II", len(packet), at)
            + packet[at : at + size],
        )
        for at in range(0, len(packet), size)
    ]


OWNER_XMP = xmp_packet(
    '<dc:title><rdf:Alt><rdf:li xml:lang="x-default">Garden</rdf:li></rdf:Alt></dc:title>'
    "<dc:subject><rdf:Bag><rdf:li>family</rdf:li><rdf:li> family </rdf:li></rdf:Bag></dc:subject>"
    '<photoshop:Source rdf:resource="http://example.com/x"/>'
    "<photoshop:Instructions><![CDATA[a & c]]></photoshop:Instructions>"
    "<asif:RunId>run-1</asif:RunId>"
    "<asif:AddedKeywords><rdf:Bag><rdf:li>dog</rdf:li></rdf:Bag></asif:AddedKeywords>"
    "<xmpNote:HasExtendedXMP>0123456789ABCDEF0123456789ABCDEF</xmpNote:HasExtendedXMP>",
    attrs='photoshop:City="  Rio  " asif:ModelId="stub" rdf:nodeID="n1"',
)
OWNER_VALUES = Counter({"Garden": 1, "family": 2, "http://example.com/x": 1, "a & c": 1, "Rio": 1})


def test_xmp_values_skip_syntax_and_as_ifs_own_record() -> None:
    text = read_xmp(jpeg_xmp(std_xmp(OWNER_XMP)))
    assert text.values == OWNER_VALUES
    assert text.problem is None
    assert "Garden" not in repr(text)


def test_no_xmp_means_no_values() -> None:
    assert read_xmp(jpeg()) == read_xmp(png()) == read_xmp(riff(VP8))
    assert read_xmp(jpeg()).values == Counter()


PNG_KEY = b"XML:com.adobe.xmp\x00"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(png(chunk(b"iTXt", PNG_KEY + b"\x00\x00\x00\x00" + OWNER_XMP)), id="itxt"),
        pytest.param(
            png(chunk(b"iTXt", PNG_KEY + b"\x01\x00en\x00\x00" + zlib.compress(OWNER_XMP))),
            id="itxt-compressed",
        ),
        pytest.param(png(chunk(b"zTXt", PNG_KEY + b"\x00" + zlib.compress(OWNER_XMP))), id="ztxt"),
        pytest.param(png(chunk(b"tEXt", PNG_KEY + OWNER_XMP)), id="text"),
        pytest.param(riff(vp8x(0x04), VP8, rchunk(b"XMP ", OWNER_XMP)), id="webp"),
        pytest.param(jpeg_xmp(std_xmp(OWNER_XMP + b"\x00\x00\x00")), id="nul-padding"),
    ],
)
def test_xmp_is_found_in_every_format(data: bytes) -> None:
    assert read_xmp(data).values == OWNER_VALUES


def test_xmp_in_a_tiff() -> None:
    t = Tiff(">")
    data = t.done(t.page(4, 4, undefined(0x02BC, OWNER_XMP)))
    assert read_xmp(data).values == OWNER_VALUES


def test_other_png_text_is_not_xmp() -> None:
    assert read_xmp(png(chunk(b"tEXt", b"Comment\x00" + SECRET))).values == Counter()


def test_imagemagick_raw_profile_xmp_is_a_problem() -> None:
    # exiftool reads this hex-encoded copy as XMP and rewrites it on every write.
    raw = chunk(b"zTXt", b"Raw profile type xmp\x00\x00" + zlib.compress(b"\nxmp\n  4\n3c612f3e\n"))
    text = read_xmp(png(raw, chunk(b"iTXt", PNG_KEY + b"\x00\x00\x00\x00" + OWNER_XMP)))
    assert text.problem == (
        "its XMP is in ImageMagick's raw-profile form, which exiftool would rewrite"
    )
    assert text.values == OWNER_VALUES


LITERAL = '<xmpRights:UsageTerms rdf:parseType="Literal">{}</xmpRights:UsageTerms>'


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(LITERAL.format("Use only with <i>credit</i> to Jane"), id="literal"),
        pytest.param(LITERAL.format("<i>credit</i>"), id="literal-markup-only"),
        pytest.param("<photoshop:Headline>Head <b>bold</b> tail</photoshop:Headline>", id="mixed"),
    ],
)
def test_markup_inside_a_value_is_a_problem(body: str) -> None:
    # exiftool keeps the elements as a structure and drops the text around them.
    text = read_xmp(jpeg_xmp(std_xmp(xmp_packet(body))))
    assert text.problem == "its XMP has markup inside a value, which exiftool can't carry over"
    assert "credit" in text.values or "bold" in text.values  # the values are still counted


@pytest.mark.parametrize(
    "cdata",
    [
        pytest.param("Lisbon <b>old town</b>", id="element"),  # exiftool: "City B: old town"
        pytest.param("a <br/> b", id="empty-element"),
        pytest.param("a <b> & c", id="unclosed"),  # exiftool refuses the write
        pytest.param("x < y", id="lone-angle-bracket"),  # fine for exiftool; refused anyway
    ],
)
def test_markup_inside_a_cdata_value_is_a_problem(cdata: str) -> None:
    # exiftool's parser doesn't know CDATA: it reads the markup as elements.
    body = f"<photoshop:City><![CDATA[{cdata}]]></photoshop:City>"
    text = read_xmp(jpeg_xmp(std_xmp(xmp_packet(body))))
    assert text.problem == "its XMP has markup inside a value, which exiftool can't carry over"
    assert text.values[cdata] == 1


@pytest.mark.parametrize(
    "body",
    [
        LITERAL.format("Use only with credit"),
        '<dc:rights rdf:parseType="Resource"><photoshop:Inner>x</photoshop:Inner></dc:rights>',
        "<asif:Note>As If <b>own</b></asif:Note>",
        "<photoshop:City><![CDATA[Lisbon & old town]]></photoshop:City>",
        "<asif:Note><![CDATA[As If <b>own</b>]]></asif:Note>",
    ],
)
def test_plain_values_and_structures_are_no_problem(body: str) -> None:
    assert read_xmp(jpeg_xmp(std_xmp(xmp_packet(body)))).problem is None


def test_old_adobe_toolkit_attribute_is_not_a_value() -> None:
    # Photoshop 7-era XMP: exiftool reads x:xaptk as the toolkit name and
    # writes it back as x:xmptk, so it is bookkeeping, like x:xmptk.
    packet = OWNER_XMP.replace(b'x:xmptk="Some Toolkit 1.0"', b'x:xaptk="XMP toolkit 2.8.2-33"')
    assert b"x:xaptk" in packet
    assert read_xmp(jpeg_xmp(std_xmp(packet))).values == OWNER_VALUES


def test_extended_xmp_linked_from_the_main_packet() -> None:
    extended = xmp_packet("<photoshop:History>" + "long history " * 50 + "</photoshop:History>")
    guid = hashlib.md5(extended).hexdigest().upper().encode()  # XMP's own choice of hash
    main = xmp_packet("", attrs=f'xmpNote:HasExtendedXMP="{guid.decode()}"')
    text = read_xmp(jpeg_xmp(std_xmp(main), *ext_xmp(guid, extended, parts=3)))
    assert text.problem is None
    assert text.values == Counter({("long history " * 50).strip(): 1})


@pytest.mark.parametrize("link", ["", 'xmpNote:HasExtendedXMP="FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"'])
def test_extended_xmp_nobody_links_to_is_a_problem(link: str) -> None:
    # exiftool ignores it ("non-standard extended XMP") and drops it on write.
    extended = xmp_packet("<photoshop:History>depth map</photoshop:History>")
    guid = hashlib.md5(extended).hexdigest().upper().encode()
    text = read_xmp(jpeg_xmp(std_xmp(xmp_packet("", attrs=link)), *ext_xmp(guid, extended)))
    assert text.problem == (
        "it has extended XMP its main XMP doesn't link to, which exiftool would drop"
    )


def test_incomplete_extended_xmp_raises() -> None:
    extended = xmp_packet("<photoshop:History>depth map</photoshop:History>")
    guid = b"0" * 32
    parts = ext_xmp(guid, extended, parts=3)
    with pytest.raises(StructureError, match="incomplete"):
        read_xmp(jpeg_xmp(std_xmp(xmp_packet("")), parts[0], parts[2]))
    with pytest.raises(StructureError, match="truncated"):
        read_xmp(jpeg_xmp(seg(0xE1, b"http://ns.adobe.com/xmp/extension/\x00" + guid)))


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(jpeg_xmp(std_xmp(b"<x:xmpmeta>" + SECRET)), id="not-well-formed"),
        pytest.param(jpeg_xmp(XMP), id="unbound-prefix"),
        pytest.param(
            jpeg_xmp(std_xmp(b'<!DOCTYPE x [<!ENTITY a "' + SECRET + b'">]><x>&a;&a;</x>')),
            id="dtd-entities",
        ),
        pytest.param(
            png(chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x01\x00\x00\x00" + b"not zlib")),
            id="bad-compression",
        ),
        pytest.param(
            png(chunk(b"zTXt", b"XML:com.adobe.xmp\x00\x00" + zlib.compress(b" " * (17 << 20)))),
            id="compression-bomb",
        ),
        pytest.param(png(chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00")), id="itxt-cut"),
        pytest.param(b"GIF89a" + b"\x00" * 16, id="unknown-format"),
        pytest.param(jpeg()[:-2], id="jpeg-cut"),
    ],
)
def test_unreadable_xmp_raises(data: bytes) -> None:
    with pytest.raises(StructureError) as exc:
        read_xmp(data)
    assert_private([str(exc.value)])


def test_fixture_xmp(fixtures_dir: Path) -> None:
    text = read_xmp((fixtures_dir / "user_xmp.jpg").read_bytes())
    assert {"Grandma's garden, summer 2003", "Garden", "family"} <= text.values.keys()
    assert text.problem is None
    for name in ("plain.jpg", "plain.png", "lossy.webp", "plain.tif"):
        assert read_xmp((fixtures_dir / name).read_bytes()).problem is None
