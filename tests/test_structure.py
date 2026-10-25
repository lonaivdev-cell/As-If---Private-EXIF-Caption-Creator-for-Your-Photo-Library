"""Byte-level structure checks, mostly on hand-built files.

Hand-built bytes make each JPEG/PNG/WebP corner case explicit; a few tests
also run on Pillow output and the generated fixtures to prove the parsers
agree with real encoders.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image

from asif.meta.structure import (
    StructureError,
    compare,
    compare_jpeg,
    compare_png,
    compare_webp,
    detect_format,
    jpeg_eoi_offset,
    jpeg_trailer,
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
