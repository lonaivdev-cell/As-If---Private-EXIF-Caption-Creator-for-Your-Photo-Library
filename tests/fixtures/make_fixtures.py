"""Generate the synthetic fixture photos (§16, pass 3 of docs/DESIGN.md).

Nothing here is a real photo: every file is drawn by Pillow and tagged by
exiftool (or laid out byte by byte, as a camera would), so the fixture set
is reproducible and contains nobody's data.

    python tests/fixtures/make_fixtures.py OUTPUT_DIR [--with-panorama]

Each fixture's purpose is in ``FIXTURES`` below. ``expect`` is what the
pipeline must do with it: ``ok`` (captioned), ``quarantine`` (refused before
anything is written) or ``skip-format`` (a format As If doesn't write yet, or
a file it can't prove a write safe for).
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import subprocess
import sys
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageDraw

EXIFTOOL = shutil.which("exiftool") or "exiftool"


@dataclass(frozen=True)
class Fixture:
    name: str
    expect: str
    purpose: str


FIXTURES: tuple[Fixture, ...] = (
    Fixture("plain.jpg", "ok", "JPEG without any metadata"),
    Fixture("exif.jpg", "ok", "JPEG with EXIF dates, camera, ICC profile and a thumbnail"),
    Fixture("gps.jpg", "ok", "JPEG with GPS coordinates"),
    Fixture("rotated.jpg", "ok", "JPEG with Orientation = 6 (rotate 90 CW)"),
    Fixture("progressive.jpg", "ok", "progressive JPEG (several scans)"),
    Fixture("motion.jpg", "ok", "JPEG with a motion-photo style trailer after EOI"),
    Fixture("mpo.jpg", "ok", "multi-picture JPEG (MPF), like Ultra HDR gain maps"),
    Fixture("user_xmp.jpg", "ok", "JPEG whose owner already wrote a caption and keywords"),
    Fixture("user_iptc.jpg", "ok", "JPEG with Latin-1 IPTC keywords from an old app"),
    Fixture("plain.png", "ok", "PNG without metadata"),
    Fixture("alpha.png", "ok", "PNG with transparency"),
    Fixture("lossy.webp", "ok", "lossy WebP (simple format, no VP8X)"),
    Fixture("lossless.webp", "ok", "lossless WebP with alpha"),
    Fixture("plain.tif", "ok", "uncompressed TIFF"),
    Fixture("truncated.jpg", "quarantine", "JPEG cut off halfway through the image data"),
    Fixture("corrupt.jpg", "quarantine", "random bytes with a .jpg name"),
    Fixture("photo.heic", "skip-format", "HEIC (not written until the fixture pass proves it)"),
    Fixture("raw.dng", "skip-format", "RAW gets sidecars only, not in M1"),
    Fixture(
        "sony_preview.jpg",
        "skip-format",
        "Sony JPEG whose maker note points into the MPF preview after the image",
    ),
    Fixture(
        "olympus_preview.jpg",
        "skip-format",
        "Olympus JPEG whose maker note points to a preview after the image",
    ),
    Fixture(
        "pentax_preview.jpg",
        "ok",
        "Pentax JPEG with its preview in an APP7 segment that new XMP moves",
    ),
    Fixture("panorama.jpg", "ok", "50 MP panorama (slow; only with --with-panorama)"),
)


def _scene(size: tuple[int, int], seed: int) -> Image.Image:
    """A deterministic, photo-ish picture: sky, ground, a sun and a box."""
    w, h = size
    im = Image.new("RGB", size, (90 + seed % 40, 150, 220))
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, h * 2 // 3, w, h], fill=(60, 140 - seed % 30, 60))
    r = max(4, min(w, h) // 8)
    draw.ellipse([w * 3 // 4 - r, h // 5 - r, w * 3 // 4 + r, h // 5 + r], fill=(250, 220, 80))
    draw.rectangle([w // 5, h // 2, w // 5 + w // 6, h * 2 // 3], fill=(150, 60, 40))
    for i in range(0, w, max(8, w // 40)):  # some texture so JPEG has real entropy data
        draw.line([i, h * 2 // 3, i + h // 10, h], fill=(40 + (i * 7 + seed) % 60, 110, 50))
    return im


def _jpeg(im: Image.Image, **kw: object) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, **kw)
    return buf.getvalue()


def _exiftool(path: Path, *args: str) -> None:
    subprocess.run(
        [EXIFTOOL, "-q", "-overwrite_original", *args, str(path)],
        check=True,
        capture_output=True,
    )


def _plain_jpg(out: Path) -> None:
    out.write_bytes(_jpeg(_scene((640, 480), 1)))


def _exif_jpg(out: Path) -> None:
    im = _scene((800, 600), 2)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif()
    exif[0x010F] = "AsIfTestCam"  # Make
    exif[0x0110] = "Model 1"  # Model
    exif[0x0132] = "2019:07:14 10:20:30"  # DateTime
    out.write_bytes(_jpeg(im, exif=exif.tobytes(), icc_profile=icc))
    _exiftool(
        out,
        "-EXIF:DateTimeOriginal=2019:07:14 10:20:30",
        "-EXIF:CreateDate=2019:07:14 10:20:30",
        "-EXIF:OffsetTimeOriginal=+02:00",
        "-EXIF:ExposureTime=1/250",
        "-EXIF:FNumber=2.8",
        "-EXIF:ISO=100",
        "-EXIF:Artist=Test Person",
    )
    thumb = io.BytesIO()
    _scene((160, 120), 2).save(thumb, "JPEG", quality=70)
    thumb_path = out.with_suffix(".thumb.jpg")
    thumb_path.write_bytes(thumb.getvalue())
    try:
        _exiftool(out, f"-ThumbnailImage<={thumb_path}")
    finally:
        thumb_path.unlink()


def _gps_jpg(out: Path) -> None:
    out.write_bytes(_jpeg(_scene((640, 480), 3)))
    _exiftool(
        out,
        "-EXIF:DateTimeOriginal=2021:12:24 18:00:00",
        "-GPSLatitude=-22.951916",
        "-GPSLatitudeRef=S",
        "-GPSLongitude=-43.210487",
        "-GPSLongitudeRef=W",
        "-GPSAltitude=710",
        "-GPSAltitudeRef=0",
    )


def _rotated_jpg(out: Path) -> None:
    out.write_bytes(_jpeg(_scene((480, 640), 4)))
    _exiftool(out, "-IFD0:Orientation=6", "-n")


def _progressive_jpg(out: Path) -> None:
    out.write_bytes(_jpeg(_scene((640, 480), 5), progressive=True, optimize=True))


def _motion_jpg(out: Path) -> None:
    # Real motion photos append an MP4 after the JPEG's EOI and describe it in
    # XMP. This imitates the layout: the trailer must survive byte-for-byte.
    primary = _jpeg(_scene((640, 480), 6))
    trailer = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 16 + b"MotionPhoto_Data"
    out.write_bytes(primary + trailer)
    _exiftool(
        out,
        "-XMP-GCamera:MotionPhoto=1",
        "-XMP-GCamera:MotionPhotoVersion=1",
        f"-XMP-GCamera:MicroVideoOffset={len(trailer)}",
    )


def _mpo_jpg(out: Path) -> None:
    primary = _scene((640, 480), 7)
    secondary = _scene((320, 240), 8).convert("L").convert("RGB")
    buf = io.BytesIO()
    primary.save(buf, "MPO", save_all=True, append_images=[secondary], quality=88)
    out.write_bytes(buf.getvalue())


def _user_xmp_jpg(out: Path) -> None:
    out.write_bytes(_jpeg(_scene((640, 480), 9)))
    _exiftool(
        out,
        "-XMP-dc:Description-x-default=Grandma's garden, summer 2003",
        "-XMP-dc:Subject=Garden",
        "-XMP-dc:Subject=family",
        "-XMP-dc:Creator=Owner",
    )


def _user_iptc_jpg(out: Path) -> None:
    out.write_bytes(_jpeg(_scene((640, 480), 10)))
    _exiftool(out, "-charset", "iptc=Latin", "-IPTC:Keywords=café", "-IPTC:Keywords=old")


def _plain_png(out: Path) -> None:
    _scene((640, 480), 11).save(out, "PNG")


def _alpha_png(out: Path) -> None:
    im = _scene((320, 240), 12).convert("RGBA")
    im.putalpha(Image.linear_gradient("L").resize((320, 240)))
    im.save(out, "PNG")


def _lossy_webp(out: Path) -> None:
    _scene((640, 480), 13).save(out, "WEBP", quality=80)


def _lossless_webp(out: Path) -> None:
    im = _scene((320, 240), 14).convert("RGBA")
    im.putalpha(200)
    im.save(out, "WEBP", lossless=True)


def _plain_tif(out: Path) -> None:
    _scene((320, 240), 15).save(out, "TIFF")


def _truncated_jpg(out: Path) -> None:
    data = _jpeg(_scene((640, 480), 16))
    out.write_bytes(data[: len(data) // 2])


def _corrupt_jpg(out: Path) -> None:
    # Deterministic junk (no Math.random in fixtures): a counter through a LCG.
    state = 12345
    junk = bytearray()
    for _ in range(4096):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        junk.append(state >> 16 & 0xFF)
    out.write_bytes(bytes(junk))


def _heic(out: Path) -> None:
    # Only the extension matters for M1 (HEIC is skipped by format).
    out.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64)


def _raw_dng(out: Path) -> None:
    # A TIFF-shaped stub with a .dng name: RAW is skipped by extension in M1.
    out.write_bytes(b"II*\x00" + struct.pack("<I", 8) + b"\x00" * 64)


def _panorama_jpg(out: Path) -> None:
    out.write_bytes(_jpeg(_scene((10000, 5000), 17)))
    _exiftool(out, "-EXIF:DateTimeOriginal=2020:01:01 12:00:00")


# -- camera JPEGs, built byte by byte ------------------------------------------
# Laid out the way the cameras write them. Their maker notes point to a preview
# by offset, so each is built twice: once to measure, then with the offsets in.

# (tag, field type, count, value): the value's bytes, or an offset already known.
IfdEntry = tuple[int, int, int, bytes | int]


def ifd(at: int, *entries: IfdEntry) -> bytes:
    """A little-endian TIFF IFD placed at offset ``at``, its longer values right
    after it. Offsets count from where ``at`` does."""
    table = struct.pack("<H", len(entries))
    values = b""
    values_at = at + 2 + 12 * len(entries) + 4
    for tag, typ, count, value in sorted(entries, key=lambda e: e[0]):
        head = struct.pack("<HHI", tag, typ, count)
        if isinstance(value, int):
            table += head + struct.pack("<I", value)
        elif len(value) <= 4:
            table += head + value.ljust(4, b"\x00")
        else:
            table += head + struct.pack("<I", values_at + len(values))
            values += value + b"\x00" * (len(value) % 2)
    return table + b"\x00\x00\x00\x00" + values


def text(tag: int, value: str) -> IfdEntry:
    raw = value.encode() + b"\x00"
    return (tag, 2, len(raw), raw)


def long(tag: int, value: int) -> IfdEntry:
    return (tag, 4, 1, value)


def segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload


def scene_jpeg(size: tuple[int, int], seed: int) -> bytes:
    """The test scene as a Pillow JPEG (SOI, JFIF, tables, frame, scan, EOI)."""
    return _jpeg(_scene(size, seed))


def jpeg_body(data: bytes) -> bytes:
    """A Pillow JPEG without its SOI and JFIF segment: tables, frame, scan, EOI."""
    assert data[:4] == b"\xff\xd8\xff\xe0", "expected Pillow's JFIF JPEG"
    return data[4 + int.from_bytes(data[4:6], "big") :]


def exif_segment(make: str, model: str, maker_note: Callable[[int], bytes]) -> bytes:
    """APP1 EXIF with Make, Model, a date and ``maker_note(at)``: the maker note
    as it should be placed at TIFF offset ``at``."""
    tags = (text(0x010F, make), text(0x0110, model), long(0x8769, 0))
    exif_at = 8 + len(ifd(8, *tags))
    exif_tags = [(0x9000, 7, 4, b"0230"), text(0x9003, "2013:06:01 12:00:00")]
    note_at = exif_at + len(ifd(exif_at, *exif_tags, (0x927C, 7, 0, 0)))
    note = maker_note(note_at)
    tiff = (
        b"II*\x00\x08\x00\x00\x00"
        + ifd(8, *tags[:2], long(0x8769, exif_at))
        + ifd(exif_at, *exif_tags, (0x927C, 7, len(note), note_at))
        + note
    )
    return segment(0xE1, b"Exif\x00\x00" + tiff)


# Where EXIF offsets count from when the EXIF comes first: past SOI, the APP1
# marker and length, and "Exif\0\0".
EXIF_BASE = 12


def mpf_segment(primary: int, second: int, second_at: int) -> bytes:
    """APP2 MPF listing the primary image (``primary`` bytes long) and a large
    preview, ``second`` bytes at ``second_at`` from the MPF header (the "II")."""
    entries = struct.pack("<IIIHH", 0x20030000, primary, 0, 0, 0)
    entries += struct.pack("<IIIHH", 0x00010002, second, second_at, 0, 0)
    index = ifd(8, (0xB000, 7, 4, b"0100"), long(0xB001, 2), (0xB002, 7, 32, entries))
    return segment(0xE2, b"MPF\x00II*\x00\x08\x00\x00\x00" + index)


def with_mpf(app: bytes, second: bytes, seed: int, padding: int = 0) -> bytes:
    """SOI, the ``app`` segments, MPF, the test scene, then ``padding`` zeros
    and the MPF's ``second`` image."""
    body = jpeg_body(scene_jpeg((640, 480), seed))

    def build(primary: int) -> bytes:
        mpf_at = 2 + len(app) + 8
        mpf = mpf_segment(primary, len(second), max(0, primary + padding - mpf_at))
        return b"\xff\xd8" + app + mpf + body

    return build(len(build(0))) + b"\x00" * padding + second


def sony_jpeg(seed: int, *, preview_pointer: bool) -> bytes:
    """A Sony NEX-6 JPEG: EXIF, MPF, the image, zeros, then the MPF preview.

    With ``preview_pointer`` the maker note's tag 0x2001 points into that
    preview, 130 bytes in, as on the NEX-5N, NEX-6, A6000 and A7 among others
    (exiftool's Sony.pm): inside the preview's APP1, at Sony's 32-byte header
    and "\\x00\\xd8", which run on into the preview's own tables and scan.
    Without it there is no 0x2001, as on later models.
    """
    small = (320, 240)
    rest = jpeg_body(scene_jpeg(small, seed + 1))
    size = struct.pack("<II", *small)
    header = struct.pack("<I", len(rest) + 2) + size + b"\x00" * 8 + size + b"\x00" * 4
    # The preview's own EXIF (its size), then zeros up to Sony's header.
    exif = b"II*\x00\x08\x00\x00\x00" + ifd(8, (0x0128, 3, 1, b"\x02\x00"), long(0x8769, 38))
    exif += ifd(38, long(0xA002, small[0]), long(0xA003, small[1]))
    app1 = b"Exif\x00\x00" + exif.ljust(118, b"\x00") + header + b"\x00\xd8"
    preview = b"\xff\xd8" + segment(0xE1, app1) + rest
    assert preview.index(header) == 130

    def exif_for(target: int) -> bytes:
        def note(at: int) -> bytes:
            entries: list[IfdEntry] = [text(0xB020, "Standard")]  # CreativeStyle
            if preview_pointer:
                entries.append((0x2001, 7, len(header) + 2 + len(rest), target))
            return b"SONY DSC \x00\x00\x00" + ifd(at + 12, *entries)

        return exif_segment("SONY", "NEX-6", note)

    primary = len(with_mpf(exif_for(0), b"", seed))  # no offset changes a length
    target = primary + 2002 + 130 - EXIF_BASE  # counted from the EXIF header
    return with_mpf(exif_for(target), preview, seed, padding=2002)


def olympus_jpeg(seed: int, *, preview_at: int | None = None) -> bytes:
    """An Olympus E-M5 JPEG whose maker note points to the preview stored right
    after the image, as the OM-D cameras (and the E-1 and E-300) do, or to
    ``preview_at`` when given."""
    body = jpeg_body(scene_jpeg((640, 480), seed))
    preview = scene_jpeg((320, 240), seed + 1)

    def build(start: int) -> bytes:
        def note(at: int) -> bytes:
            # Olympus' newer maker notes count their offsets from the note's start.
            settings_at = 12 + len(ifd(12, (0x0000, 7, 4, b"0100"), long(0x2020, 0)))
            settings = ifd(
                settings_at,
                (0x0000, 7, 4, b"0100"),
                long(0x0100, 1),  # PreviewImageValid
                long(0x0101, max(0, start - EXIF_BASE - at)),  # PreviewImageStart
                long(0x0102, len(preview)),  # PreviewImageLength
            )
            main = ifd(12, (0x0000, 7, 4, b"0100"), (0x2020, 13, 1, settings_at))
            return b"OLYMPUS\x00II\x03\x00" + main + settings

        return b"\xff\xd8" + exif_segment("OLYMPUS IMAGING CORP.", "E-M5", note) + body

    return build(len(build(0)) if preview_at is None else preview_at) + preview


PENTAX_APP7 = b"PENTAX \x00II"
"""How Pentax's APP7 segment starts: its name, then the byte order."""


def pentax_jpeg(seed: int) -> bytes:
    """A Pentax K-1 Mark II JPEG: EXIF with Pentax's maker note, an APP7
    "PENTAX" segment holding the preview, then the image, as the K-3 and later
    write it. The preview's offset counts from the start of APP7's data
    (exiftool's ProcessJPEG), so it holds wherever the segment moves."""
    preview = scene_jpeg((320, 240), seed + 1)  # the camera's is 720x480

    def note(at: int) -> bytes:
        # MakerNotePentax5: "PENTAX \0", the byte order, then an IFD; its
        # offsets count from the note's start.
        version = (0x0000, 1, 4, bytes((12, 3, 2, 0)))  # PentaxVersion
        return PENTAX_APP7 + ifd(10, version, long(0x0005, 0x13240))  # PentaxModelID

    def app7(start: int) -> bytes:
        return PENTAX_APP7 + ifd(
            10,
            (0x0002, 3, 2, struct.pack("<HH", 320, 240)),  # PreviewImageSize
            long(0x0003, len(preview)),  # PreviewImageLength
            long(0x0004, start),  # PreviewImageStart
            (0x003E, 1, 4, b"\x00" * 4),  # PreviewImageBorders
        )

    # The camera fills the rest of the segment: 32 bytes of 0xFF, then 0xD8.
    pentax = segment(0xE7, app7(len(app7(0))) + preview + b"\xff" * 32 + b"\xd8")
    exif = exif_segment("RICOH IMAGING COMPANY, LTD.", "PENTAX K-1 Mark II", note)
    return b"\xff\xd8" + exif + pentax + jpeg_body(scene_jpeg((640, 480), seed))


# -- shaped like real photos that were once misjudged ----------------------------
# Stand-ins for files in a real-image corpus (drewnoakes/metadata-extractor-images)
# whose layout once made As If refuse a safe write, or find out only after
# the model's work that a write couldn't be proven safe.

XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def xmp_packet(toolkit: str, body: str = "", *, end: str = "w") -> bytes:
    """A whole XMP packet, wrapper and trailer included, stamped with ``toolkit``."""
    return (
        "<?xpacket begin='﻿' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        f"<x:xmpmeta xmlns:x='adobe:ns:meta/' x:xmptk='{toolkit}'>"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<rdf:Description rdf:about='' xmlns:dc='http://purl.org/dc/elements/1.1/'"
        " xmlns:xmp='http://ns.adobe.com/xap/1.0/'>"
        f"{body}</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end='{end}'?>"
    ).encode()


def fpxr_stub_jpeg(seed: int) -> bytes:
    """A OnePlus 8 JPEG: XMP stamped with the toolkit Google's camera apps use,
    and an empty FlashPix APP2 ("FPXR" and four zero bytes). exiftool warns
    "FPXR segment too small" every time it reads the file, not when it
    writes XMP: a write that makes it read the file again shows the warning."""
    body = "<xmp:CreatorTool>OnePlus</xmp:CreatorTool>"
    xmp = segment(0xE1, XMP_HEADER + xmp_packet("Adobe XMP Core 5.1.0-jc003", body))
    fpxr = segment(0xE2, b"FPXR\x00\x00\x00\x00")
    return b"\xff\xd8" + xmp + fpxr + jpeg_body(scene_jpeg((320, 240), seed))


def thumbnail_overrun_jpeg(seed: int, overrun: int) -> bytes:
    """A Sigma DP2 Quattro JPEG: the EXIF thumbnail's length (IFD1) counts
    ``overrun`` bytes past the end of the EXIF segment, into the APP2 segment
    after it. The DP2 Quattro counts two: the APP2's 0xFF and its marker,
    which turns into new XMP's APP1 marker. Sony's DCR-PC110 counts one."""
    thumb = scene_jpeg((48, 36), seed + 1)
    main = (text(0x010F, "SIGMA"), text(0x0110, "dp2 Quattro"))
    ifd0 = bytearray(ifd(8, *main))
    ifd1_at = 8 + len(ifd0)
    ifd0[2 + 12 * len(main) : 6 + 12 * len(main)] = struct.pack("<I", ifd1_at)  # next IFD

    def ifd1(thumb_at: int) -> bytes:
        compression = (0x0103, 3, 1, b"\x06\x00")  # JPEG
        return ifd(ifd1_at, compression, long(0x0201, thumb_at), long(0x0202, len(thumb) + overrun))

    thumb_at = ifd1_at + len(ifd1(0))
    tiff = b"II*\x00\x08\x00\x00\x00" + bytes(ifd0) + ifd1(thumb_at) + thumb
    exif = segment(0xE1, b"Exif\x00\x00" + tiff)
    app2 = segment(0xE2, b"SIGMA\x00" + bytes(range(64)))  # a vendor block exiftool skips
    return b"\xff\xd8" + exif + app2 + jpeg_body(scene_jpeg((320, 240), seed))


MICROSOFT_PHOTO = "http://ns.microsoft.com/photo/1.0/"
"""The namespace Windows (Photo Viewer, File Explorer) puts its rating and
keyword fields in. exiftool knows it without the final "/", and "fixes" it
on every write, with a [minor] warning."""


def windows_rated_jpeg(seed: int, prefix: str = "MicrosoftPhoto") -> bytes:
    """A JPEG rated and tagged in Windows Photo Viewer, like the corpus's
    "Canon EOS REBEL T2i" and "Issue 587": its own MicrosoftPhoto Rating and
    LastKeywordXMP list in a separate rdf:Description. Some apps name the
    prefix ``MicrosoftPhoto_1_`` ("Issue 80")."""
    body = (
        "<xmp:Rating>3</xmp:Rating>"
        "<dc:subject><rdf:Bag><rdf:li>piano</rdf:li></rdf:Bag></dc:subject>"
        f"</rdf:Description><rdf:Description rdf:about='' xmlns:{prefix}='{MICROSOFT_PHOTO}'>"
        f"<{prefix}:Rating>50</{prefix}:Rating>"
        f"<{prefix}:LastKeywordXMP><rdf:Bag><rdf:li>piano</rdf:li></rdf:Bag>"
        f"</{prefix}:LastKeywordXMP>"
    )
    xmp = segment(0xE1, XMP_HEADER + xmp_packet("Adobe XMP Core 5.1.0-jc003", body))
    return b"\xff\xd8" + xmp + jpeg_body(scene_jpeg((320, 240), seed))


def png_chunk(ctype: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)


def scene_png(size: tuple[int, int], seed: int, *chunks: bytes) -> bytes:
    """The test scene as a Pillow PNG, with ``chunks`` added right after IHDR."""
    buf = io.BytesIO()
    _scene(size, seed).save(buf, "PNG")
    data = buf.getvalue()
    ihdr_end = 8 + 12 + int.from_bytes(data[8:12], "big")
    return data[:ihdr_end] + b"".join(chunks) + data[ihdr_end:]


def png_xmp_chunk(packet: bytes) -> bytes:
    """An uncompressed iTXt chunk holding ``packet``, as Adobe's apps write it."""
    return png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + packet)


def bad_crc_png(seed: int) -> bytes:
    """A PNG whose image data (IDAT) has a wrong CRC, like PngSuite's
    xcsn0g01: decoders show it, exiftool refuses to write it."""
    data = bytearray(scene_png((64, 48), seed))
    at = data.index(b"IDAT") - 4
    crc_at = at + 8 + int.from_bytes(data[at : at + 4], "big")
    data[crc_at] ^= 0xFF
    return bytes(data)


def xmp_leftovers_png(seed: int, leftovers: bytes) -> bytes:
    """A PNG edited in Photoshop or Lightroom: its XMP packet is followed, after
    the trailer, by ``leftovers`` of a longer packet it was rewritten over (a
    single ">" in one corpus photo, the end of an older face region in
    another)."""
    body = "<dc:title><rdf:Alt><rdf:li xml:lang='x-default'>Pier</rdf:li></rdf:Alt></dc:title>"
    packet = xmp_packet("Adobe XMP Core 5.6-c145 79.163499, 2018/08/13-16:40:22        ", body)
    return scene_png((320, 240), seed, png_xmp_chunk(packet + leftovers))


STALE_PACKET = (
    b"li>\n  <rdf:Description mwgrs:Name='Someone' mwgrs:Type='Face'/>\n </rdf:li>\n"
    b"</rdf:Bag>\n</x:xmpmeta>\n<?xpacket end='r'?>"
)
"""The end of an older, longer packet: values (a face region's name) after the trailer."""


def _tiff(entries: Callable[[int], list[IfdEntry]], data: bytes) -> bytes:
    """A little-endian TIFF: the header, ``data`` at offset 8, then IFD0 with
    ``entries(8)`` (the offset ``data`` starts at)."""
    data += b"\x00" * (len(data) % 2)
    return b"II*\x00" + struct.pack("<I", 8 + len(data)) + data + ifd(8 + len(data), *entries(8))


def _short(tag: int, *values: int) -> IfdEntry:
    return (tag, 3, len(values), struct.pack(f"<{len(values)}H", *values))


def _longs(tag: int, *values: int) -> IfdEntry:
    return (tag, 4, len(values), struct.pack(f"<{len(values)}I", *values))


def _gray(width: int, height: int, seed: int) -> bytes:
    return bytes((seed + 7 * i + i // width) % 251 for i in range(width * height))


def jpeg_tables_tiff(seed: int) -> bytes:
    """A TIFF holding TIFF 6.0's old-style JPEG tables (tags 0x207-0x209), as
    old scanners wrote them (an HP Scanjet with Trapeze): they point into a
    tables-only JPEG stream (tags 0x201/0x202) next to the strips. exiftool
    refuses to write any TIFF with those tags, whatever its compression, so
    the image here is plain gray strips that any decoder reads."""
    width, height = 64, 48
    pixels = _gray(width, height, seed)
    dqt = segment(0xDB, b"\x00" + bytes(range(1, 65)))
    dc = segment(0xC4, b"\x00" + bytes([0, 1, 5, 1, 1, 1, 1, 1, 1] + [0] * 7) + bytes(range(12)))
    ac = segment(0xC4, b"\x10" + bytes([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D]))
    tables = b"\xff\xd8" + dqt + dc + ac + b"\xff\xd9"

    def entries(at: int) -> list[IfdEntry]:
        stream = at + len(pixels)
        q, d, a = stream + 2 + 5, stream + 2 + len(dqt) + 5, stream + 2 + len(dqt) + len(dc) + 5
        return [
            long(0x0100, width),
            long(0x0101, height),
            _short(0x0102, 8),
            _short(0x0103, 1),  # no compression
            _short(0x0106, 1),  # black is zero
            long(0x0111, at),
            _short(0x0115, 1),
            long(0x0116, height),
            long(0x0117, len(pixels)),
            long(0x0201, stream),  # JPEGInterchangeFormat: the tables-only stream
            long(0x0202, len(tables)),
            _longs(0x0207, q),  # JPEGQTables
            _longs(0x0208, d),  # JPEGDCTables
            _longs(0x0209, a),  # JPEGACTables
        ]

    return _tiff(entries, pixels + tables)


def shared_strips_tiff(seed: int) -> bytes:
    """A TIFF whose two strips are the same bytes: the top half of the image
    repeats as the bottom half. Old-style JPEG scans share bytes the same way
    (each page's JPEG block is its strips). exiftool writes each strip out on
    its own, so the file grows by the shared bytes, here past the size gate."""
    width, height = 512, 300
    half = _gray(width, height // 2, seed)

    def entries(at: int) -> list[IfdEntry]:
        return [
            long(0x0100, width),
            long(0x0101, height),
            _short(0x0102, 8),
            _short(0x0103, 1),
            _short(0x0106, 1),
            _longs(0x0111, at, at),  # both strips start at the same byte
            _short(0x0115, 1),
            long(0x0116, height // 2),
            _longs(0x0117, len(half), len(half)),
        ]

    return _tiff(entries, half)


def bigtiff(seed: int) -> bytes:
    """A small, valid BigTIFF (64-bit offsets): one uncompressed gray strip."""
    width, height = 16, 12
    pixels = _gray(width, height, seed)
    data_at, ifd_at = 16, 16 + len(pixels)
    entries = [
        (0x0100, 3, 1, width),
        (0x0101, 3, 1, height),
        (0x0102, 3, 1, 8),
        (0x0103, 3, 1, 1),
        (0x0106, 3, 1, 1),
        (0x0111, 16, 1, data_at),  # LONG8
        (0x0115, 3, 1, 1),
        (0x0116, 3, 1, height),
        (0x0117, 16, 1, len(pixels)),
    ]
    table = struct.pack("<Q", len(entries))
    table += b"".join(struct.pack("<HHQQ", *entry) for entry in entries)
    return b"II+\x00" + struct.pack("<HHQ", 8, 0, ifd_at) + pixels + table + struct.pack("<Q", 0)


def _sony_preview_jpg(out: Path) -> None:
    out.write_bytes(sony_jpeg(20, preview_pointer=True))


def _olympus_preview_jpg(out: Path) -> None:
    out.write_bytes(olympus_jpeg(22))


def _pentax_preview_jpg(out: Path) -> None:
    out.write_bytes(pentax_jpeg(24))


MAKERS: dict[str, Callable[[Path], None]] = {
    "plain.jpg": _plain_jpg,
    "exif.jpg": _exif_jpg,
    "gps.jpg": _gps_jpg,
    "rotated.jpg": _rotated_jpg,
    "progressive.jpg": _progressive_jpg,
    "motion.jpg": _motion_jpg,
    "mpo.jpg": _mpo_jpg,
    "user_xmp.jpg": _user_xmp_jpg,
    "user_iptc.jpg": _user_iptc_jpg,
    "plain.png": _plain_png,
    "alpha.png": _alpha_png,
    "lossy.webp": _lossy_webp,
    "lossless.webp": _lossless_webp,
    "plain.tif": _plain_tif,
    "truncated.jpg": _truncated_jpg,
    "corrupt.jpg": _corrupt_jpg,
    "photo.heic": _heic,
    "raw.dng": _raw_dng,
    "sony_preview.jpg": _sony_preview_jpg,
    "olympus_preview.jpg": _olympus_preview_jpg,
    "pentax_preview.jpg": _pentax_preview_jpg,
    "panorama.jpg": _panorama_jpg,
}


def make_fixtures(out_dir: Path, *, with_panorama: bool = False) -> dict[str, Path]:
    """Write every fixture into ``out_dir``. Returns name -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    for fixture in FIXTURES:
        if fixture.name == "panorama.jpg" and not with_panorama:
            continue
        path = out_dir / fixture.name
        MAKERS[fixture.name](path)
        made[fixture.name] = path
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--with-panorama", action="store_true")
    args = parser.parse_args(argv)
    made = make_fixtures(args.output_dir, with_panorama=args.with_panorama)
    for name, path in made.items():
        print(f"{path.stat().st_size:>10}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
