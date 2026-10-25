"""The integrity gates, against real exiftool writes on the generated fixtures.

``legit_write`` performs the same kind of write As If's writer does (XMP
caption, subjects, provenance, original-image hash; IPTC keywords with
compat "wide"). Every gate must pass for it on every writable fixture, and
each gate must fail when the thing it guards is tampered with afterwards.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import shutil
import struct
import time
import zlib
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import pytest
from PIL import Image, ImageCms, PngImagePlugin

from asif.config import Config
from asif.meta import structure
from asif.meta.exiftool import ExecResult, ExifTool, ExifToolError
from asif.meta.gates import (
    BIGTIFF,
    MAX_GROWTH,
    PAST_EXIF_SEGMENT,
    PREVIEW_AFTER_IMAGE,
    Fingerprint,
    GateResult,
    Snapshot,
    _allowed,
    _protected,
    all_ok,
    check_source_unchanged,
    check_written,
    fingerprint,
    snapshot,
)
from asif.meta.structure import (
    C2PA_LINK_PROBLEM,
    C2PA_PROBLEM,
    LEFTOVERS_PROBLEM,
    jpeg_eoi_offset,
)
from asif.pipeline import local
from asif.pipeline.local import caption_folder, choose_model
from asif.pipeline.states import State
from asif.pipeline.timings import TimingsLog
from tests.conftest import HAVE_EXIFTOOL
from tests.fixtures.make_fixtures import (
    PENTAX_APP7,
    STALE_PACKET,
    bad_crc_png,
    bigtiff,
    exif_segment,
    ifd,
    jpeg_tables_tiff,
    long,
    olympus_jpeg,
    pentax_jpeg,
    scene_jpeg,
    shared_strips_tiff,
    sony_jpeg,
    thumbnail_overrun_jpeg,
    with_mpf,
    xmp_leftovers_png,
)
from tests.test_structure import (
    LITERAL,
    Tiff,
    app11,
    ascii_entry,
    canon_maker_note,
    chunk,
    ext_xmp,
    jumbf,
    pillow_tiff,
    rchunk,
    std_xmp,
    undefined,
    xmp_packet,
)

Compat = Literal["xmp", "wide"]
FixtureCopy = Callable[[str], Path]
Tamper = Callable[[bytes, int], bytes]

WRITABLE = [
    "plain.jpg",
    "exif.jpg",
    "gps.jpg",
    "rotated.jpg",
    "progressive.jpg",
    "motion.jpg",
    "mpo.jpg",
    "user_xmp.jpg",
    "user_iptc.jpg",
    "plain.png",
    "alpha.png",
    "lossy.webp",
    "lossless.webp",
    "plain.tif",
    "pentax_preview.jpg",
]
GATES = ["equality", "protected", "trailer", "structure", "size"]
CAPTION = "Um cachorro na praia, ao pôr do sol"
KEYWORDS = ("cachorro", "praia", "pôr do sol")


@pytest.fixture(scope="module")
def et() -> Iterator[ExifTool]:
    if not HAVE_EXIFTOOL:
        pytest.skip("exiftool not installed")
    with ExifTool() as tool:
        yield tool


def legit_write(
    et: ExifTool,
    path: Path,
    compat: Compat,
    image_hash: str,
    *,
    keywords: tuple[str, ...] = KEYWORDS,
    caption: str = CAPTION,
) -> None:
    """Write like As If's writer: only XMP-dc, XMP-asif, XMP-et (and IPTC if
    wide, keeping a Photoshop:IPTCDigest that matched it in step).

    Like the writer, it leaves a caption alone unless it's blank or As If's
    own, and writes the original-image hash only once.
    """
    existing = et.read_tags(path, "-XMP-dc:Description*", "-XMP-asif:CaptionDigest", "-XMP-et:all")
    digest = existing.pop("XMP-asif:CaptionDigest", None)
    has_hash = existing.pop("XMP-et:OriginalImageHash", None) is not None
    existing.pop("XMP-et:OriginalImageHashType", None)
    ours = all(
        not str(text).strip() or hashlib.sha256(str(text).encode()).hexdigest() == digest
        for text in existing.values()
    )
    args = [*(f"-XMP-dc:Subject+={k}" for k in keywords)]
    if ours:
        args += [
            f"-XMP-dc:Description-x-default={caption}",
            f"-XMP-dc:Description-pt-BR={caption}",
            f"-XMP-asif:CaptionDigest={hashlib.sha256(caption.encode()).hexdigest()}",
            "-XMP-asif:CaptionLang=pt-BR",
        ]
    if not has_hash:
        args += [
            f"-XMP-et:OriginalImageHash={image_hash}",
            "-XMP-et:OriginalImageHashType=SHA256",
        ]
    args += [
        "-XMP-asif:ToolVersion=0.1.0",
        "-XMP-asif:ModelId=stub",
        "-XMP-asif:ModelSHA256=",
        "-XMP-asif:PromptVersion=caption_v1",
        "-XMP-asif:RunId=20261025-abcdef",
        "-XMP-asif:Timestamp=2026:10:25 10:00:00Z",
        *(f"-XMP-asif:AddedKeywords={k}" for k in keywords),
        "-XMP-asif:WroteOriginalImageHash=True",
    ]
    iptc: list[str] = []
    if compat == "wide":
        existing = et.read_tags(path, "-IPTC:all")
        if not existing:  # create IPTC, in UTF-8
            iptc = ["-charset", "iptc=UTF8", "-IPTC:CodedCharacterSet=UTF8"]
        elif existing.get("IPTC:CodedCharacterSet") == "UTF8":
            iptc = ["-charset", "iptc=UTF8"]
        if iptc:  # the writer leaves non-UTF-8 IPTC alone
            iptc += [f"-IPTC:Keywords+={k}" for k in keywords]
            iptc += [f"-XMP-asif:AddedIPTCKeywords={k}" for k in keywords]
            # A digest in step with the IPTC (or standing in for none) is kept in step.
            digests = et.read_tags(path, "-Photoshop:IPTCDigest", "-File:CurrentIPTCDigest")
            stored = digests.get("Photoshop:IPTCDigest")
            if stored is not None and digests.get("File:CurrentIPTCDigest") in (None, stored):
                iptc.append("-Photoshop:IPTCDigest=new")
    if not iptc:
        iptc = ["-XMP-asif:AddedIPTCKeywords="]
    et.execute("-overwrite_original", "-P", *args, *iptc, str(path))


def by_name(results: list[GateResult]) -> dict[str, GateResult]:
    return {r.name: r for r in results}


def failing(results: list[GateResult]) -> set[str]:
    return {r.name for r in results if not r.ok}


def assert_private(results: list[GateResult], *secrets: str) -> None:
    for r in results:
        for secret in (CAPTION, *KEYWORDS, *secrets):
            assert secret not in r.detail, (r.name, r.detail)


# -- legitimate writes pass ----------------------------------------------------------


@pytest.mark.parametrize("compat", ["xmp", "wide"])
@pytest.mark.parametrize("name", WRITABLE)
def test_legitimate_write_passes_every_gate(
    et: ExifTool, fixture_copy: FixtureCopy, name: str, compat: Compat
) -> None:
    path = fixture_copy(name)
    before = snapshot(et, path)
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    assert et.read_tags(path, "-XMP-asif:RunId") == {"XMP-asif:RunId": "20261025-abcdef"}
    # IPTC really is written where it can be: Latin-1 IPTC is left alone, and
    # exiftool silently skips IPTC in WebP.
    iptc_expected = compat == "wide" and name not in {
        "user_iptc.jpg",
        "lossy.webp",
        "lossless.webp",
    }
    assert ("cachorro" in str(et.read_tags(path, "-IPTC:Keywords"))) == iptc_expected

    results = check_written(et, before, path, compat=compat)
    assert [r.name for r in results] == GATES
    assert all_ok(results), [r for r in results if not r.ok]
    gates = by_name(results)
    if before.fmt == "jpeg":
        assert gates["trailer"].detail != "n/a"
    else:
        assert gates["trailer"].detail == "n/a"
    assert gates["structure"].detail == ""  # checked, TIFF included
    assert_private(results)


def test_a_second_write_also_passes(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    # As If re-running on its own output: XMP and IPTC already exist.
    path = fixture_copy("exif.jpg")
    first = snapshot(et, path)
    legit_write(et, path, "wide", first.image_hash)
    assert all_ok(check_written(et, first, path, compat="wide"))
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash, keywords=("cachorro", "areia"))
    assert all_ok(check_written(et, before, path, compat="wide"))


def test_existing_utf8_iptc_gets_keywords_in_wide_mode(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    path = fixture_copy("plain.jpg")
    et.execute(
        "-overwrite_original",
        "-charset",
        "iptc=UTF8",
        "-IPTC:CodedCharacterSet=UTF8",
        "-IPTC:Keywords=família",
        "-Photoshop:IPTCDigest=new",
        str(path),
    )
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash)
    assert et.read_tags(path, "-IPTC:Keywords")["IPTC:Keywords"][0] == "família"
    # The digest followed the IPTC, so apps keep showing the XMP.
    digests = et.read_tags(path, "-Photoshop:IPTCDigest", "-File:CurrentIPTCDigest")
    assert digests["Photoshop:IPTCDigest"] == digests["File:CurrentIPTCDigest"]
    assert digests["Photoshop:IPTCDigest"] != before.tags["Photoshop:IPTCDigest"]
    assert all_ok(check_written(et, before, path, compat="wide"))


@pytest.mark.parametrize(
    "owner",
    [
        ("-charset", "iptc=UTF8", "-IPTC:Keywords=família", "-Photoshop:IPTCDigest=new"),
        (f"-Photoshop:IPTCDigest={hashlib.md5(b'').hexdigest()}",),  # no IPTC yet
    ],
    ids=["in-step", "no-iptc"],
)
def test_iptc_written_under_a_digest_that_goes_stale_fails(
    et: ExifTool, fixture_copy: FixtureCopy, owner: tuple[str, ...]
) -> None:
    # Apps following the MWG guidelines would show the IPTC instead of the
    # owner's XMP: the gate doesn't trust the writer to have updated it.
    path = fixture_copy("plain.jpg")
    et.execute("-overwrite_original", "-XMP-dc:Subject=Lisboa", *owner, str(path))
    before = snapshot(et, path)
    keywords = ["-charset", "iptc=UTF8", "-IPTC:Keywords+=praia"]
    et.execute("-overwrite_original", *keywords, str(path))
    detail = by_name(check_written(et, before, path, compat="wide"))["protected"].detail
    assert detail == "out of step with the IPTC: Photoshop:IPTCDigest"
    et.execute("-overwrite_original", f"-Photoshop:IPTCDigest={'0' * 32}", str(path))
    detail = by_name(check_written(et, before, path, compat="wide"))["protected"].detail
    assert detail == (
        "changed: Photoshop:IPTCDigest; out of step with the IPTC: Photoshop:IPTCDigest"
    )
    et.execute("-overwrite_original", *keywords[:2], "-Photoshop:IPTCDigest=new", str(path))
    assert all_ok(results := check_written(et, before, path, compat="wide")), results


def test_iptc_changes_fail_in_xmp_mode(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("plain.jpg")
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash)  # but checked as compat "xmp"
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"protected", "structure"}
    assert "IPTC:Keywords" in by_name(results)["protected"].detail
    assert "APP13" in by_name(results)["structure"].detail


def test_png_text_description_survives_xmp_description(
    et: ExifTool, fixture_copy: FixtureCopy, tmp_path: Path
) -> None:
    # exiftool's own copy numbers count across groups: adding XMP-dc:Description
    # renames an untouched PNG:Description to PNG:Copy1:Description. The gate
    # must not mistake that for a change.
    info = PngImagePlugin.PngInfo()
    info.add_text("Description", "owner's text")
    path = tmp_path / "described.png"
    Image.new("RGB", (64, 48), (10, 120, 200)).save(path, "PNG", pnginfo=info, dpi=(300, 300))
    before = snapshot(et, path)
    assert "PNG:Description" in before.tags
    legit_write(et, path, "xmp", before.image_hash)
    assert all_ok(check_written(et, before, path, compat="xmp"))

    # ...and an ancillary chunk (pHYs, the DPI) changed afterwards is caught.
    data = bytearray(path.read_bytes())
    i = data.index(b"pHYs")
    data[i + 4 : i + 13] = struct.pack(">IIB", 2835, 2835, 1)
    data[i + 13 : i + 17] = struct.pack(">I", zlib.crc32(bytes(data[i : i + 13])) & 0xFFFFFFFF)
    path.write_bytes(bytes(data))
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"protected", "structure"}
    assert "pHYs" in by_name(results)["structure"].detail


def test_extended_xmp(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    # Standard XMP in a JPEG holds about 64 KB; the rest goes to extended XMP.
    text = "owner rights text " * 4000
    # Split by its owner already, or split by As If's write.
    for user_text, split_before in ((text[:70000], True), (text[:64400], False)):
        path = fixture_copy("plain.jpg")
        et.execute("-overwrite_original", f"-XMP-dc:Rights={user_text}", str(path))
        before = snapshot(et, path)
        assert ("XMP-xmpNote:HasExtendedXMP" in before.tags) == split_before
        legit_write(et, path, "xmp", before.image_hash)
        assert "XMP-xmpNote:HasExtendedXMP" in et.read_tags(path, "-XMP-xmpNote:all")
        assert all_ok(check_written(et, before, path, compat="xmp"))

        # The owner's text is still protected, wherever it ended up.
        et.execute("-overwrite_original", f"-XMP-dc:Rights={user_text[:-1]}!", str(path))
        detail = by_name(check_written(et, before, path, compat="xmp"))["protected"].detail
        assert detail == "changed: XMP-dc:Rights"
        path.unlink()


def _two_comments(path: Path, second: bytes) -> None:
    com = [b"\xff\xfe" + (len(t) + 2).to_bytes(2, "big") + t for t in (b"first", second)]
    data = path.read_bytes()
    path.write_bytes(data[:2] + com[0] + com[1] + data[2:])


def test_duplicate_tags_in_one_group_are_all_compared(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    path = fixture_copy("plain.jpg")
    _two_comments(path, b"second")
    before = snapshot(et, path)
    assert before.tags["File:Comment"] == "first"
    assert before.tags["File:Comment#2"] == "second"
    legit_write(et, path, "xmp", before.image_hash)
    assert all_ok(check_written(et, before, path, compat="xmp"))

    data = path.read_bytes()
    path.write_bytes(data.replace(b"second", b"SECOND", 1))
    results = check_written(et, before, path, compat="xmp")
    assert by_name(results)["protected"].detail == "changed: File:Comment#2"
    assert "COM" in by_name(results)["structure"].detail


def _tiled_tiff(path: Path) -> None:
    """A 32x32 RGB TIFF in four 16x16 tiles (Pillow can't write tiled TIFFs)."""
    tile = 16 * 16 * 3
    entries = [
        (256, 3, 1, 32),  # ImageWidth
        (257, 3, 1, 32),  # ImageLength
        (258, 3, 3, 0),  # BitsPerSample -> offset filled in below
        (259, 3, 1, 1),  # Compression: none
        (262, 3, 1, 2),  # Photometric: RGB
        (277, 3, 1, 3),  # SamplesPerPixel
        (322, 3, 1, 16),  # TileWidth
        (323, 3, 1, 16),  # TileLength
        (324, 4, 4, 0),  # TileOffsets -> offset
        (325, 4, 4, 0),  # TileByteCounts -> offset
    ]
    ifd_end = 8 + 2 + 12 * len(entries) + 4
    bps, offsets, counts = ifd_end, ifd_end + 6, ifd_end + 22
    data_start = ifd_end + 38
    out = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(entries)))
    for tag, typ, count, value in entries:
        value = {258: bps, 324: offsets, 325: counts}.get(tag, value)
        packed = (
            struct.pack("<HH", value, 0) if typ == 3 and count == 1 else struct.pack("<I", value)
        )
        out += struct.pack("<HHI", tag, typ, count) + packed
    out += struct.pack("<I", 0) + struct.pack("<HHH", 8, 8, 8)
    out += struct.pack("<4I", *(data_start + i * tile for i in range(4)))
    out += struct.pack("<4I", *([tile] * 4))
    for i in range(4):
        out += bytes((i * 50 + j) % 256 for j in range(tile))
    path.write_bytes(bytes(out))


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_tiled_tiff(et: ExifTool, tmp_path: Path, compat: Compat) -> None:
    path = tmp_path / "tiled.tif"
    _tiled_tiff(path)
    before = snapshot(et, path)
    assert before.fmt == "tiff" and before.data is None
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(check_written(et, before, path, compat=compat))


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_tiff_with_exif_gps_icc_xmp_and_iptc(et: ExifTool, tmp_path: Path, compat: Compat) -> None:
    # exiftool rewrites a TIFF as a whole: nothing of the owner's may move but the strips.
    path = tmp_path / "scan.tif"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (320, 240), (10, 120, 200)).save(
        path, "TIFF", icc_profile=icc, compression="tiff_lzw"
    )
    et.execute(
        "-overwrite_original",
        "-EXIF:DateTimeOriginal=2019:07:14 10:20:30",
        "-GPSLatitude=22.951916",
        "-GPSLatitudeRef=S",
        "-IFD0:Orientation=6",
        "-n",
        "-XMP-dc:Creator=Owner",
        "-IPTC:Keywords=old",  # no CodedCharacterSet: As If leaves this IPTC alone
        str(path),
    )
    before = snapshot(et, path)
    assert {"GPS:GPSLatitude", "ICC_Profile:ICC_Profile", "IPTC:Keywords"} <= before.tags.keys()
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(check_written(et, before, path, compat=compat))

    et.execute("-overwrite_original", "-P", "-GPSLatitude=22.9519161", str(path))
    assert by_name(check_written(et, before, path, compat=compat))["protected"].detail == (
        "changed: GPS:GPSLatitude"
    )


@pytest.mark.slow
def test_panorama_passes_quickly(et: ExifTool, panorama: Path, tmp_path: Path) -> None:
    path = tmp_path / "panorama.jpg"
    path.write_bytes(panorama.read_bytes())
    started = time.monotonic()
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash)
    results = check_written(et, before, path, compat="wide")
    elapsed = time.monotonic() - started
    assert all_ok(results), results
    assert elapsed < 60, f"snapshot + write + gates took {elapsed:.1f} s"


# -- snapshot and the dump ----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("exif.jpg", "jpeg"), ("alpha.png", "png"), ("lossy.webp", "webp"), ("plain.tif", "tiff")],
)
def test_snapshot_records_the_file(et: ExifTool, fixtures_dir: Path, name: str, fmt: str) -> None:
    path = fixtures_dir / name
    snap = snapshot(et, path)
    raw = path.read_bytes()
    assert snap.fmt == fmt
    assert snap.size == len(raw)
    assert snap.sha256 == hashlib.sha256(raw).hexdigest()
    assert snap.image_hash == et.image_data_hash(path)
    assert snap.data == (None if fmt == "tiff" else raw)
    assert snap.tags["File:FileType"]
    assert not [k for k in snap.tags if k.split(":")[0] in {"System", "ExifTool", "Composite"}]


def test_snapshot_dump_is_raw_and_binary(et: ExifTool, fixtures_dir: Path) -> None:
    snap = snapshot(et, fixtures_dir / "exif.jpg")
    assert snap.tags["ExifIFD:DateTimeOriginal"] == "2019:07:14 10:20:30"
    assert snap.tags["IFD1:ThumbnailImage"].startswith("base64:")
    assert snap.tags["ICC_Profile:ICC_Profile"].startswith("base64:")  # the raw block
    gps = snapshot(et, fixtures_dir / "gps.jpg")
    assert gps.tags["GPS:GPSLatitude"] == pytest.approx(22.951916)  # not "22 deg 57' 6.90\""


def test_snapshot_repr_leaks_nothing(et: ExifTool, fixtures_dir: Path) -> None:
    text = repr(snapshot(et, fixtures_dir / "gps.jpg"))
    assert "22.95" not in text and "GPS" not in text and "\\xff" not in text
    text = repr(snapshot(et, fixtures_dir / "user_xmp.jpg"))
    assert "Grandma" not in text and "family" not in text  # the XMP's own values


def test_snapshot_of_corrupt_file_raises(et: ExifTool, fixtures_dir: Path) -> None:
    with pytest.raises(ExifToolError):
        snapshot(et, fixtures_dir / "corrupt.jpg")


def test_allow_list_never_covers_protected_tags() -> None:
    protected = [
        "ExifIFD:DateTimeOriginal",
        "ExifIFD:CreateDate",
        "ExifIFD:OffsetTimeOriginal",
        "IFD0:ModifyDate",
        "GPS:GPSLatitude",
        "GPS:GPSLongitude",
        "GPS:GPSAltitude",
        "GPS:GPSLatitudeRef",
        "IFD0:Orientation",
        "ICC_Profile:ICC_Profile",
        "ICC-header:ProfileDateTime",
        "ExifIFD:MakerNoteCanon",
        "ExifIFD:MakerNoteUnknown",
        "IFD1:ThumbnailImage",
        "IFD1:ThumbnailLength",
        "IFD0:PreviewImage",
        "Photoshop:PhotoshopThumbnail",
        "Photoshop:IPTCDigest",
        "MPImage2:MPImage2",
        "MPImage2:MPImageLength",
        "MPF0:NumberOfImages",
        "XMP-exif:DateTimeOriginal",
        "XMP-exif:GPSLatitude",
        "XMP-xmp:CreateDate",
        "XMP-tiff:Orientation",
        "XMP-photoshop:DateCreated",
        "XMP-dc:Creator",
        "XMP-dc:Title",
        "XMP-dc:Rights",
        "XMP-GCamera:MicroVideoOffset",
        "IPTC:DateCreated",
        "IPTC:Caption-Abstract",
        "IFD0:ImageDescription",
        "PNG:Description",
        "File:Comment",
        "File:ImageWidth",
        "IFD0:StripByteCounts",
        "IFD1:StripByteCounts",
        "IFD1:TileByteCounts",
        "SubIFD:StripByteCounts",
        "SubIFD:ThumbnailOffset",
        "IFD0:PreviewImageStart",
        "IFD0:Exif_0xc5ab",
    ]
    for key in protected:
        for variant in (key, f"{key}#2"):
            for fmt in ("jpeg", "png", "webp", "tiff", "other"):
                for compat in ("xmp", "wide"):
                    assert not _allowed(variant, fmt, compat), (variant, fmt, compat)  # type: ignore[arg-type]
    # A WebP's size is reported twice once it has a VP8X header; the first copy counts.
    assert not _allowed("RIFF:ImageWidth", "webp", "xmp")
    assert not _allowed("RIFF:ImageHeight", "webp", "wide")


def test_tiff_data_of_every_page_may_move() -> None:
    # exiftool rebuilds a TIFF; the data each IFD points at is checked elsewhere.
    moved = [
        "IFD0:StripOffsets",
        "IFD1:StripOffsets",
        "IFD2:TileOffsets",
        "IFD1:ThumbnailOffset",
        "SubIFD:StripOffsets",
        "SubIFD2:TileOffsets",
        # exiftool names a second page's SubIFD, or one inside another, plain "SubIFD" again.
        "SubIFD:StripOffsets#2",
        # JPEG images behind tag 0x201 outside IFD1, covered by the image-data hash.
        "IFD0:OtherImageStart",
        "SubIFD1:OtherImageStart",
        "SubIFD:JpgFromRawStart",
        "IFD2:JpgFromRawStart",
    ]
    for key in moved:
        assert _allowed(key, "tiff", "xmp")
        for fmt in ("jpeg", "png", "webp", "other"):  # in a JPEG's EXIF nothing may move
            assert not _allowed(key, fmt, "wide"), (key, fmt)
    for key in ("IFD1:StripOffsetsX", "XIFD1:StripOffsets", "IFD:StripOffsets"):
        assert not _allowed(key, "tiff", "xmp"), key


class FakeExifTool:
    """Stands in for exiftool to feed the dump parser odd output."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def image_data_hash(self, path: Path) -> str:
        return "0" * 64

    def execute(self, *args: str) -> ExecResult:
        return ExecResult(stdout=self.stdout, stderr="")


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("", id="nothing"),
        pytest.param("not json", id="not-json"),
        pytest.param("[]", id="no-rows"),
        pytest.param('[{"SourceFile": "a"}, {"SourceFile": "b"}]', id="two-rows"),
        pytest.param('{"SourceFile": "a"}', id="not-a-list"),
        pytest.param('[{"SourceFile": "a", "ExifTool:Error": "File format error"}]', id="error"),
    ],
)
def test_unusable_dumps_raise(tmp_path: Path, stdout: str) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(ExifToolError):
        snapshot(FakeExifTool(stdout), path)  # type: ignore[arg-type]


def test_dump_keeps_repeated_keys_and_renumbers_copies(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"\xff\xd8\xff\xd9")
    stdout = """[{
      "SourceFile": "photo.jpg",
      "System:FileSize": 4,
      "ExifTool:Warning": "something",
      "Composite:ImageSize": "1 1",
      "File:Comment": "a",
      "File:Comment": "b",
      "XMP-dc:Description": "caption",
      "PNG:Copy1:Description": "owner's",
      "File:Copy2:Comment": "c"
    }]"""
    snap = snapshot(FakeExifTool(stdout), path)  # type: ignore[arg-type]
    assert snap.tags == {
        "File:Comment": "a",
        "File:Comment#2": "b",
        "File:Comment#3": "c",
        "XMP-dc:Description": "caption",
        "PNG:Description": "owner's",
    }


# -- protected ------------------------------------------------------------------------


def _icc_with_one_reserved_byte_changed(path: Path, out: Path) -> Path:
    icc = bytearray(Image.open(path).info["icc_profile"])
    icc[120] ^= 0x01  # header bytes 100-127 are reserved: no decoded tag shows them
    out.write_bytes(bytes(icc))
    return out


def _other_thumbnail(out: Path) -> Path:
    buf = io.BytesIO()
    Image.new("RGB", (160, 120), (200, 30, 30)).save(buf, "JPEG")
    out.write_bytes(buf.getvalue())
    return out


@pytest.mark.parametrize(
    ("name", "edit", "expected", "secret"),
    [
        (
            "exif.jpg",
            ["-ExifIFD:DateTimeOriginal=2019:07:14 10:20:31"],
            "changed: ExifIFD:DateTimeOriginal",
            "2019:07:14",
        ),
        # 1e-7 degrees (about 1 cm): prints exactly like the original without -n.
        ("gps.jpg", ["-GPSLatitude=22.9519161"], "changed: GPS:GPSLatitude", "22.95"),
        ("gps.jpg", ["-GPS:all="], "removed: GPS:GPSAltitude, GPS:GPSAltitudeRef", "710"),
        ("rotated.jpg", ["-IFD0:Orientation=1", "-n"], "changed: IFD0:Orientation", ""),
        ("plain.jpg", ["-ExifIFD:DateTimeOriginal=2001:01:01 00:00:00"], "added: ", "2001"),
        ("user_xmp.jpg", ["-XMP-dc:Creator=Someone Else"], "changed: XMP-dc:Creator", "Owner"),
        ("motion.jpg", ["-XMP-GCamera:MicroVideoOffset=1"], "changed: XMP-GCamera", ""),
        ("plain.png", ["-PNG:Comment=hello"], "added: PNG:Comment", "hello"),
        ("lossy.webp", ["-EXIF:Artist=x"], "added: ", ""),
        ("plain.tif", ["-IFD0:Artist=x"], "added: IFD0:Artist", ""),
    ],
)
def test_protected_tag_edits_fail(
    et: ExifTool,
    fixture_copy: FixtureCopy,
    name: str,
    edit: list[str],
    expected: str,
    secret: str,
) -> None:
    path = fixture_copy(name)
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash)
    et.execute("-overwrite_original", "-P", *edit, str(path))
    results = check_written(et, before, path, compat="wide")
    protected = by_name(results)["protected"]
    assert not protected.ok
    assert protected.detail.startswith(expected), protected.detail
    assert_private(results, *([secret] if secret else []))


def test_icc_profile_compared_byte_for_byte(
    et: ExifTool, fixture_copy: FixtureCopy, tmp_path: Path
) -> None:
    path = fixture_copy("exif.jpg")
    icc = _icc_with_one_reserved_byte_changed(path, tmp_path / "other.icc")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    et.execute("-overwrite_original", "-P", f"-ICC_Profile<={icc}", str(path))
    results = check_written(et, before, path, compat="xmp")
    assert by_name(results)["protected"].detail == "changed: ICC_Profile:ICC_Profile"
    assert "APP2" in by_name(results)["structure"].detail


def test_thumbnail_change_fails(et: ExifTool, fixture_copy: FixtureCopy, tmp_path: Path) -> None:
    path = fixture_copy("exif.jpg")
    thumb = _other_thumbnail(tmp_path / "thumb.jpg")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    et.execute("-overwrite_original", "-P", f"-ThumbnailImage<={thumb}", str(path))
    results = check_written(et, before, path, compat="xmp")
    assert "IFD1:ThumbnailImage" in by_name(results)["protected"].detail
    assert failing(results) == {"protected", "structure"}


def test_protected_detail_is_capped(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("exif.jpg")
    before = snapshot(et, path)
    et.execute("-overwrite_original", "-ICC_Profile:all=", str(path))
    detail = by_name(check_written(et, before, path, compat="xmp"))["protected"].detail
    assert detail.startswith("removed: ") and "more)" in detail


# -- equality -------------------------------------------------------------------------


def test_altered_jpeg_pixels_fail_equality(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("exif.jpg")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    data = bytearray(path.read_bytes())
    i = jpeg_eoi_offset(bytes(data)) - 50
    while data[i] in (0xFE, 0xFF) or data[i - 1] == 0xFF:
        i -= 1
    data[i] ^= 0x01
    path.write_bytes(bytes(data))
    results = check_written(et, before, path, compat="xmp")
    assert by_name(results)["equality"].detail == "the image data changed"
    assert failing(results) == {"equality", "structure"}


def test_altered_png_pixels_fail_equality(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("plain.png")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    data = bytearray(path.read_bytes())
    i = data.index(b"IDAT")
    length = int.from_bytes(data[i - 4 : i], "big")
    data[i + 100] ^= 0x01
    crc = zlib.crc32(bytes(data[i : i + 4 + length])) & 0xFFFFFFFF
    data[i + 4 + length : i + 8 + length] = crc.to_bytes(4, "big")
    path.write_bytes(bytes(data))
    assert {"equality", "structure"} <= failing(check_written(et, before, path, compat="xmp"))


# -- trailer ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "tamper", "detail"),
    [
        (
            "motion.jpg",
            lambda d, eoi: d[:-1] + bytes([d[-1] ^ 1]),
            "(4124 bytes before, 4124 after)",
        ),
        ("motion.jpg", lambda d, eoi: d[:eoi], "(4124 bytes before, 0 after)"),
        ("motion.jpg", lambda d, eoi: d[:-100], "(4124 bytes before, 4024 after)"),
        ("motion.jpg", lambda d, eoi: d + b"appended", "(4124 bytes before, 4132 after)"),
        (
            "mpo.jpg",
            lambda d, eoi: d[: eoi + 200] + bytes([d[eoi + 200] ^ 1]) + d[eoi + 201 :],
            "bytes before",
        ),
        ("plain.jpg", lambda d, eoi: d + b"\x00" * 16, "(0 bytes before, 16 after)"),
    ],
)
def test_trailer_tampering_fails(
    et: ExifTool, fixture_copy: FixtureCopy, name: str, tamper: Tamper, detail: str
) -> None:
    path = fixture_copy(name)
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    data = path.read_bytes()
    path.write_bytes(tamper(data, jpeg_eoi_offset(data)))
    results = check_written(et, before, path, compat="xmp")
    gates = by_name(results)
    assert gates["equality"].ok  # the trailer isn't part of ImageDataHash...
    assert not gates["trailer"].ok  # ...which is why this gate exists
    assert gates["trailer"].detail.startswith("the trailer changed")
    assert detail in gates["trailer"].detail
    assert not gates["structure"].ok


# -- structure ---------------------------------------------------------------------------


def _insert_after_soi(data: bytes, segment: bytes) -> bytes:
    return data[:2] + segment + data[2:]


def test_inserted_app_segment_fails_structure(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("exif.jpg")
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash)
    app15 = b"\xff\xef\x00\x0bvendor\x00\x01\x02"
    path.write_bytes(_insert_after_soi(path.read_bytes(), app15))
    results = check_written(et, before, path, compat="wide")
    assert "structure" in failing(results)
    assert "APP15" in by_name(results)["structure"].detail
    assert by_name(results)["equality"].ok and by_name(results)["trailer"].ok


def test_reordered_segments_fail_structure(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("plain.jpg")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    data = path.read_bytes()
    first = data.index(b"\xff\xdb")
    second = data.index(b"\xff\xdb", first + 4)
    n1 = int.from_bytes(data[first + 2 : first + 4], "big") + 2
    n2 = int.from_bytes(data[second + 2 : second + 4], "big") + 2
    assert first + n1 == second  # Pillow writes the two tables back to back
    swapped = data[:first] + data[second : second + n2] + data[first:second] + data[second + n2 :]
    path.write_bytes(swapped)
    results = check_written(et, before, path, compat="xmp")
    assert "structure" in failing(results)
    assert "DQT" in by_name(results)["structure"].detail


def test_rewritten_mpf_index_fails_structure(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("mpo.jpg")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    data = bytearray(path.read_bytes())
    base = data.index(b"MPF\x00II*\x00") + 4  # MPF offsets count from this TIFF header
    ifd = base + struct.unpack("<I", data[base + 4 : base + 8])[0]
    for k in range(struct.unpack("<H", data[ifd : ifd + 2])[0]):
        tag, _typ, _count, value = struct.unpack(
            "<HHII", data[ifd + 2 + 12 * k : ifd + 14 + 12 * k]
        )
        if tag == 0xB002:  # MPEntry: 16 bytes per image; the start offset is at +8
            start = base + value + 16 + 8
            old = struct.unpack("<I", data[start : start + 4])[0]
            data[start : start + 4] = struct.pack("<I", old + 2)
            break
    else:
        pytest.fail("no MPEntry tag")
    path.write_bytes(bytes(data))
    results = check_written(et, before, path, compat="xmp")
    assert "structure" in failing(results)
    assert "APP2" in by_name(results)["structure"].detail
    assert by_name(results)["trailer"].ok  # the images themselves are untouched


def _riff_insert(data: bytes, at: int, chunk: bytes) -> bytes:
    out = data[:at] + chunk + data[at:]
    return out[:4] + struct.pack("<I", len(out) - 8) + out[8:]


def test_webp_non_xmp_chunk_changes_fail_structure(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("lossy.webp")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    written_data = path.read_bytes()
    assert written_data[12:16] == b"VP8X"

    flagged = bytearray(written_data)
    flagged[20] |= 0x20  # claim an ICC profile
    path.write_bytes(bytes(flagged))
    results = check_written(et, before, path, compat="xmp")
    assert by_name(results)["structure"].detail == (
        "WebP gained a VP8X header with flags other than XMP/alpha"
    )

    path.write_bytes(_riff_insert(written_data, 30, b"ABCD\x02\x00\x00\x00hi"))
    results = check_written(et, before, path, compat="xmp")
    assert "WebP chunk" in by_name(results)["structure"].detail
    assert by_name(results)["equality"].ok


def test_webp_bytes_after_riff_fail_structure(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("lossless.webp")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    path.write_bytes(path.read_bytes() + b"tail")
    results = check_written(et, before, path, compat="xmp")
    assert "after the RIFF data" in by_name(results)["structure"].detail


def test_png_chunk_added_fails_structure(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("alpha.png")
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash)
    data = path.read_bytes()
    text = b"tEXt" + b"Comment\x00added later"
    chunk = struct.pack(">I", len(text) - 4) + text + struct.pack(">I", zlib.crc32(text))
    iend = data.rindex(b"IEND") - 4
    path.write_bytes(data[:iend] + chunk + data[iend:])
    results = check_written(et, before, path, compat="wide")
    assert {"protected", "structure"} <= failing(results)
    assert "tEXt" in by_name(results)["structure"].detail


# -- size -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["plain.jpg", "plain.png"])
def test_oversized_xmp_fails_size_only(et: ExifTool, fixture_copy: FixtureCopy, name: str) -> None:
    path = fixture_copy(name)
    before = snapshot(et, path)
    many = tuple(f"k{i:03d}-" + "x" * 990 for i in range(80))
    legit_write(et, path, "xmp", before.image_hash, keywords=many)
    growth = path.stat().st_size - before.size
    assert growth >= MAX_GROWTH
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"size"}  # extended XMP is still just XMP
    assert by_name(results)["size"].detail == f"grew by {growth} bytes (limit {MAX_GROWTH - 1})"


# -- files that can't be proven --------------------------------------------------------------


def test_truncated_jpeg_fails_trailer_and_structure(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    path = fixture_copy("truncated.jpg")
    before = snapshot(et, path)  # exiftool can still hash what's there
    assert before.fmt == "jpeg"
    assert before.unprovable is not None
    assert before.unprovable.startswith("its structure can't be checked: JPEG ")
    with contextlib.suppress(ExifToolError):  # exiftool may refuse; the gates must fail anyway
        legit_write(et, path, "xmp", before.image_hash)
    results = check_written(et, before, path, compat="xmp")
    gates = by_name(results)
    assert not gates["trailer"].ok and "can't prove" in gates["trailer"].detail
    assert not gates["structure"].ok and "can't prove" in gates["structure"].detail
    assert not all_ok(results)


def test_written_file_turned_to_garbage(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("exif.jpg")
    garbage = fixture_copy("corrupt.jpg")
    before = snapshot(et, path)
    path.write_bytes(garbage.read_bytes())
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"equality", "protected", "trailer", "structure"}
    gates = by_name(results)
    assert gates["structure"].detail == "file format changed (JPEG before, unknown after)"
    assert "can't prove" in gates["trailer"].detail


def test_snapshot_claiming_jpeg_for_garbage(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("plain.jpg")
    garbage = fixture_copy("corrupt.jpg").read_bytes()
    before = dataclasses.replace(snapshot(et, path), data=garbage)
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"trailer", "structure"}
    path.write_bytes(garbage)
    results = check_written(et, before, path, compat="xmp")
    assert by_name(results)["structure"].detail == (
        "can't prove the structure is intact: unknown format"
    )


def test_missing_written_file(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("motion.jpg")
    before = snapshot(et, path)
    path.unlink()
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == set(GATES)


def test_snapshot_without_bytes_cant_prove_anything(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    path = fixture_copy("plain.png")
    before = dataclasses.replace(snapshot(et, path), data=None)
    assert failing(check_written(et, before, path, compat="xmp")) == {"structure"}


# -- remote-unchanged --------------------------------------------------------------------------


def test_fingerprint_matches_the_file(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"pixels" * 1000)
    st = path.stat()
    assert fingerprint(path) == Fingerprint(
        st.st_size, st.st_mtime_ns, hashlib.sha256(b"pixels" * 1000).hexdigest()
    )


def test_source_unchanged_passes(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"pixels")
    result = check_source_unchanged(fingerprint(path), fingerprint(path))
    assert result == GateResult("remote-unchanged", True)


def test_same_size_and_mtime_but_new_content_is_caught(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"pixels")
    then = fingerprint(path)
    path.write_bytes(b"PIXELS")
    os.utime(path, ns=(then.mtime_ns, then.mtime_ns))
    result = check_source_unchanged(then, fingerprint(path))
    assert not result.ok
    assert result.detail == "changed since it was read: content"


@pytest.mark.parametrize(
    ("now", "detail"),
    [
        (Fingerprint(11, 5, "a" * 64), "size"),
        (Fingerprint(10, 6, "a" * 64), "modification time"),
        (Fingerprint(10, 5, "b" * 64), "content"),
        (Fingerprint(0, 0, "changed"), "size, modification time, content"),
    ],
)
def test_source_changes_fail(now: Fingerprint, detail: str) -> None:
    result = check_source_unchanged(Fingerprint(10, 5, "a" * 64), now)
    assert result == GateResult("remote-unchanged", False, f"changed since it was read: {detail}")


# -- all_ok -------------------------------------------------------------------------------------


def test_all_ok() -> None:
    good, bad = GateResult("a", True), GateResult("b", False, "why")
    assert all_ok([good, good])
    assert all_ok(r for r in [good])
    assert not all_ok([good, bad])
    assert not all_ok([])  # nothing checked proves nothing


# -- TIFF: every page, preview and thumbnail --------------------------------------------------


def _offset(et: ExifTool, path: Path, tag: str) -> int:
    """Where the (first) strip or thumbnail of ``tag`` starts in the file."""
    return int(str(et.read_tags(path, "-n", f"-{tag}")[tag]).split()[0])


def _flip(path: Path, at: int) -> None:
    data = bytearray(path.read_bytes())
    data[at] ^= 0x01
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("compat", ["xmp", "wide"])
@pytest.mark.parametrize(
    ("pages", "options"), [(2, {}), (3, {"compression": "tiff_lzw"})], ids=["2-pages", "3-lzw"]
)
def test_multi_page_tiff_passes_and_every_page_stays_checked(
    et: ExifTool, tmp_path: Path, compat: Compat, pages: int, options: dict[str, Any]
) -> None:
    # exiftool moves every page's strips; the pages themselves must not change.
    path = tmp_path / "scan.tif"
    path.write_bytes(pillow_tiff(pages, **options))
    before = snapshot(et, path)
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    results = check_written(et, before, path, compat=compat)
    assert all_ok(results), results

    _flip(path, _offset(et, path, f"IFD{pages - 1}:StripOffsets") + 3)  # the last page
    assert failing(check_written(et, before, path, compat=compat)) == {"equality"}


def _thumbnail() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _tiff_with_preview_and_thumbnail() -> bytes:
    t = Tiff()
    preview = t.page(8, 8, t.long(254, 1), seed=40)  # a reduced-size image, in a SubIFD
    thumb = _thumbnail()
    ifd1 = t.ifd(t.long(254, 1), t.long(0x0201, t.blob(thumb)), t.long(0x0202, len(thumb)))
    sub_ifds = (0x014A, 13, 1, struct.pack("<I", preview))
    return t.done(t.page(32, 24, sub_ifds, next_ifd=ifd1))


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_tiff_preview_and_thumbnail_move_but_stay_checked(
    et: ExifTool, tmp_path: Path, compat: Compat
) -> None:
    path = tmp_path / "preview.tif"
    path.write_bytes(_tiff_with_preview_and_thumbnail())
    before = snapshot(et, path)
    assert {"SubIFD:StripOffsets", "IFD1:ThumbnailOffset", "IFD1:ThumbnailImage"} <= set(
        before.tags
    )
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat=compat)), results

    written = path.read_bytes()
    _flip(path, _offset(et, path, "SubIFD:StripOffsets") + 5)
    # (exiftool also rebuilds the preview as SubIFD:ThumbnailTIFF, so protected sees it too.)
    assert "equality" in failing(check_written(et, before, path, compat=compat))
    path.write_bytes(written)
    _flip(path, _offset(et, path, "IFD1:ThumbnailOffset") + 30)
    protected = by_name(check_written(et, before, path, compat=compat))["protected"]
    assert "IFD1:ThumbnailImage" in protected.detail


def _subifd(t: Tiff, at: int) -> tuple[int, int, int, bytes]:
    return (0x014A, 13, 1, struct.pack("<I", at))


def _subifd_on_each_page(t: Tiff) -> int:
    page2 = t.page(16, 12, _subifd(t, t.page(8, 8, t.long(254, 1), seed=50)), seed=60)
    return t.page(32, 24, _subifd(t, t.page(8, 8, t.long(254, 1), seed=40)), next_ifd=page2)


def _nested_subifds(t: Tiff) -> int:
    inner = t.page(4, 4, t.long(254, 1), seed=50)
    return t.page(32, 24, _subifd(t, t.page(8, 8, t.long(254, 1), _subifd(t, inner), seed=40)))


def _jpeg_in(t: Tiff, where: str) -> int:
    """Tag 0x201's JPEG in IFD0, a SubIFD or IFD2 (exiftool's OtherImage, JpgFromRaw)."""
    thumb = _thumbnail()
    entries = (t.long(0x0201, t.blob(thumb)), t.long(0x0202, len(thumb)))
    if where == "IFD0":
        return t.page(32, 24, *entries)
    if where == "SubIFD":
        return t.page(32, 24, _subifd(t, t.ifd(t.long(254, 1), *entries)))
    return t.page(32, 24, next_ifd=t.page(16, 12, seed=60, next_ifd=t.ifd(*entries)))


@pytest.mark.parametrize("compat", ["xmp", "wide"])
@pytest.mark.parametrize(
    ("build", "moved"),
    [
        pytest.param(_subifd_on_each_page, "SubIFD:StripOffsets#2", id="subifd-on-each-page"),
        pytest.param(_nested_subifds, "SubIFD:StripOffsets#2", id="nested-subifds"),
        pytest.param(lambda t: _jpeg_in(t, "IFD0"), "IFD0:OtherImageStart", id="ifd0-jpeg"),
        pytest.param(lambda t: _jpeg_in(t, "SubIFD"), "SubIFD:JpgFromRawStart", id="subifd-jpeg"),
        pytest.param(lambda t: _jpeg_in(t, "IFD2"), "IFD2:JpgFromRawStart", id="ifd2-jpeg"),
    ],
)
def test_every_moved_tiff_image_passes_and_stays_checked(
    et: ExifTool, tmp_path: Path, compat: Compat, build: Callable[[Tiff], int], moved: str
) -> None:
    path = tmp_path / "pyramid.tif"
    t = Tiff()
    path.write_bytes(t.done(build(t)))
    before = snapshot(et, path)
    assert moved in before.tags
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat=compat)), results

    # The image moved, but what it holds is still checked.
    at = int(str(snapshot(et, path).tags[moved]).split()[0])
    _flip(path, at + 5)
    assert "equality" in failing(check_written(et, before, path, compat=compat))


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_tiled_jpeg_tiff_is_still_hashed_and_passes(
    et: ExifTool, tmp_path: Path, compat: Compat
) -> None:
    # Only JPEG *strips* hide the image from exiftool's hash; tiles don't.
    path = tmp_path / "tiled.tif"
    _tiled_tiff(path)
    data = bytearray(path.read_bytes())
    entry = data.index(struct.pack("<HHI", 259, 3, 1))
    data[entry + 8 : entry + 10] = struct.pack("<H", 7)  # Compression: JPEG
    path.write_bytes(bytes(data))
    before = snapshot(et, path)
    assert before.tags["IFD0:Compression"] == 7 and before.image_hash
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat=compat)), results


def test_unknown_tiff_tags_are_compared(et: ExifTool, tmp_path: Path) -> None:
    path = tmp_path / "scanner.tif"
    settings = b"PRIVATE-SCANNER-SETTINGS" * 4
    t = Tiff()
    path.write_bytes(t.done(t.page(16, 8, undefined(0xC5AD, settings))))
    before = snapshot(et, path)
    assert "IFD0:Exif_0xc5ad" in before.tags
    legit_write(et, path, "xmp", before.image_hash)
    assert all_ok(check_written(et, before, path, compat="xmp"))

    data = path.read_bytes()
    assert data.count(settings) == 1
    path.write_bytes(data.replace(settings, settings.lower()))
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"protected"}
    assert by_name(results)["protected"].detail == "changed: IFD0:Exif_0xc5ad"


# -- files no write to can be proven safe ------------------------------------------------------


def test_tiff_data_nothing_points_to_is_refused(et: ExifTool, tmp_path: Path) -> None:
    # A scanner's private tag holds the offset of its own calibration block.
    path = tmp_path / "scanner.tif"
    vendor = b"VENDOR-SCAN-CALIBRATION:" + b"\x5a" * 512
    t = Tiff()
    at = t.blob(vendor)
    path.write_bytes(t.done(t.page(32, 16, t.long(0xC5AB, at), t.long(0xC5AC, len(vendor)))))
    before = snapshot(et, path)
    assert before.unprovable == (
        "the TIFF holds 536 bytes nothing points to, which exiftool would drop"
    )

    legit_write(et, path, "xmp", before.image_hash)
    assert vendor not in path.read_bytes()  # why: exiftool copied the offset, not the block
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"structure"}  # nothing else can see it
    assert by_name(results)["structure"].detail == (
        f"can't prove a write is safe: {before.unprovable}"
    )


def _canon_exif(*, image: bool) -> tuple[bytes, bytes]:
    """EXIF (a TIFF structure, with an image when ``image``) holding a Canon
    maker note with out-of-line values and trailing bytes; also the note."""
    t = Tiff()
    at = t.here()
    note = canon_maker_note(at, trailing=b"VENDOR-TRAILING-BYTES")
    t.blob(note)
    exif = t.ifd((0x927C, 7, len(note), at))
    tags = (ascii_entry(0x010F, "Canon"), ascii_entry(0x0110, "Canon EOS"), t.long(0x8769, exif))
    return t.done(t.page(32, 16, *tags) if image else t.ifd(*tags)), note


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_jpeg_maker_note_passes_and_is_compared_byte_for_byte(
    et: ExifTool, fixture_copy: FixtureCopy, compat: Compat
) -> None:
    path = fixture_copy("plain.jpg")
    exif, note = _canon_exif(image=False)
    data = path.read_bytes()
    app1 = b"\xff\xe1" + (len(exif) + 8).to_bytes(2, "big") + b"Exif\x00\x00" + exif
    path.write_bytes(data[:2] + app1 + data[2:])
    before = snapshot(et, path)
    assert "ExifIFD:MakerNoteCanon" in before.tags and "Canon:CanonFirmwareVersion" in before.tags
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat=compat)), results

    written = path.read_bytes()
    start = written.index(note)
    # Entry count, an offset inside an entry, an out-of-line value, the
    # next-IFD link and a trailing byte: exiftool decodes only some of them.
    for at in (0, 10, 42, 38, len(note) - 1):
        _flip(path, start + at)
        results = check_written(et, before, path, compat=compat)
        assert "APP1" in by_name(results)["structure"].detail, at
        path.write_bytes(written)


def test_tiff_with_maker_notes_is_refused(et: ExifTool, tmp_path: Path) -> None:
    path = tmp_path / "dpp_export.tif"
    path.write_bytes(_canon_exif(image=True)[0])
    before = snapshot(et, path)
    assert before.unprovable == "it's a TIFF with maker notes, which exiftool would rewrite"

    legit_write(et, path, "xmp", before.image_hash)
    # Why: exiftool moved the note, rewrote its offsets and dropped its trailing bytes.
    assert b"VENDOR-TRAILING-BYTES" not in path.read_bytes()
    detail = by_name(check_written(et, before, path, compat="xmp"))["structure"].detail
    assert detail == f"can't prove a write is safe: {before.unprovable}"


def _with_xmp(fmt: str, jpeg: Path, packet: bytes) -> Path:
    """``jpeg`` with ``packet`` as its XMP, or a TIFF next to it holding the packet."""
    if fmt == "jpeg":
        data = jpeg.read_bytes()
        jpeg.write_bytes(data[:2] + std_xmp(packet) + data[2:])
        return jpeg
    t = Tiff()
    path = jpeg.with_suffix(".tif")
    path.write_bytes(t.done(t.page(16, 8, undefined(0x02BC, packet))))
    return path


@pytest.mark.parametrize("fmt", ["jpeg", "tiff"])
@pytest.mark.parametrize(
    "body",
    [
        LITERAL.format("Use only with <i>credit</i> to Jane"),
        "<photoshop:Headline>Use only with <b>credit</b> to Jane</photoshop:Headline>",
    ],
    ids=["literal", "mixed"],
)
def test_xmp_exiftool_cant_carry_over_is_refused_and_caught(
    et: ExifTool, fixture_copy: FixtureCopy, fmt: str, body: str
) -> None:
    path = _with_xmp(fmt, fixture_copy("plain.jpg"), xmp_packet(body))
    before = snapshot(et, path)
    assert before.unprovable == "its XMP has markup inside a value, which exiftool can't carry over"
    legit_write(et, path, "xmp", before.image_hash)
    assert failing(check_written(et, before, path, compat="xmp")) == {"structure"}

    # Even unflagged, the XMP's own text shows what exiftool silently dropped:
    # the protected gate can't, as it reads the XMP with exiftool's parser.
    unflagged = dataclasses.replace(before, unprovable=None)
    results = check_written(et, unflagged, path, compat="xmp")
    assert failing(results) == {"structure"}
    assert by_name(results)["structure"].detail == "2 of the original XMP's values went missing"
    assert_private(results, "Jane", "Use only")


def test_extended_xmp_nobody_links_to_is_refused(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("plain.jpg")
    extended = xmp_packet("<photoshop:History>" + "depth " * 200 + "</photoshop:History>")
    guid = hashlib.md5(extended).hexdigest().upper().encode()
    data = path.read_bytes()
    segments = std_xmp(xmp_packet("")) + b"".join(ext_xmp(guid, extended))
    path.write_bytes(data[:2] + segments + data[2:])
    before = snapshot(et, path)
    assert before.unprovable == (
        "it has extended XMP its main XMP doesn't link to, which exiftool would drop"
    )


def test_png_with_imagemagick_raw_profile_is_refused(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    path = fixture_copy("plain.png")
    packet = xmp_packet("<photoshop:City>Rio</photoshop:City>").hex()
    profile = f"\nxmp\n{len(packet) // 2:8d}\n{packet}\n".encode()
    text = b"zTXt" + b"Raw profile type xmp\x00\x00" + zlib.compress(profile)
    raw = struct.pack(">I", len(text) - 4) + text + struct.pack(">I", zlib.crc32(text))
    data = path.read_bytes()
    idat = data.index(b"IDAT") - 4
    path.write_bytes(data[:idat] + raw + data[idat:])
    before = snapshot(et, path)
    assert before.unprovable == (
        "its XMP is in ImageMagick's raw-profile form, which exiftool would rewrite"
    )
    legit_write(et, path, "xmp", before.image_hash)
    assert raw not in path.read_bytes()  # why: exiftool rewrote the profile too


def test_unknown_format_cant_be_proven(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"GIF89a" + b"\x00" * 32)
    fake = FakeExifTool('[{"SourceFile": "photo.jpg"}]')
    before = snapshot(fake, path)  # type: ignore[arg-type]
    assert before.fmt == "other"
    assert before.unprovable == "As If can't check this file type byte by byte"
    structure = by_name(check_written(fake, before, path, compat="xmp"))["structure"]  # type: ignore[arg-type]
    assert structure == GateResult(
        "structure", False, f"can't prove a write is safe: {before.unprovable}"
    )


def test_tiff_preview_exiftool_would_move_is_refused(et: ExifTool, tmp_path: Path) -> None:
    # Page 2 in JPEG strips without a SubfileType: exiftool takes it for a
    # preview, which its hash doesn't cover, and moves it on every write.
    path = tmp_path / "scan.tif"
    t = Tiff()
    thumb = _thumbnail()
    at = t.blob(thumb)
    page2 = t.ifd(
        t.long(256, 16),
        t.long(257, 12),
        t.short(258, 8, 8, 8),
        t.short(259, 7),  # JPEG
        t.short(262, 6),  # YCbCr
        t.long(273, at),
        t.short(277, 3),
        t.long(278, 12),
        t.long(279, len(thumb)),
    )
    path.write_bytes(t.done(t.page(32, 24, next_ifd=page2)))
    before = snapshot(et, path)
    assert "IFD1:PreviewImageStart" in before.tags
    assert before.unprovable == (
        "it's a TIFF with an embedded preview that exiftool would move, which As If can't check"
    )
    legit_write(et, path, "xmp", before.image_hash)
    # Why: the protected gate would fail after the model's work.
    unflagged = dataclasses.replace(before, unprovable=None)
    protected = by_name(check_written(et, unflagged, path, compat="xmp"))["protected"]
    assert protected.detail == "changed: IFD1:PreviewImageStart"


def test_jpeg_compressed_tiff_is_a_format_limit_not_corrupt(et: ExifTool, tmp_path: Path) -> None:
    # exiftool takes the JPEG strips of Pillow's (and ImageMagick's, libtiff's)
    # JPEG TIFFs for a preview and computes no image-data hash at all.
    path = tmp_path / "scan.tif"
    path.write_bytes(pillow_tiff(compression="jpeg"))
    with pytest.raises(ExifToolError):
        et.image_data_hash(path)
    before = snapshot(et, path)
    assert before.image_hash == ""
    assert before.unprovable == "it's a JPEG-compressed TIFF, whose image data exiftool can't hash"
    assert not before.unprovable.startswith("its structure can't be checked")  # not quarantined
    assert "structure" in failing(check_written(et, before, path, compat="xmp"))


class UnhashableExifTool(FakeExifTool):
    def image_data_hash(self, path: Path) -> str:
        raise ExifToolError("exiftool couldn't compute the image-data hash")


@pytest.mark.parametrize(
    "dump",
    [
        pytest.param({}, id="nothing-known"),
        pytest.param({"IFD0:Compression": 1, "IFD0:StripOffsets": 8}, id="uncompressed"),
        pytest.param(
            {"IFD0:Compression": 7, "IFD0:PreviewImageStart": 8, "IFD0:StripOffsets": 8},
            id="strips-too",
        ),
    ],
)
def test_other_files_exiftool_cant_hash_still_raise(tmp_path: Path, dump: dict[str, Any]) -> None:
    # Only the one known format limit is let through; anything else may be corrupt.
    path = tmp_path / "scan.tif"
    path.write_bytes(pillow_tiff())
    fake = UnhashableExifTool(json.dumps([{"SourceFile": "scan.tif", **dump}]))
    with pytest.raises(ExifToolError):
        snapshot(fake, path)  # type: ignore[arg-type]
    jpeg_like = tmp_path / "photo.jpg"
    jpeg_like.write_bytes(b"\xff\xd8\xff\xd9")
    fake = UnhashableExifTool(
        json.dumps(
            [{"SourceFile": "photo.jpg", "IFD0:Compression": 7, "IFD0:PreviewImageStart": 8}]
        )
    )
    with pytest.raises(ExifToolError):
        snapshot(fake, jpeg_like)  # type: ignore[arg-type]


def _document_ancestors(path: Path, count: int) -> None:
    """Photoshop's well-known bloat: an XMP list of every document ever pasted in."""
    items = "".join(f"<rdf:li>xmp.did:{i:032x}</rdf:li>" for i in range(count))
    body = f"<photoshop:DocumentAncestors><rdf:Bag>{items}</rdf:Bag></photoshop:DocumentAncestors>"
    data = path.read_bytes()
    path.write_bytes(data[:2] + std_xmp(xmp_packet(body)) + data[2:])


def test_metadata_exiftool_reads_only_in_part_is_refused(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    path = fixture_copy("plain.jpg")
    _document_ancestors(path, 1001)
    before = snapshot(et, path)
    assert before.unprovable == (
        "exiftool reads only part of its metadata, so As If can't prove a write keeps the rest"
    )
    # Why: exiftool read 1000 of the 1001 items, so no gate could compare the last one.
    assert len(before.tags["XMP-photoshop:DocumentAncestors"]) == 1000

    path = fixture_copy("exif.jpg")
    _document_ancestors(path, 1000)  # within exiftool's limit: nothing to refuse
    assert snapshot(et, path).unprovable is None


@pytest.mark.parametrize(
    ("warning", "unprovable"),
    [
        pytest.param("[Minor] Extracted only 1000 ns:Tag items", True, id="Minor"),
        pytest.param("[Minor] Not decoding some large array(s)", True, id="large-array"),
        pytest.param("[minor] Possibly incorrect maker notes offsets", False, id="minor"),
        pytest.param("Unrecognized MakerNotes", False, id="plain"),
    ],
)
def test_exiftool_read_warnings_that_mean_lost_data(
    tmp_path: Path, warning: str, unprovable: bool
) -> None:
    # "[Minor]" (capital M) is exiftool's mark for a part it skipped while reading.
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"\xff\xd8\xff\xd9")
    dump = [{"SourceFile": "photo.jpg", "ExifTool:Warning": warning, "File:Comment": "x"}]
    before = snapshot(FakeExifTool(json.dumps(dump)), path)  # type: ignore[arg-type]
    expected = (
        "exiftool reads only part of its metadata, so As If can't prove a write keeps the rest"
    )
    assert before.unprovable == (expected if unprovable else None)
    assert "ExifTool:Warning" not in before.tags
    copies = [
        {
            "SourceFile": "photo.jpg",
            "ExifTool:Warning": "[minor] other",
            "ExifTool:Copy1:Warning": warning,
        }
    ]
    again = snapshot(FakeExifTool(json.dumps(copies)), path)  # type: ignore[arg-type]
    assert again.unprovable == before.unprovable  # every warning counts, not just the first


def test_cdata_markup_exiftool_cant_carry_over_is_refused(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    body = "<photoshop:City><![CDATA[Lisbon <b>old town</b>]]></photoshop:City>"
    path = _with_xmp("jpeg", fixture_copy("plain.jpg"), xmp_packet(body))
    before = snapshot(et, path)
    assert before.unprovable == "its XMP has markup inside a value, which exiftool can't carry over"
    legit_write(et, path, "xmp", before.image_hash)
    # Why: exiftool read the markup as a structure and dropped the value.
    unflagged = dataclasses.replace(before, unprovable=None)
    results = check_written(et, unflagged, path, compat="xmp")
    assert by_name(results)["structure"].detail == "1 of the original XMP's values went missing"
    assert_private(results, "Lisbon", "old town")


def _with_c2pa(fmt: str, path: Path) -> Path:
    """``path`` (a JPEG, PNG or WebP) with a C2PA manifest store where that format keeps it."""
    data = path.read_bytes()
    if fmt == "jpeg":
        path.write_bytes(data[:2] + app11(jumbf()) + data[2:])
    elif fmt == "png":
        at = data.index(b"IDAT") - 4
        path.write_bytes(data[:at] + chunk(b"caBX", jumbf()) + data[at:])
    else:
        body = data[12:] + rchunk(b"C2PA", jumbf())
        path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body)
    return path


@pytest.mark.parametrize(
    ("fmt", "name"), [("jpeg", "exif.jpg"), ("png", "plain.png"), ("webp", "lossless.webp")]
)
def test_content_credentials_are_refused(
    et: ExifTool, fixture_copy: FixtureCopy, monkeypatch: pytest.MonkeyPatch, fmt: str, name: str
) -> None:
    # A write leaves the manifest's bytes alone but breaks its signature.
    path = _with_c2pa(fmt, fixture_copy(name))
    before = snapshot(et, path)
    assert before.tags["JUMBF:JUMDLabel"] == "c2pa"
    assert before.unprovable == C2PA_PROBLEM
    assert not before.unprovable.startswith("its structure can't be checked")
    # exiftool's reading alone is enough, should the byte check ever miss one.
    monkeypatch.setattr(structure, "write_hazard", lambda data: None)
    assert snapshot(et, path).unprovable == C2PA_PROBLEM


@pytest.mark.parametrize("fmt", ["jpeg", "tiff"])
def test_content_credentials_kept_elsewhere_are_refused(
    et: ExifTool, fixture_copy: FixtureCopy, monkeypatch: pytest.MonkeyPatch, fmt: str
) -> None:
    # A remote manifest leaves only a link in the XMP, and its signature
    # covers the whole file, XMP included.
    link = "https://cai-manifests.example/manifests/urn:uuid:6f1c"
    packet = xmp_packet(
        f"<dcterms:provenance>{link}</dcterms:provenance>",
        attrs='xmlns:dcterms="http://purl.org/dc/terms/"',
    )
    path = _with_xmp(fmt, fixture_copy("plain.jpg"), packet)
    before = snapshot(et, path)
    assert before.tags["XMP-dcterms:Provenance"] == link
    assert before.unprovable == C2PA_LINK_PROBLEM
    # exiftool's reading alone is enough, should the XMP check ever miss one.
    monkeypatch.setattr(structure, "read_xmp", lambda data: structure.XmpText(Counter()))
    assert snapshot(et, path).unprovable == C2PA_LINK_PROBLEM


def _apple_png(path: Path, *, xmp_first: bool) -> None:
    """``path`` as Apple's ImageIO writes it: iDOT, then the image data in two
    IDAT chunks (and, with ``xmp_first``, XMP ahead of iDOT, as in a screenshot)."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (90, 140, 200)).save(buf, "PNG")
    data = buf.getvalue()
    idat_at = data.index(b"IDAT") - 4
    size = int.from_bytes(data[idat_at : idat_at + 4], "big")
    pixels = data[idat_at + 8 : idat_at + 8 + size]
    idats = chunk(b"IDAT", pixels[: size // 2]) + chunk(b"IDAT", pixels[size // 2 :])
    second = 40 + 12 + size // 2  # from iDOT's start: iDOT itself, then the first IDAT
    idot = chunk(b"iDOT", struct.pack(">7I", 2, 0, 12, 40, 12, 12, second))
    xmp = (
        chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp_packet(""))
        if xmp_first
        else b""
    )
    path.write_bytes(data[:idat_at] + xmp + idot + idats + data[idat_at + 12 + size :])


def test_apple_idot_png_is_refused_and_caught(et: ExifTool, tmp_path: Path) -> None:
    path = tmp_path / "IMG_0001.png"
    _apple_png(path, xmp_first=False)
    before = snapshot(et, path)
    assert before.unprovable == "it has Apple's iDOT chunk, whose offsets new XMP would break"
    legit_write(et, path, "xmp", before.image_hash)
    # Why: exiftool put the XMP between iDOT and the image data. Every byte
    # it points to is unchanged, yet it points at the wrong place now.
    unflagged = dataclasses.replace(before, unprovable=None)
    results = check_written(et, unflagged, path, compat="xmp")
    assert failing(results) == {"structure"}
    assert by_name(results)["structure"].detail == (
        "PNG iDOT chunk no longer points at the same image data"
    )


def test_apple_png_with_xmp_ahead_of_idot_passes(et: ExifTool, tmp_path: Path) -> None:
    # A macOS screenshot: exiftool rewrites that XMP where it is. (As If
    # writes no IPTC into a PNG, so "wide" is the same here.)
    path = tmp_path / "Screenshot.png"
    _apple_png(path, xmp_first=True)
    before = snapshot(et, path)
    assert before.unprovable is None
    legit_write(et, path, "xmp", before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat="xmp")), results


# -- maker notes pointing past a JPEG's image ------------------------------------------


@pytest.mark.parametrize(
    ("name", "detail"),
    [
        ("sony_preview.jpg", "added: File:PreviewImage; removed: Sony:PreviewImage"),
        ("olympus_preview.jpg", "changed: Olympus:PreviewImageStart"),
    ],
)
def test_preview_after_the_image_is_refused_and_caught(
    et: ExifTool, fixture_copy: FixtureCopy, name: str, detail: str
) -> None:
    path = fixture_copy(name)
    before = snapshot(et, path)
    assert before.unprovable == PREVIEW_AFTER_IMAGE
    legit_write(et, path, "xmp", before.image_hash)
    # Why: the new XMP moved the preview, and the maker note still points
    # where it was. Every byte is still there, so only this gate sees it.
    unflagged = dataclasses.replace(before, unprovable=None)
    results = check_written(et, unflagged, path, compat="xmp")
    assert failing(results) == {"protected"}
    assert by_name(results)["protected"].detail == detail


def _nikon_note(at: int) -> bytes:
    """A Nikon maker note: its own TIFF header, and a PreviewIFD whose preview
    sits inside the note, so inside the EXIF segment."""
    preview = scene_jpeg((160, 120), 41)
    tags = ((0x0001, 7, 4, b"0210"), long(0x0011, 0))
    preview_ifd = 8 + len(ifd(8, *tags))
    starts = 0x0201, 0x0202  # PreviewImageStart and -Length
    preview_at = preview_ifd + len(ifd(preview_ifd, long(starts[0], 0), long(starts[1], 0)))
    return (
        b"Nikon\x00\x02\x10\x00\x00II*\x00\x08\x00\x00\x00"
        + ifd(8, tags[0], (0x0011, 13, 1, preview_ifd))
        + ifd(preview_ifd, long(starts[0], preview_at), long(starts[1], len(preview)))
        + preview
    )


def _samsung_motion_photo() -> bytes:
    """A Galaxy motion photo: a Samsung maker note, and the video in Samsung's
    own trailer (SEFT), whose offsets count back from its directory."""
    exif = exif_segment(
        "samsung", "SM-G950F", lambda at: ifd(at, (0x0001, 7, 4, b"0100"), long(0x0002, 0x12000))
    )
    image = scene_jpeg((640, 480), 42)
    name = b"MotionPhoto_Data"
    video = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 400
    block = struct.pack("<HHI", 0, 0x0A30, len(name)) + name + video
    sefh = b"SEFH" + struct.pack("<IIHHII", 101, 1, 0, 0x0A30, len(block), len(block))
    seft = block + sefh + struct.pack("<I", len(sefh)) + b"SEFT"
    return image[:2] + exif + image[2:] + seft


_CONTAINER = (
    'xmlns:Container="http://ns.google.com/photos/1.0/container/" '
    'xmlns:Item="http://ns.google.com/photos/1.0/container/item/" '
)


def _directory(*items: str) -> str:
    """A Google container directory: the files in the JPEG, the primary image first."""
    lis = "".join(
        f'<rdf:li rdf:parseType="Resource"><Container:Item {item}/></rdf:li>' for item in items
    )
    return f"<Container:Directory><rdf:Seq>{lis}</rdf:Seq></Container:Directory>"


def _pixel_motion_photo() -> bytes:
    """A Pixel motion photo: its video, longer than the image, right after it."""
    video = b"\x00\x00\x00\x18ftypisom" + bytes(range(256)) * 400
    packet = xmp_packet(
        _directory(
            'Item:Semantic="Primary" Item:Mime="image/jpeg" Item:Length="0" Item:Padding="0"',
            f'Item:Semantic="MotionPhoto" Item:Mime="video/mp4" Item:Length="{len(video)}"',
        ),
        attrs=_CONTAINER + 'xmlns:GCamera="http://ns.google.com/photos/1.0/camera/" '
        f'GCamera:MotionPhoto="1" GCamera:MotionPhotoVersion="1" GCamera:MicroVideo="1" '
        f'GCamera:MicroVideoOffset="{len(video)}"',
    )
    image = scene_jpeg((640, 480), 43)
    return image[:2] + std_xmp(packet) + image[2:] + video


def _ultra_hdr() -> bytes:
    """An Ultra HDR photo: XMP, MPF, and the gain map (with XMP of its own) after it."""
    hdrgm = 'xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0"'
    gain = scene_jpeg((320, 240), 44)
    ranges = ' hdrgm:GainMapMax="2.3" hdrgm:OffsetSDR="0.015625" hdrgm:OffsetHDR="0.015625"'
    gain = gain[:2] + std_xmp(xmp_packet("", attrs=hdrgm + ranges)) + gain[2:]
    primary = xmp_packet(
        _directory(
            'Item:Semantic="Primary" Item:Mime="image/jpeg"',
            f'Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="{len(gain)}"',
        ),
        attrs=_CONTAINER + hdrgm,
    )
    return with_mpf(std_xmp(primary), gain, 45)


def _fotostation_preview() -> bytes:
    """A FotoWare FotoStation trailer: a record holding a preview of its own."""
    preview = scene_jpeg((160, 120), 46)
    record = preview + struct.pack(">HII", 4, len(preview) + 10, 0xA1B2C3D4)  # tag, size, signature
    return scene_jpeg((640, 480), 47) + record


def _olympus_pointing_past_the_end() -> bytes:
    """An Olympus whose preview pointer leads past the end of the file, and no preview."""
    data = olympus_jpeg(48, preview_at=10**7)
    return data[: jpeg_eoi_offset(data)]


# Each one with a tag showing exiftool read what the case is about.
UNBROKEN: dict[str, tuple[Callable[[], bytes], str]] = {
    "sony-without-0x2001": (
        lambda: sony_jpeg(49, preview_pointer=False),
        "MPImage2:MPImageStart",  # past the image, but counted from the MPF segment
    ),
    "olympus-pointing-past-the-end": (_olympus_pointing_past_the_end, "Olympus:PreviewImageStart"),
    "nikon-preview-in-the-exif-and-mpf": (
        lambda: with_mpf(
            exif_segment("NIKON CORPORATION", "NIKON D750", _nikon_note),
            scene_jpeg((320, 240), 50),
            51,
        ),
        "PreviewIFD:PreviewImage",
    ),
    "canon-with-mpf": (
        lambda: with_mpf(
            exif_segment("Canon", "Canon EOS R", canon_maker_note), scene_jpeg((320, 240), 52), 53
        ),
        "Canon:CanonFirmwareVersion",
    ),
    "samsung-motion-photo": (_samsung_motion_photo, "Samsung:EmbeddedVideoFile"),
    "pixel-motion-photo": (_pixel_motion_photo, "XMP-GCamera:MicroVideoOffset"),
    "ultra-hdr": (_ultra_hdr, "XMP-hdrgm:Version"),
    "fotostation-preview": (_fotostation_preview, "FotoStation:PreviewImage"),
}


@pytest.mark.parametrize("case", UNBROKEN)
def test_previews_a_write_cant_break_pass(et: ExifTool, tmp_path: Path, case: str) -> None:
    make, shown = UNBROKEN[case]
    path = tmp_path / "photo.jpg"
    path.write_bytes(make())
    before = snapshot(et, path)
    assert shown in before.tags
    assert before.unprovable is None
    legit_write(et, path, "xmp", before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat="xmp")), results


def _unprovable(path: Path, dump: dict[str, Any]) -> str | None:
    fake = FakeExifTool(json.dumps([{"SourceFile": path.name, **dump}]))
    return snapshot(fake, path).unprovable  # type: ignore[arg-type]


def test_what_points_past_the_image(tmp_path: Path) -> None:
    # Dumps as exiftool gives them, for a file laid out like the Sony fixture.
    path = tmp_path / "DSC00001.JPG"
    path.write_bytes(data := sony_jpeg(54, preview_pointer=True))
    eoi = jpeg_eoi_offset(data)
    # Sony's preview, as exiftool gives it: the first byte made a marker again.
    preview = b"\xff" + data[data.index(b"\x00\xd8\xff\xdb", eoi) + 1 :]
    sony = "base64:" + base64.b64encode(preview).decode()
    maker_note = {"ExifIFD:MakerNoteSony": "base64:AA=="}
    assert _unprovable(path, {**maker_note, "Sony:PreviewImage": sony}) == PREVIEW_AFTER_IMAGE
    # The maker note's own previews only: a trailer exiftool reads may carry
    # its own (FotoStation's does), which survives a write.
    assert _unprovable(path, {**maker_note, "FotoStation:PreviewImage": sony}) is None
    assert _unprovable(path, {"Sony:PreviewImage": sony}) is None  # no Sony maker note
    for key, value in {
        "Olympus:PreviewImageStart": eoi,
        "PreviewIFD:PreviewImageStart": f"{eoi - 100} {eoi + 10}",
        "Canon:OriginalDecisionDataOffset": len(data) - 1,
    }.items():
        assert _unprovable(path, {key: value}) == PREVIEW_AFTER_IMAGE, key
    for key, value in {
        "Olympus:PreviewImageStart": eoi - 1,  # in the image's own segments
        "Minolta:PreviewImageStart": len(data),  # past the end of the file: nowhere
        "MPImage2:MPImageStart": eoi,  # counted from the MPF segment, which moves too
        "XMP-GCamera:MicroVideoOffset": eoi,  # counted from the end of the file
        "XMP-xmp:PreviewImageStart": eoi,
        "Olympus:PreviewImageLength": eoi,
        "Casio:PreviewImageStart": "(none)",
    }.items():
        assert _unprovable(path, {key: value}) is None, key


def test_photos_with_a_preview_after_the_image_never_reach_the_model(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # More than the failure budget allows (5 in 200), had they counted as failures.
    src = tmp_path / "Pictures"
    src.mkdir()
    for i in range(6):
        shutil.copy2(fixtures_dir / "sony_preview.jpg", src / f"DSC0000{i}.JPG")
    shutil.copy2(fixtures_dir / "olympus_preview.jpg", src / "P1010001.JPG")
    shutil.copy2(fixtures_dir / "plain.jpg", src / "plain.jpg")
    asked: list[str] = []
    describe = local._describe
    monkeypatch.setattr(
        local, "_describe", lambda ctx, ram, img: asked.append(img) or describe(ctx, ram, img)
    )
    cfg = Config()
    cfg.model.id = "stub"
    cfg.resources.work_dir = str(tmp_path / "ram")
    cfg.resources.allow_disk_workdir = True
    log = TimingsLog(tmp_path / "timings.jsonl", run_id="r_test", model="stub")
    summary = caption_folder(
        src, tmp_path / "out", cfg, model=choose_model(cfg), run_id="r_test", log=log
    )
    assert summary.aborted is None
    outcomes = {o.rel: (o.state, o.reason, o.systematic) for o in summary.outcomes}
    assert outcomes.pop("plain.jpg")[0] == State.CLEAN
    skipped = (State.SKIPPED_FORMAT, f"can't be written safely: {PREVIEW_AFTER_IMAGE}", False)
    assert len(outcomes) == 7 and set(outcomes.values()) == {skipped}
    assert len(asked) == 1  # plain.jpg alone


# -- a Pentax preview in its own APP7 segment ------------------------------------------------

PENTAX_START, PENTAX_LENGTH = "Pentax:PreviewImageStart", "Pentax:PreviewImageLength"
# Where the fixture's (and the K-1 Mark II's) APP7 keeps the preview: past
# "PENTAX \0II" and an IFD of four entries. And where it keeps that offset.
IN_APP7, POINTER_IN_APP7 = 64, 44


def _pentax_preview(et: ExifTool, path: Path) -> bytes:
    """The preview, as ``exiftool -b -PreviewImage`` extracts it."""
    value = et.read_tags(path, "-Pentax:PreviewImage", binary=True)["Pentax:PreviewImage"]
    assert isinstance(value, str) and value.startswith("base64:")
    return base64.b64decode(value[7:])


def _app7(data: bytes | None) -> tuple[int, bytes]:
    """Where the Pentax APP7 segment's data starts in the file, and the data."""
    assert data is not None
    (found,) = structure.jpeg_segments(data, 0xE7, b"PENTAX \x00")
    return found


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_pentax_preview_moves_with_its_app7_segment_and_passes(
    et: ExifTool, fixture_copy: FixtureCopy, compat: Compat
) -> None:
    path = fixture_copy("pentax_preview.jpg")
    preview = _pentax_preview(et, path)
    assert preview == scene_jpeg((320, 240), 25)  # the camera's preview, start to end
    before = snapshot(et, path)
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    # The new XMP went in front of APP7, and exiftool's number moved with it...
    (old_at, old_app7), (new_at, new_app7) = _app7(before.data), _app7(path.read_bytes())
    assert new_at > old_at and new_app7 == old_app7
    new_start = et.read_tags(path, "-n", f"-{PENTAX_START}")[PENTAX_START]
    assert before.tags[PENTAX_START] - old_at == new_start - new_at == IN_APP7
    # ...while the preview stayed where it was, as exiftool reads the written file.
    assert _pentax_preview(et, path) == preview
    assert all_ok(results := check_written(et, before, path, compat=compat)), results


@pytest.mark.parametrize("where", [IN_APP7 + 100, POINTER_IN_APP7], ids=["preview", "pointer"])
def test_pentax_app7_changes_still_fail(
    et: ExifTool, fixture_copy: FixtureCopy, where: int
) -> None:
    path = fixture_copy("pentax_preview.jpg")
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash)
    _flip(path, _app7(path.read_bytes())[0] + where)
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"protected", "structure"}
    # A segment that changed carries no offset along: the new number counts too.
    assert by_name(results)["protected"].detail == f"changed: Pentax:PreviewImage, {PENTAX_START}"
    assert "APP7" in by_name(results)["structure"].detail


def _xmp_after_exif(data: bytes) -> bytes:
    """``data`` with an XMP segment right after its EXIF, where exiftool puts new XMP."""
    end = 4 + int.from_bytes(data[4:6], "big")
    return data[:end] + std_xmp(xmp_packet("")) + data[end:]


def _pentax_protected(
    before: dict[str, Any], after: dict[str, Any], data: bytes, written: bytes | None
) -> str:
    """The protected gate's verdict for two tag dumps and the files' bytes."""
    snap = Snapshot("jpeg", "0" * 64, len(data), "", before, data, Counter(), None)
    fake = FakeExifTool(json.dumps([{"SourceFile": "photo.jpg", **after}]))
    result = _protected(fake, snap, Path("photo.jpg"), written, "xmp")  # type: ignore[arg-type]
    return result.detail


def test_only_an_offset_that_moved_with_its_unchanged_segment_may_change() -> None:
    data = pentax_jpeg(60)
    written = _xmp_after_exif(data)
    (old_at, app7), (new_at, _) = _app7(data), _app7(written)
    assert app7.startswith(PENTAX_APP7) and new_at > old_at
    length = len(scene_jpeg((320, 240), 61))
    old, new = old_at + IN_APP7, new_at + IN_APP7
    tags = {PENTAX_START: old, PENTAX_LENGTH: length}
    assert _pentax_protected(tags, {**tags, PENTAX_START: new}, data, written) == ""
    # The second copy, after one from the maker note, which XMP doesn't move.
    two = {
        PENTAX_START: 100,
        PENTAX_LENGTH: 10,
        f"{PENTAX_START}#2": old,
        f"{PENTAX_LENGTH}#2": length,
    }
    assert _pentax_protected(two, {**two, f"{PENTAX_START}#2": new}, data, written) == ""
    full = {PENTAX_START: old, PENTAX_LENGTH: len(app7) - IN_APP7}  # up to the segment's end
    assert _pentax_protected(full, {**full, PENTAX_START: new}, data, written) == ""
    flipped = bytearray(written)
    flipped[new_at + IN_APP7 + 100] ^= 1
    renamed = PENTAX_APP7, b"PENTAX!\x00II"
    olympus = {"Olympus:PreviewImageStart": old, "Olympus:PreviewImageLength": length}
    for case, (before, after, before_data, after_data) in {
        "moved by another amount": (tags, {**tags, PENTAX_START: new + 1}, data, written),
        "a segment that changed": (tags, {**tags, PENTAX_START: new}, data, bytes(flipped)),
        "not Pentax's segment": (
            tags,
            {**tags, PENTAX_START: new},
            data.replace(*renamed),
            written.replace(*renamed),
        ),
        "a preview running past the segment": (
            {**tags, PENTAX_LENGTH: len(app7) - IN_APP7 + 1},
            {PENTAX_START: new, PENTAX_LENGTH: len(app7) - IN_APP7 + 1},
            data,
            written,
        ),
        "no length": ({PENTAX_START: old}, {PENTAX_START: new}, data, written),
        "a length that changed": (
            tags,
            {PENTAX_START: new, PENTAX_LENGTH: length - 1},
            data,
            written,
        ),
        "the written bytes unread": (tags, {**tags, PENTAX_START: new}, data, None),
        "a start in the EXIF": (
            {**tags, PENTAX_START: 20},
            {**tags, PENTAX_START: 20 + new_at - old_at},
            data,
            written,
        ),
    }.items():
        detail = _pentax_protected(before, after, before_data, after_data)
        assert detail.startswith("changed: ") and PENTAX_START in detail, case
    # Pentax's alone: another maker's offset doesn't move with a segment.
    moved = {**olympus, "Olympus:PreviewImageStart": new}
    assert _pentax_protected(olympus, moved, data, written) == (
        "changed: Olympus:PreviewImageStart"
    )


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_photoshop_7_era_xmp_passes(
    et: ExifTool, fixture_copy: FixtureCopy, compat: Compat
) -> None:
    # x:xapmeta/x:xaptk: exiftool reads x:xaptk as the toolkit name and
    # writes it back as x:xmptk. Nothing of the owner's is lost.
    packet = (
        "<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        "<x:xapmeta xmlns:x='adobe:ns:meta/' x:xaptk='XMP toolkit 2.8.2-33, framework 1.5'>"
        "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        "<rdf:Description about='uuid:5d0f8a64-4a2f-11d9-8d8a-000a95c8b4b2'"
        " xmlns:xapMM='http://ns.adobe.com/xap/1.0/mm/'>"
        "<xapMM:DocumentID>adobe:docid:photoshop:5d0f8a63-4a2f-11d9</xapMM:DocumentID>"
        "</rdf:Description></rdf:RDF></x:xapmeta><?xpacket end='w'?>"
    ).encode()
    path = _with_xmp("jpeg", fixture_copy("plain.jpg"), packet)
    before = snapshot(et, path)
    assert before.tags["XMP-x:XMPToolkit"] == "XMP toolkit 2.8.2-33, framework 1.5"
    assert before.unprovable is None
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat=compat)), results


# -- additive-only, enforced by the gate --------------------------------------------------------

OURS = "A dog on the beach"
OURS_DIGEST = hashlib.sha256(OURS.encode()).hexdigest()


def _protected_detail(before: dict[str, Any], after: dict[str, Any], compat: Compat) -> str:
    """The protected gate's verdict for two tag dumps, without exiftool."""
    snap = Snapshot("jpeg", "0" * 64, 0, "", before, None, Counter(), None)
    fake = FakeExifTool(json.dumps([{"SourceFile": "photo.jpg", **after}]))
    result = _protected(fake, snap, Path("photo.jpg"), None, compat)  # type: ignore[arg-type]
    assert result.ok == (result.detail == "")
    return result.detail


SUBJECT, KEYWORDS_IPTC, DESC = "XMP-dc:Subject", "IPTC:Keywords", "XMP-dc:Description"
DIGEST, HASH = "XMP-asif:CaptionDigest", "XMP-et:OriginalImageHash"
CHARSET, RECORD = "IPTC:CodedCharacterSet", "IPTC:ApplicationRecordVersion"
LOST_SUBJECT = "lost values: XMP-dc:Subject"
ADDITIVE = {
    "added": ({SUBJECT: ["a", "b"]}, {SUBJECT: ["b", "a", "c"]}, ""),
    "one-to-list": ({SUBJECT: "a"}, {SUBJECT: ["a", "b"]}, ""),
    "removed": ({SUBJECT: ["a", "b"]}, {SUBJECT: ["a", "c"]}, LOST_SUBJECT),
    "duplicate-collapsed": ({SUBJECT: ["a", "a"]}, {SUBJECT: ["a", "b"]}, LOST_SUBJECT),
    "case-changed": ({SUBJECT: ["Dog"]}, {SUBJECT: ["dog"]}, LOST_SUBJECT),
    "type-changed": ({SUBJECT: [2019]}, {SUBJECT: ["2019"]}, LOST_SUBJECT),
    "all-gone": ({SUBJECT: ["a"]}, {}, LOST_SUBJECT),
    "iptc-removed": (
        {KEYWORDS_IPTC: ["old"]},
        {KEYWORDS_IPTC: ["new"]},
        "lost values: IPTC:Keywords",
    ),
    "iptc-copies": (
        {KEYWORDS_IPTC: ["a"], f"{KEYWORDS_IPTC}#2": ["b"]},
        {KEYWORDS_IPTC: ["b", "a"]},
        "",
    ),
    # Captions: the owner's never change; As If's own (by the digest recorded before) may.
    "owner-caption": ({DESC: "Oma"}, {DESC: OURS}, "changed: XMP-dc:Description"),
    "owner-caption-removed": (
        {f"{DESC}-de": "Oma"},
        {DESC: OURS},
        "removed: XMP-dc:Description-de",
    ),
    "ours-replaced": (
        {DESC: OURS, f"{DESC}-en": OURS, DIGEST: OURS_DIGEST},
        {DESC: "new", f"{DESC}-pt-BR": "new"},
        "",
    ),
    "ours-edited-since": (
        {DESC: OURS + "!", DIGEST: OURS_DIGEST},
        {DESC: "new"},
        "changed: XMP-dc:Description",
    ),
    "digest-written-now": (
        {DESC: OURS},
        {DESC: "new", DIGEST: OURS_DIGEST},
        "changed: XMP-dc:Description",
    ),
    "digest-malformed": (
        {DESC: OURS, DIGEST: OURS_DIGEST.upper()},
        {DESC: "new"},
        "changed: XMP-dc:Description",
    ),
    "not-text": (
        {DESC: 2019, DIGEST: hashlib.sha256(b"2019").hexdigest()},
        {DESC: "new"},
        "changed: XMP-dc:Description",
    ),
    "blank": ({DESC: "  "}, {DESC: "new"}, ""),
    "empty": ({}, {DESC: "new", f"{DESC}-en": "new"}, ""),
    # Write-once fields.
    "hash-added": ({}, {HASH: "b" * 64, "XMP-et:OriginalImageHashType": "SHA256"}, ""),
    "hash-overwritten": ({HASH: "a" * 64}, {HASH: "b" * 64}, "changed: XMP-et:OriginalImageHash"),
    "iptc-relabelled": (
        {CHARSET: "\x1b%A"},
        {CHARSET: "\x1b%G"},
        "changed: IPTC:CodedCharacterSet",
    ),
    "iptc-version-removed": ({RECORD: 2}, {}, "removed: IPTC:ApplicationRecordVersion"),
    "iptc-created": ({}, {CHARSET: "\x1b%G", RECORD: 4}, ""),
}


@pytest.mark.parametrize(("before", "after", "detail"), ADDITIVE.values(), ids=ADDITIVE.keys())
def test_fields_as_if_writes_are_additive_only(
    before: dict[str, Any], after: dict[str, Any], detail: str
) -> None:
    assert _protected_detail(before, after, "wide") == detail
    for value in ("Oma", OURS, "Dog", "old", "a" * 64):
        assert value not in detail


IPTC_DIGEST, CURRENT_DIGEST = "Photoshop:IPTCDigest", "File:CurrentIPTCDigest"
MD5_A, MD5_B, MD5_C = "a" * 32, "b" * 32, "c" * 32
STALE = "out of step with the IPTC: Photoshop:IPTCDigest"
# Apps following the MWG guidelines show the XMP while Photoshop:IPTCDigest
# matches the IPTC (File:CurrentIPTCDigest), or there is no IPTC.
DIGESTS = {
    "kept-in-step": (
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_A},
        {IPTC_DIGEST: MD5_B, CURRENT_DIGEST: MD5_B},
        "",
    ),
    "no-iptc-before": ({IPTC_DIGEST: MD5_A}, {IPTC_DIGEST: MD5_B, CURRENT_DIGEST: MD5_B}, ""),
    "iptc-untouched": (
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_A},
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_A},
        "",
    ),
    "no-digest": ({CURRENT_DIGEST: MD5_A}, {CURRENT_DIGEST: MD5_B}, ""),
    # Already stale, the IPTC is shown either way: the owner's digest stays.
    "stale-left-alone": (
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_C},
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_B},
        "",
    ),
    "stale-changed": (
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_C},
        {IPTC_DIGEST: MD5_B, CURRENT_DIGEST: MD5_B},
        "changed: Photoshop:IPTCDigest",
    ),
    # The XMP was shown, and the IPTC would be.
    "left-stale": (
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_A},
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_B},
        STALE,
    ),
    "left-stale-new-iptc": (
        {IPTC_DIGEST: MD5_A},
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_B},
        STALE,
    ),
    "made-stale": (
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_A},
        {IPTC_DIGEST: MD5_C, CURRENT_DIGEST: MD5_B},
        f"changed: Photoshop:IPTCDigest; {STALE}",
    ),
    "no-iptc-after": ({IPTC_DIGEST: MD5_A}, {IPTC_DIGEST: MD5_B}, "changed: Photoshop:IPTCDigest"),
    "added": (
        {CURRENT_DIGEST: MD5_A},
        {IPTC_DIGEST: MD5_B, CURRENT_DIGEST: MD5_B},
        "added: Photoshop:IPTCDigest",
    ),
    "removed": (
        {IPTC_DIGEST: MD5_A, CURRENT_DIGEST: MD5_A},
        {CURRENT_DIGEST: MD5_B},
        "removed: Photoshop:IPTCDigest",
    ),
}


@pytest.mark.parametrize(("before", "after", "detail"), DIGESTS.values(), ids=DIGESTS.keys())
def test_iptc_digest_may_only_stay_in_step(
    before: dict[str, Any], after: dict[str, Any], detail: str
) -> None:
    assert _protected_detail(before, after, "wide") == detail


def test_iptc_digest_cant_change_at_all_in_xmp_mode() -> None:
    before, after, _ = DIGESTS["kept-in-step"]
    detail = _protected_detail(before, after, "xmp")
    assert detail == "changed: File:CurrentIPTCDigest, Photoshop:IPTCDigest"


def test_iptc_keywords_cant_change_at_all_in_xmp_mode() -> None:
    before, after = {"IPTC:Keywords": ["old"]}, {"IPTC:Keywords": ["old", "new"]}
    assert _protected_detail(before, after, "xmp") == "changed: IPTC:Keywords"
    assert _protected_detail(before, after, "wide") == ""


def test_owner_caption_overwritten_by_exiftool_fails(
    et: ExifTool, fixture_copy: FixtureCopy
) -> None:
    path = fixture_copy("user_xmp.jpg")
    before = snapshot(et, path)
    # As if the writer's own rule had failed: the gate doesn't trust it.
    et.execute("-overwrite_original", f"-XMP-dc:Description-x-default={CAPTION}", str(path))
    results = check_written(et, before, path, compat="xmp")
    assert failing(results) == {"protected", "structure"}
    assert by_name(results)["protected"].detail == "changed: XMP-dc:Description"
    assert by_name(results)["structure"].detail == "1 of the original XMP's values went missing"
    assert_private(results, "Grandma")


def test_as_ifs_own_caption_may_be_replaced(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("exif.jpg")
    first = snapshot(et, path)
    legit_write(et, path, "xmp", first.image_hash)
    before = snapshot(et, path)
    legit_write(et, path, "xmp", before.image_hash, caption="Outra legenda")
    assert et.read_tags(path, "-XMP-dc:Description")["XMP-dc:Description"] == "Outra legenda"
    assert all_ok(results := check_written(et, before, path, compat="xmp")), results

    # Once the owner edits it, it's theirs.
    et.execute("-overwrite_original", "-XMP-dc:Description-pt-BR=Minha legenda", str(path))
    before = snapshot(et, path)
    et.execute("-overwrite_original", "-XMP-dc:Description-pt-BR=Outra", str(path))
    results = check_written(et, before, path, compat="xmp")
    assert by_name(results)["protected"].detail == "changed: XMP-dc:Description-pt-BR"


@pytest.mark.parametrize(
    ("name", "edit", "detail"),
    [
        ("user_xmp.jpg", "-XMP-dc:Subject-=family", "lost values: XMP-dc:Subject"),
        ("exif.jpg", "-XMP-et:OriginalImageHash=" + "0" * 64, "changed: XMP-et:OriginalImageHash"),
    ],
)
def test_keywords_and_original_hash_survive_real_writes(
    et: ExifTool, fixture_copy: FixtureCopy, name: str, edit: str, detail: str
) -> None:
    path = fixture_copy(name)
    first = snapshot(et, path)
    legit_write(et, path, "wide", first.image_hash)
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash, keywords=("areia",))
    assert all_ok(check_written(et, before, path, compat="wide"))
    et.execute("-overwrite_original", edit, str(path))
    assert by_name(check_written(et, before, path, compat="wide"))["protected"].detail == detail


def test_iptc_keywords_survive_real_writes(et: ExifTool, fixture_copy: FixtureCopy) -> None:
    path = fixture_copy("plain.jpg")
    et.execute(
        "-overwrite_original", "-charset", "iptc=UTF8", "-IPTC:CodedCharacterSet=UTF8",
        "-IPTC:Keywords=família", str(path),
    )  # fmt: skip
    before = snapshot(et, path)
    legit_write(et, path, "wide", before.image_hash)
    assert all_ok(check_written(et, before, path, compat="wide"))
    et.execute("-overwrite_original", "-charset", "iptc=UTF8", "-IPTC:Keywords-=família", str(path))
    detail = by_name(check_written(et, before, path, compat="wide"))["protected"].detail
    assert detail == "lost values: IPTC:Keywords"


# -- shaped like real photos that were once misjudged -------------------------------------
# Each stands in for a photo of a real-image corpus that the gates failed
# after the model's work (a systematic failure, which stops runs) or that
# was refused as corrupt although it is only a limit of exiftool.


@pytest.mark.parametrize(("overrun", "refused"), [(0, False), (1, False), (2, True)])
def test_exif_thumbnail_running_past_its_segment(
    et: ExifTool, tmp_path: Path, overrun: int, refused: bool
) -> None:
    # Sigma's DP2 Quattro: the thumbnail's length counts two bytes past the
    # EXIF segment. The first, the next segment's 0xFF, is 0xFF after any
    # write (Sony's DCR-PC110 counts just that one); the second, its marker,
    # becomes new XMP's APP1 marker.
    path = tmp_path / "SDIM0001.JPG"
    path.write_bytes(thumbnail_overrun_jpeg(31, overrun))
    before = snapshot(et, path)
    assert before.unprovable == (PAST_EXIF_SEGMENT if refused else None)
    legit_write(et, path, "xmp", before.image_hash)
    unflagged = dataclasses.replace(before, unprovable=None)
    results = check_written(et, unflagged, path, compat="xmp")
    if refused:
        # Why: the protected gate would fail after the model's work.
        assert failing(results) == {"protected"}
        assert by_name(results)["protected"].detail == "changed: IFD1:ThumbnailImage"
    else:
        assert all_ok(results), results


def test_what_runs_past_the_exif_segment(tmp_path: Path) -> None:
    # Dumps as exiftool gives them (-n: offsets from the start of the file),
    # for a file laid out like the DP2 Quattro's.
    path = tmp_path / "SDIM0001.JPG"
    path.write_bytes(data := thumbnail_overrun_jpeg(31, 0))
    [(begin, payload)] = structure.jpeg_segments(data, 0xE1, b"Exif\x00\x00")
    end = begin + len(payload)
    at = end - 100  # a block ending 100 bytes before the segment does
    for dump in (
        {"IFD1:ThumbnailOffset": at, "IFD1:ThumbnailLength": 102},
        {"IFD1:ThumbnailOffset": at, "IFD1:ThumbnailLength": 100_000},
        {"Olympus:PreviewImageStart": at, "Olympus:PreviewImageLength": 150},
        {"Sigma:PreviewImageStart": f"{at} {at}", "Sigma:PreviewImageLength": "10 102"},
        {"IFD1:ThumbnailOffset#2": at, "IFD1:ThumbnailLength#2": 102},
    ):
        assert _unprovable(path, dump) == PAST_EXIF_SEGMENT, dump
    for dump in (
        {"IFD1:ThumbnailOffset": at, "IFD1:ThumbnailLength": 100},  # ends where the segment does
        {"IFD1:ThumbnailOffset": at, "IFD1:ThumbnailLength": 101},  # the next segment's 0xFF
        {"IFD1:ThumbnailOffset": at, "IFD1:ThumbnailLength#2": 102},  # not this one's length
        {"IFD1:ThumbnailOffset": at},  # no length: nothing to tell
        {"IFD1:ThumbnailOffset": end, "IFD1:ThumbnailLength": 102},  # starts past the EXIF
        {"MPImage2:MPImageStart": at, "MPImage2:MPImageLength": 102},  # counted from MPF
        {"XMP-xmp:ThumbnailOffset": at, "XMP-xmp:ThumbnailLength": 102},
        {"IFD1:ThumbnailOffset": "(none)", "IFD1:ThumbnailLength": 102},
    ):
        assert _unprovable(path, dump) is None, dump


def test_png_with_a_bad_crc_is_refused_as_corrupt(et: ExifTool, tmp_path: Path) -> None:
    # PngSuite's xcsn0g01: the image data's CRC is wrong. Decoders show it,
    # a plain read finds nothing wrong, but exiftool refuses to write it.
    path = tmp_path / "scan.png"
    path.write_bytes(bad_crc_png(32))
    before = snapshot(et, path)
    assert before.unprovable == "its structure can't be checked: PNG IDAT chunk has a bad CRC"
    with pytest.raises(ExifToolError, match=r"Error: \[minor\] Bad CRC for IDAT chunk"):
        legit_write(et, path, "xmp", before.image_hash)


def test_tiff_with_old_style_jpeg_tables_is_refused(et: ExifTool, tmp_path: Path) -> None:
    path = tmp_path / "scan.tif"
    path.write_bytes(jpeg_tables_tiff(34))
    before = snapshot(et, path)
    assert {"IFD0:JPEGQTables", "IFD0:JPEGDCTables", "IFD0:JPEGACTables"} <= before.tags.keys()
    assert before.unprovable == (
        "it's a TIFF with old-style JPEG tables (or shared data) stored by offset, "
        "which exiftool can't write"
    )
    # Why: exiftool refuses every write to it.
    with pytest.raises(ExifToolError, match="No size tag for IFD0:JPEG"):
        legit_write(et, path, "xmp", before.image_hash)


@pytest.mark.parametrize(
    "key",
    ["IFD0:JPEGQTables", "IFD3:JPEGDCTables", "SubIFD:JPEGACTables#2", "IFD0:SharedData"],
)
def test_every_tag_exiftool_cant_write_a_tiff_with(tmp_path: Path, key: str) -> None:
    tiff = tmp_path / "scan.tif"
    tiff.write_bytes(pillow_tiff())
    expected = "it's a TIFF with old-style JPEG tables (or shared data) stored by offset"
    assert (_unprovable(tiff, {key: "8"}) or "").startswith(expected)
    jpeg = tmp_path / "photo.jpg"
    jpeg.write_bytes(scene_jpeg((64, 48), 1))
    assert _unprovable(jpeg, {key: "8"}) is None  # a JPEG's EXIF is copied, not rebuilt


def test_tiff_reusing_image_data_is_refused(et: ExifTool, tmp_path: Path) -> None:
    # Old-style JPEG scans point each page's JPEG block at its strips;
    # here two strips are the same bytes. exiftool writes each out.
    path = tmp_path / "scan.tif"
    path.write_bytes(shared_strips_tiff(35))
    before = snapshot(et, path)
    shared = 512 * 150
    assert before.unprovable == (
        f"the TIFF reuses {shared} bytes of image data, which exiftool would write out again "
        "for each use"
    )
    legit_write(et, path, "xmp", before.image_hash)
    # Why: the size gate fails after the model's work; nothing else does.
    results = check_written(et, dataclasses.replace(before, unprovable=None), path, compat="xmp")
    assert failing(results) == {"size"}
    assert path.stat().st_size - before.size > shared


def test_bigtiff_is_a_format_limit_not_corrupt(et: ExifTool, tmp_path: Path) -> None:
    path = tmp_path / "pano.tif"
    path.write_bytes(bigtiff(36))
    assert et.read_tags(path, "-File:FileType") == {"File:FileType": "BTF"}  # readable
    with pytest.raises(ExifToolError):
        et.image_data_hash(path)  # but not hashable, so it looked corrupt
    before = snapshot(et, path)
    assert before.unprovable == BIGTIFF
    assert not BIGTIFF.startswith("its structure can't be checked")  # not quarantined
    assert "structure" in failing(check_written(et, before, path, compat="xmp"))
    with pytest.raises(ExifToolError, match="BigTIFF"):
        legit_write(et, path, "xmp", "0" * 64)
    # A damaged one exiftool can't read as a BigTIFF may well be corrupt.
    header = bytearray(path.read_bytes())
    header[4] = 9  # 64-bit offsets are 8 bytes, never 9
    path.write_bytes(bytes(header))
    with pytest.raises(ExifToolError):
        snapshot(et, path)


@pytest.mark.parametrize("compat", ["xmp", "wide"])
def test_xmp_leftovers_holding_nothing_pass(et: ExifTool, tmp_path: Path, compat: Compat) -> None:
    # Photoshop CC 2019 left a ">" of an older packet after the trailer.
    # exiftool drops it, which nothing that reads XMP can tell.
    path = tmp_path / "edited.png"
    path.write_bytes(xmp_leftovers_png(33, b">"))
    before = snapshot(et, path)
    assert before.unprovable is None
    assert before.xmp == Counter({"Pier": 1})
    legit_write(et, path, compat, before.image_hash)
    assert all_ok(results := check_written(et, before, path, compat=compat)), results


def test_xmp_leftovers_that_may_hold_values_are_refused(et: ExifTool, tmp_path: Path) -> None:
    # Lightroom left the end of an older packet (a face region) after the trailer.
    path = tmp_path / "scan.png"
    path.write_bytes(xmp_leftovers_png(33, STALE_PACKET))
    before = snapshot(et, path)
    assert before.unprovable == LEFTOVERS_PROBLEM
    assert not LEFTOVERS_PROBLEM.startswith("its structure can't be checked")  # not quarantined
