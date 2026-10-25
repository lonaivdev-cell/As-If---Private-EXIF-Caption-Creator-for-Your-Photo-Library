"""The integrity gates, against real exiftool writes on the generated fixtures.

``legit_write`` performs the same kind of write As If's writer does (XMP
caption, subjects, provenance, original-image hash; IPTC keywords with
compat "wide"). Every gate must pass for it on every writable fixture, and
each gate must fail when the thing it guards is tampered with afterwards.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import os
import struct
import time
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image, ImageCms, PngImagePlugin

from asif.meta.exiftool import ExecResult, ExifTool, ExifToolError
from asif.meta.gates import (
    MAX_GROWTH,
    Fingerprint,
    GateResult,
    _allowed,
    all_ok,
    check_source_unchanged,
    check_written,
    fingerprint,
    snapshot,
)
from asif.meta.structure import jpeg_eoi_offset
from tests.conftest import HAVE_EXIFTOOL

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
) -> None:
    """Write like As If's writer: only XMP-dc, XMP-asif, XMP-et (and IPTC if wide)."""
    args = [
        f"-XMP-dc:Description-x-default={CAPTION}",
        f"-XMP-dc:Description-pt-BR={CAPTION}",
        *(f"-XMP-dc:Subject+={k}" for k in keywords),
        "-XMP-asif:ToolVersion=0.1.0",
        "-XMP-asif:ModelId=stub",
        "-XMP-asif:ModelSHA256=",
        "-XMP-asif:PromptVersion=caption_v1",
        "-XMP-asif:RunId=20261025-abcdef",
        "-XMP-asif:Timestamp=2026:10:25 10:00:00Z",
        f"-XMP-asif:CaptionDigest={hashlib.sha256(CAPTION.encode()).hexdigest()}",
        "-XMP-asif:CaptionLang=pt-BR",
        *(f"-XMP-asif:AddedKeywords={k}" for k in keywords),
        "-XMP-asif:WroteOriginalImageHash=True",
        f"-XMP-et:OriginalImageHash={image_hash}",
        "-XMP-et:OriginalImageHashType=SHA256",
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
    assert (gates["structure"].detail == "n/a") == (before.fmt == "tiff")
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
    assert all_ok(check_written(et, before, path, compat="wide"))


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
        "IFD1:ThumbnailOffset",
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
        "IFD1:StripOffsets",
    ]
    for key in protected:
        for variant in (key, f"{key}#2"):
            for fmt in ("jpeg", "png", "webp", "tiff", "other"):
                for compat in ("xmp", "wide"):
                    assert not _allowed(variant, fmt, compat), (variant, fmt, compat)  # type: ignore[arg-type]
    # A WebP's size is reported twice once it has a VP8X header; the first copy counts.
    assert not _allowed("RIFF:ImageWidth", "webp", "xmp")
    assert not _allowed("RIFF:ImageHeight", "webp", "wide")


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
