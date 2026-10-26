"""What the model sees: small, upright, flattened, metadata-free JPEGs."""

from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageCms, ImageFile, JpegImagePlugin

from asif.infer import preprocess
from asif.infer.preprocess import MAX_PIXELS, ImageDecodeError, prepare_image
from tests.conftest import FIXTURE_EXPECT, requires_exiftool
from tests.fake_llama_server import jpeg_info

DECODABLE = sorted(n for n, e in FIXTURE_EXPECT.items() if e == "ok" and n != "panorama.jpg")
UNDECODABLE = ("truncated.jpg", "corrupt.jpg", "photo.heic", "raw.dng")
WITH_METADATA = ("exif.jpg", "gps.jpg", "user_xmp.jpg", "user_iptc.jpg", "motion.jpg", "mpo.jpg")


def _open(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def _save(tmp_path: Path, name: str, im: Image.Image, fmt: str, **kw: Any) -> Path:
    path = tmp_path / name
    im.save(path, fmt, **kw)
    return path


def _png_header(path: Path, width: int, height: int) -> Path:
    """A PNG that *claims* to be width x height, without the pixels to back it."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b""))
    return path


@pytest.fixture
def draft_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any, Any]]:
    """Records (requested size, decoded size) whenever JPEG draft mode is asked for."""
    calls: list[tuple[Any, Any, Any]] = []
    original = JpegImagePlugin.JpegImageFile.draft

    def spy(self: JpegImagePlugin.JpegImageFile, mode: str | None, size: Any) -> Any:
        before = self.size
        result = original(self, mode, size)
        calls.append((size, before, self.size))
        return result

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy)
    return calls


# -- the happy path --------------------------------------------------------------


@pytest.mark.parametrize("name", DECODABLE)
def test_every_photo_fixture_becomes_a_small_rgb_jpeg(fixtures_dir: Path, name: str) -> None:
    out = _open(prepare_image(fixtures_dir / name, 256))
    assert out.format == "JPEG"
    assert out.mode == "RGB"
    assert max(out.size) == 256  # every fixture is bigger than that


def test_never_enlarges(fixtures_dir: Path) -> None:
    assert _open(prepare_image(fixtures_dir / "plain.jpg", 4096)).size == (640, 480)


def test_caps_the_longest_edge_and_keeps_the_aspect_ratio(fixtures_dir: Path) -> None:
    assert _open(prepare_image(fixtures_dir / "plain.jpg", 320)).size == (320, 240)
    assert _open(prepare_image(fixtures_dir / "rotated.jpg", 320)).size == (320, 240)


def test_extreme_aspect_ratios_keep_at_least_one_pixel(tmp_path: Path) -> None:
    path = _save(tmp_path, "strip.png", Image.new("RGB", (4000, 10), "red"), "PNG")
    assert _open(prepare_image(path, 100)).size == (100, 1)


def test_applies_the_exif_orientation(fixtures_dir: Path) -> None:
    # Stored 480x640 with the sky on top and Orientation=6 (rotate 90 CW to view):
    # upright, it is 640x480 and the sky is on the right.
    out = _open(prepare_image(fixtures_dir / "rotated.jpg", 1024))
    assert out.size == (640, 480)
    r, g, b = out.getpixel((634, 240))
    assert b > 200 and b > g > r  # sky
    r, g, b = out.getpixel((5, 240))
    assert g > r and g > b  # grass


def test_a_broken_exif_block_doesnt_make_the_photo_undecodable(tmp_path: Path) -> None:
    path = tmp_path / "bad-exif.jpg"
    Image.new("RGB", (300, 200), "red").save(path, "JPEG", exif=b"Exif\x00\x00not-a-tiff")
    assert _open(prepare_image(path, 1024)).size == (300, 200)


def test_flattens_alpha_onto_white(fixtures_dir: Path) -> None:
    # alpha.png fades from fully transparent (top) to opaque (bottom).
    out = _open(prepare_image(fixtures_dir / "alpha.png", 1024))
    assert out.getpixel((0, 0)) == (255, 255, 255)
    r, g, b = out.getpixel((0, out.height - 1))
    assert g > r + 40 and g > b + 40  # the opaque grass, not white


@pytest.mark.parametrize(
    ("mode", "fmt"),
    [("RGBA", "PNG"), ("LA", "PNG"), ("RGBA", "WEBP"), ("RGBA", "TIFF")],
)
def test_every_kind_of_transparency_is_flattened(tmp_path: Path, mode: str, fmt: str) -> None:
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0)).convert(mode)
    out = _open(prepare_image(_save(tmp_path, f"t.{fmt.lower()}", im, fmt, lossless=True), 1024))
    assert out.mode == "RGB"
    assert out.getpixel((32, 32)) == (255, 255, 255)


def test_palette_transparency_is_flattened(tmp_path: Path) -> None:
    im = Image.new("P", (64, 64), 0)
    im.putpalette([0, 0, 0] * 256)
    path = _save(tmp_path, "p.png", im, "PNG", transparency=0)
    assert _open(prepare_image(path, 1024)).getpixel((10, 10)) == (255, 255, 255)


@pytest.mark.parametrize(
    ("mode", "colour", "fmt"),
    [
        ("L", 128, "PNG"),
        ("1", 1, "PNG"),
        ("CMYK", (0, 255, 255, 0), "JPEG"),
        ("P", 3, "PNG"),
        ("RGB", (10, 200, 30), "TIFF"),
    ],
)
def test_other_modes_become_rgb(tmp_path: Path, mode: str, colour: Any, fmt: str) -> None:
    path = _save(tmp_path, f"x.{fmt.lower()}", Image.new(mode, (64, 48), colour), fmt)
    out = _open(prepare_image(path, 1024))
    assert (out.mode, out.size) == ("RGB", (64, 48))


def test_16_bit_greyscale_keeps_its_tones(tmp_path: Path) -> None:
    path = _save(tmp_path, "scan.png", Image.new("I;16", (64, 64), 32768), "PNG")
    r, g, b = _open(prepare_image(path, 1024)).getpixel((32, 32))
    assert 120 <= r <= 136 and r == g == b  # mid-grey, not clipped to white


def test_output_is_deterministic(fixtures_dir: Path) -> None:
    assert prepare_image(fixtures_dir / "exif.jpg", 512) == prepare_image(
        fixtures_dir / "exif.jpg", 512
    )


def test_settings_that_keep_truncated_files_failing() -> None:
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False
    assert Image.MAX_IMAGE_PIXELS == MAX_PIXELS


def test_max_edge_must_be_positive(fixtures_dir: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        prepare_image(fixtures_dir / "plain.jpg", 0)


# -- privacy: nothing but pixels ---------------------------------------------------


@pytest.mark.parametrize("name", WITH_METADATA)
def test_output_carries_no_metadata(fixtures_dir: Path, name: str) -> None:
    data = prepare_image(fixtures_dir / name, 512)
    im = _open(data)
    assert not im.getexif()
    for key in ("exif", "icc_profile", "xmp", "XML:com.adobe.xmp", "comment", "mp"):
        assert key not in im.info
    _w, _h, markers = jpeg_info(data)
    assert [m for m in markers if 0xE1 <= m <= 0xEF or m == 0xFE] == []
    assert data.endswith(b"\xff\xd9")  # a motion-photo trailer doesn't ride along


@requires_exiftool
@pytest.mark.parametrize("name", ["exif.jpg", "gps.jpg", "user_xmp.jpg", "user_iptc.jpg"])
def test_exiftool_finds_no_metadata_in_the_output(
    fixtures_dir: Path, tmp_path: Path, name: str
) -> None:
    out = tmp_path / "out.jpg"
    out.write_bytes(prepare_image(fixtures_dir / name, 512))
    proc = subprocess.run(
        [shutil.which("exiftool") or "exiftool", "-j", "-a", "-G0", str(out)],
        capture_output=True,
        check=True,
    )
    groups = {key.split(":", 1)[0] for key in json.loads(proc.stdout)[0] if ":" in key}
    assert groups <= {"ExifTool", "File", "JFIF", "Composite"}


def test_comments_and_icc_profiles_are_dropped(tmp_path: Path) -> None:
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = _save(
        tmp_path,
        "c.jpg",
        Image.new("RGB", (64, 64), "blue"),
        "JPEG",
        comment=b"owner's secret note",
        icc_profile=icc,
    )
    assert "comment" in Image.open(path).info  # the input really has one
    data = prepare_image(path, 1024)
    assert b"secret" not in data
    assert "icc_profile" not in _open(data).info


# -- draft mode ----------------------------------------------------------------------


def test_big_jpegs_decode_in_draft_mode(
    tmp_path: Path, draft_calls: list[tuple[Any, Any, Any]]
) -> None:
    path = _save(tmp_path, "big.jpg", Image.new("RGB", (4000, 3000), "green"), "JPEG")
    out = _open(prepare_image(path, 1024))
    assert out.size == (1024, 768)
    assert len(draft_calls) == 1
    requested, before, after = draft_calls[0]
    assert requested == (1024, 768)
    assert before == (4000, 3000)
    assert after == (2000, 1500)  # decoded at half size, still >= the target


def test_small_jpegs_dont_need_draft_mode(
    fixtures_dir: Path, draft_calls: list[tuple[Any, Any, Any]]
) -> None:
    prepare_image(fixtures_dir / "plain.jpg", 1024)
    assert draft_calls == []


def test_draft_mode_respects_orientation(
    tmp_path: Path, draft_calls: list[tuple[Any, Any, Any]]
) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6
    path = _save(tmp_path, "r.jpg", Image.new("RGB", (3000, 4000), "red"), "JPEG", exif=exif)
    assert _open(prepare_image(path, 500)).size == (500, 375)
    # Draft mode works on the stored (sideways) frame: an exact 1/8 here.
    assert draft_calls == [((375, 500), (3000, 4000), (375, 500))]


@pytest.mark.slow
def test_the_50_mp_panorama_decodes_at_an_eighth(
    panorama: Path, draft_calls: list[tuple[Any, Any, Any]]
) -> None:
    out = _open(prepare_image(panorama, 1024))
    assert out.size == (1024, 512)
    assert draft_calls[0][1:] == ((10000, 5000), (1250, 625))


# -- failures ----------------------------------------------------------------------


@pytest.mark.parametrize("name", UNDECODABLE)
def test_undecodable_fixtures_raise_image_decode_error(fixtures_dir: Path, name: str) -> None:
    with pytest.raises(ImageDecodeError) as info:
        prepare_image(fixtures_dir / name, 1024)
    assert isinstance(info.value, ValueError)
    assert name not in str(info.value)  # no file names in error messages
    assert str(fixtures_dir) not in str(info.value)


@pytest.mark.parametrize("keep", [100, 0.5, -2, -200])
def test_any_truncated_jpeg_fails(fixtures_dir: Path, tmp_path: Path, keep: float) -> None:
    data = (fixtures_dir / "progressive.jpg").read_bytes()
    cut = int(len(data) * keep) if isinstance(keep, float) else keep
    path = tmp_path / "cut.jpg"
    path.write_bytes(data[:cut])
    with pytest.raises(ImageDecodeError):
        prepare_image(path, 1024)


def test_truncated_png_fails(fixtures_dir: Path, tmp_path: Path) -> None:
    data = (fixtures_dir / "plain.png").read_bytes()
    path = tmp_path / "cut.png"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ImageDecodeError, match="truncated"):
        prepare_image(path, 1024)


def test_empty_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.jpg"
    path.write_bytes(b"")
    with pytest.raises(ImageDecodeError):
        prepare_image(path, 1024)


def test_a_missing_file_is_an_os_error_not_a_decode_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_image(tmp_path / "nope.jpg", 1024)


@pytest.mark.parametrize(("width", "height"), [(30_000, 30_000), (25_000, 20_000)])
def test_too_many_pixels_is_refused_before_decoding(
    tmp_path: Path, width: int, height: int
) -> None:
    # 900 MP is past Pillow's hard limit; 500 MP is in the band where it only warns.
    path = _png_header(tmp_path / "huge.png", width, height)
    with pytest.raises(ImageDecodeError, match="pixels"):
        prepare_image(path, 1024)


def test_our_own_pixel_limit_applies(fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preprocess, "MAX_PIXELS", 640 * 480 - 1)
    with pytest.raises(ImageDecodeError, match="pixels"):
        prepare_image(fixtures_dir / "plain.jpg", 1024)
