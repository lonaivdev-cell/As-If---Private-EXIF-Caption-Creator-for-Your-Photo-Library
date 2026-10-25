"""Generate the synthetic fixture photos (§16, pass 3 of docs/DESIGN.md).

Nothing here is a real photo: every file is drawn by Pillow and tagged by
exiftool, so the fixture set is reproducible and contains nobody's data.

    python tests/fixtures/make_fixtures.py OUTPUT_DIR [--with-panorama]

Each fixture's purpose is in ``FIXTURES`` below. ``expect`` is what the
pipeline must do with it: ``ok`` (captioned), ``quarantine`` (refused before
anything is written) or ``skip-format`` (a format As If doesn't write yet).
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import subprocess
import sys
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
