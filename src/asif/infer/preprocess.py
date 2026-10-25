"""Turn a photo into the small, upright JPEG the model actually sees.

Runs inside the inference worker. The output is a fresh JPEG with no
metadata at all (no EXIF, GPS, XMP or ICC profile): the model gets pixels
and nothing else. Its longest edge is capped because, on CPU, the vision
encoder's cost grows with resolution and dominates per-image time (§9.3 of
docs/DESIGN.md).

A file that can't be decoded *completely* raises ``ImageDecodeError``, never
a bare Pillow exception: a truncated photo must fail, not be captioned from
whatever half of it arrived (``ImageFile.LOAD_TRUNCATED_IMAGES`` stays False).
"""

from __future__ import annotations

import io
import math
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 400_000_000
"""Big phone panoramas are legitimate; anything above this is refused."""
JPEG_QUALITY = 90
_WHITE = (255, 255, 255)

# Pillow's default bomb limit (~89 MP) would refuse real panoramas. Raise it
# to our own limit, which prepare_image also checks explicitly.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageDecodeError(ValueError):
    """The image can't be decoded (corrupt, truncated, unsupported, too big)."""


def prepare_image(path: Path, max_edge_px: int) -> bytes:
    """Decode ``path`` and return it as an upright, metadata-free RGB JPEG.

    Large JPEGs decode in draft mode, at the smallest 1/2, 1/4 or 1/8 scale
    that still covers the target size (a big CPU win for 50 MP panoramas).
    The EXIF orientation is applied, alpha is flattened onto white, and the
    longest edge is shrunk to ``max_edge_px`` (never enlarged).

    Raises ``ImageDecodeError`` if the file isn't a complete, decodable image
    of at most ``MAX_PIXELS`` pixels, and ``OSError`` if it can't be read at all.
    """
    if max_edge_px < 1:
        raise ValueError("max_edge_px must be positive")
    with path.open("rb") as fh:
        try:
            im = _decode(fh, max_edge_px)
        except ImageDecodeError:
            raise
        except Exception as exc:  # untrusted input: Pillow's plugins raise all sorts
            raise ImageDecodeError(f"can't decode the image: {_why(exc)}") from exc

    # Rebuild on a blank canvas so nothing from the original (EXIF, GPS, XMP,
    # ICC, comments) can ride along into the bytes we hand to the model.
    out = Image.new("RGB", im.size, _WHITE)
    out.paste(im, mask=im.getchannel("A") if im.mode == "RGBA" else None)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _decode(fh: io.BufferedReader, max_edge_px: int) -> Image.Image:
    """Fully decode, orient and shrink. Returns an ``RGB`` or ``RGBA`` image."""
    with warnings.catch_warnings():
        # Between MAX_IMAGE_PIXELS and twice that, Pillow only warns. Refuse instead.
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        im = Image.open(fh)
    with im:
        width, height = im.size
        if width * height > MAX_PIXELS:
            raise ImageDecodeError(f"can't decode the image: more than {MAX_PIXELS:,} pixels")
        if im.format in ("JPEG", "MPO") and max(width, height) > max_edge_px:
            scale = max_edge_px / max(width, height)
            im.draft("RGB", (math.ceil(width * scale), math.ceil(height * scale)))
        im.load()  # decodes everything; raises on truncated data
        try:
            upright = ImageOps.exif_transpose(im)
        except Exception:
            # A broken EXIF block isn't a broken image: describe it as stored.
            upright = im.copy()

    if upright.has_transparency_data:
        upright = upright.convert("RGBA")
    elif upright.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
        # 16-bit greyscale (scans, some TIFFs and PNGs): a plain convert would
        # clip almost everything to white.
        upright = upright.convert("I").point(lambda v: v * (1 / 256)).convert("RGB")
    else:
        upright = upright.convert("RGB")
    upright.thumbnail((max_edge_px, max_edge_px), Image.Resampling.LANCZOS)
    return upright


def _why(exc: BaseException) -> str:
    """A short reason that never includes the file's name or contents."""
    if isinstance(exc, UnidentifiedImageError):
        return "not an image format Pillow can read"
    if isinstance(exc, Image.DecompressionBombError | Image.DecompressionBombWarning):
        return f"more than {MAX_PIXELS:,} pixels"
    if isinstance(exc, OSError) and "truncated" in str(exc):
        return "the file is truncated"
    return f"decoder error ({type(exc).__name__})"
