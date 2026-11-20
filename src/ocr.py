"""Optional Tesseract-based OCR helpers.

Used by the experimental scripts in experiments/.
The main pipeline prefers normal PDF text extraction
and (optionally) the vision model; Tesseract is a
fallback for image-only pages.
"""

import io
import os
import shutil

import pymupdf
import pytesseract

from PIL import Image, ImageEnhance, ImageFilter


def _configure_tesseract():
    """Locate the Tesseract binary without hardcoding a path."""

    cmd = os.environ.get("TESSERACT_CMD")

    if not cmd:
        cmd = shutil.which("tesseract")

    if not cmd:
        default_windows = (
            r"C:\Program Files\Tesseract-OCR"
            r"\tesseract.exe"
        )
        if os.path.exists(default_windows):
            cmd = default_windows

    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    return cmd


TESSERACT_CMD = _configure_tesseract()


def render_page(page, scale=4):
    """
    Render PDF page as a high-resolution image.
    """

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        alpha=False
    )

    image_bytes = pixmap.tobytes("png")

    return Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


def preprocess_image(image):
    """
    Improve contrast and sharpness for OCR.
    """

    gray = image.convert("L")

    gray = ImageEnhance.Contrast(
        gray
    ).enhance(2.0)

    gray = ImageEnhance.Sharpness(
        gray
    ).enhance(2.0)

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


def run_ocr(image, psm=6):

    if not TESSERACT_CMD:
        raise RuntimeError(
            "Tesseract is not installed. Install it "
            "or set the TESSERACT_CMD environment "
            "variable."
        )

    return pytesseract.image_to_string(
        image,
        config=f"--psm {psm}"
    ).strip()


def ocr_page(page):

    image = render_page(
        page,
        scale=4
    )

    processed = preprocess_image(
        image
    )

    results = []

    # Different page segmentation modes
    # can produce different results.
    for psm in [6, 11, 12]:

        text = run_ocr(
            processed,
            psm=psm
        )

        if text:
            results.append(text)

    # Prefer the longest useful extraction.
    if results:

        results.sort(
            key=len,
            reverse=True
        )

        return results[0]

    return ""
