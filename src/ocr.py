import io
import re

import pymupdf
import pytesseract

from PIL import Image, ImageEnhance, ImageFilter


TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


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