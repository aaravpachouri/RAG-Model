"""Experimental: locate a formula near a heading and OCR that region."""

import io

import pymupdf
import pytesseract

from PIL import Image, ImageEnhance, ImageFilter

from ocr import TESSERACT_CMD


def get_page_image(page):

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(5, 5),
        alpha=False
    )

    return Image.open(
        io.BytesIO(
            pixmap.tobytes("png")
        )
    ).convert("RGB")


def preprocess(image):

    image = image.convert("L")

    image = ImageEnhance.Contrast(
        image
    ).enhance(2.5)

    image = ImageEnhance.Sharpness(
        image
    ).enhance(2.5)

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


def extract_formula_region(page):

    """
    Find the phrase:
    'For a particle at a distance'
    and OCR the area around/below it.
    """

    if not TESSERACT_CMD:
        raise RuntimeError(
            "Tesseract is not installed. Install it "
            "or set the TESSERACT_CMD environment "
            "variable."
        )

    image = get_page_image(
        page
    )

    data = pytesseract.image_to_data(
        image,
        config="--psm 11",
        output_type=pytesseract.Output.DICT
    )

    target = [
        "for",
        "a",
        "particle",
        "at",
        "a",
        "distance"
    ]

    words = [
        word.lower().strip()
        for word in data["text"]
        if word.strip()
    ]

    # Find approximate occurrence.
    found = False
    target_index = -1

    for i in range(
        len(words) - len(target) + 1
    ):

        if words[
            i:i + len(target)
        ] == target:

            found = True
            target_index = i
            break

    if not found:

        print(
            "Formula heading not located."
        )

        return ""

    # Get coordinate of the matching phrase.
    valid_indices = []

    for i, text in enumerate(
        data["text"]
    ):

        if text.strip():

            valid_indices.append(i)

    if target_index >= len(
        valid_indices
    ):

        return ""

    start_idx = valid_indices[
        target_index
    ]

    x = data["left"][start_idx]
    y = data["top"][start_idx]

    w = data["width"][start_idx]
    h = data["height"][start_idx]

    # Crop the area below/right of the heading.
    left = max(0, x - 100)
    top = max(0, y)
    right = min(
        image.width,
        x + w + 1400
    )
    bottom = min(
        image.height,
        y + h + 700
    )

    crop = image.crop(
        (left, top, right, bottom)
    )

    crop = preprocess(
        crop
    )

    # Try multiple OCR modes.
    outputs = []

    for psm in [6, 11, 12, 13]:

        text = pytesseract.image_to_string(
            crop,
            config=f"--psm {psm}"
        ).strip()

        if text:
            outputs.append(text)

    outputs.sort(
        key=len,
        reverse=True
    )

    return outputs[0] if outputs else ""
