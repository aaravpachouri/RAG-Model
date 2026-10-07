from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image


# Tesseract location
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PDF path
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "XI_PHY_ROTATIONAL_NOTES.pdf"
)


def main():

    print("=" * 60)
    print("PAGE 10 OCR TEST")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"\nPDF not found:")
        print(PDF_PATH)
        return

    document = pymupdf.open(PDF_PATH)

    # PDF pages are zero-indexed,
    # so Page 10 = index 9.
    page = document[9]

    print("\nRendering Page 10...")

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        alpha=False
    )

    image_path = (
        PROJECT_ROOT
        / "data"
        / "page10_ocr.png"
    )

    pixmap.save(str(image_path))

    print(f"Page image saved to:")
    print(image_path)

    image = Image.open(image_path)

    print("\nRunning OCR...\n")

    text = pytesseract.image_to_string(
        image,
        config="--psm 6"
    )

    print("=" * 60)
    print("OCR RESULT")
    print("=" * 60)

    print(text)

    print("=" * 60)

    document.close()


if __name__ == "__main__":
    main()