import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src")
)

from pathlib import Path

import pymupdf

from vision import extract_page_content


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "XI_PHY_ROTATIONAL_NOTES.pdf"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "page10_vision.png"
)


def main():

    print("=" * 60)
    print("VISION MODEL PDF TEST")
    print("=" * 60)

    document = pymupdf.open(
        PDF_PATH
    )

    # Page 10 = index 9
    page = document[9]

    print("\nRendering Page 10...")

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        alpha=False
    )

    pixmap.save(
        str(IMAGE_PATH)
    )

    document.close()

    print(
        f"Image saved to:\n{IMAGE_PATH}"
    )

    print(
        "\nSending page to vision model..."
    )

    result = extract_page_content(
        str(IMAGE_PATH),
        10,
        "XI_PHY_ROTATIONAL_NOTES.pdf"
    )

    print("\n" + "=" * 60)
    print("VISION MODEL RESULT")
    print("=" * 60)

    print(result)

    print("=" * 60)


if __name__ == "__main__":
    main()
