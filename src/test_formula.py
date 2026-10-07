from pathlib import Path

import pymupdf

from formula_ocr import extract_formula_region


PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "documents"
    / "XI_PHY_ROTATIONAL_NOTES.pdf"
)


def main():

    print("=" * 60)
    print("FORMULA OCR TEST")
    print("=" * 60)

    document = pymupdf.open(
        PDF_PATH
    )

    # Page 10 = index 9
    page = document[9]

    formula = extract_formula_region(
        page
    )

    print("\nFORMULA REGION OCR:")
    print("-" * 60)
    print(formula)
    print("-" * 60)

    print(
        "\nImage saved to:"
        "\ndata/formula_region.png"
    )

    document.close()


if __name__ == "__main__":
    main()