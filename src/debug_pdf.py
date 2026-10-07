from pathlib import Path
import pymupdf


PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "documents"
    / "XI_PHY_ROTATIONAL_NOTES.pdf"
)


document = pymupdf.open(PDF_PATH)

print("=" * 60)
print("PDF EXTRACTION TEST")
print("=" * 60)

for page_number, page in enumerate(document, start=1):

    text = page.get_text("text")

    print(f"\n{'=' * 60}")
    print(f"PAGE {page_number}")
    print(f"{'=' * 60}")

    print(text[:5000])

document.close()