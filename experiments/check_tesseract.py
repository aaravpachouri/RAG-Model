import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src")
)

from ocr import TESSERACT_CMD  # noqa: E402

import pytesseract  # noqa: E402

if not TESSERACT_CMD:
    raise SystemExit(
        "Tesseract not found. Install it or set TESSERACT_CMD."
    )

print("Tesseract version:")
print(pytesseract.get_tesseract_version())

print("\nOCR is connected successfully!")
