import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

print("Tesseract version:")
print(pytesseract.get_tesseract_version())

print("\nOCR is connected successfully!")