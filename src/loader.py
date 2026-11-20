"""Document loading: extract text from PDF and TXT files.

PDF pages are first read with normal text extraction. Pages whose
extracted text looks broken or incomplete can optionally be passed to a
local vision model (via Ollama) for a second, image-based extraction.
The vision step is optional: if Ollama or the vision model is not
available, the loader falls back to the normal text.
"""

import os
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = PROJECT_ROOT / "data" / ".tmp"


def text_quality_score(text: str) -> float:
    """
    Estimate how useful the extracted PDF text is.

    This is intentionally generic so it works with
    different PDFs rather than relying on filenames
    or specific page numbers.
    """

    if not text:
        return 0.0

    stripped = text.strip()

    if not stripped:
        return 0.0

    score = 1.0

    # Very little text is suspicious.
    if len(stripped) < 100:
        score -= 0.50
    elif len(stripped) < 250:
        score -= 0.25

    # Very few words is suspicious.
    word_count = len(stripped.split())

    if word_count < 20:
        score -= 0.30
    elif word_count < 40:
        score -= 0.15

    # Lots of isolated fragments can indicate
    # broken PDF extraction.
    lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip()
    ]

    if lines:

        short_lines = sum(
            1
            for line in lines
            if len(line.split()) <= 2
        )

        short_ratio = (
            short_lines / len(lines)
        )

        if short_ratio > 0.60:
            score -= 0.25

    # Suspiciously incomplete endings.
    lower = stripped.lower().rstrip()

    suspicious_endings = (
        "at a distance",
        "from the",
        "where",
        "given by",
        "is",
        "are",
        "of",
        "the",
        "and",
    )

    if lower.endswith(suspicious_endings):
        score -= 0.25

    return max(0.0, min(1.0, score))


def page_needs_vision(page, text: str) -> bool:
    """
    Decide automatically whether a page should be
    processed by the vision model.
    """

    score = text_quality_score(text)

    # Strong indication that normal extraction failed.
    if score < 0.65:
        return True

    # Check whether the page contains images.
    image_count = len(page.get_images(full=True))

    # If the page contains visual content and text
    # quality isn't excellent, vision may help.
    if image_count > 0 and score < 0.85:
        return True

    return False


def render_page(page, scale: float = 2.5):
    """
    Render a PDF page to a PNG-compatible pixmap.
    """

    return page.get_pixmap(
        matrix=pymupdf.Matrix(
            scale,
            scale
        ),
        alpha=False
    )


def _vision_available() -> bool:
    """Vision extraction needs the ollama package and a running server."""
    if os.environ.get("RAG_DISABLE_VISION"):
        return False
    try:
        import ollama  # noqa: F401
    except ImportError:
        return False
    return True


def load_pdf(path: str, use_vision: bool = True, verbose: bool = True):

    document = pymupdf.open(path)

    source_name = Path(path).name
    total_pages = len(document)

    pages = []

    if verbose:
        print(f"\nProcessing: {source_name}")
        print(f"Total pages: {total_pages}\n")

    vision_warned = False

    for page_number, page in enumerate(
        document,
        start=1
    ):

        if verbose:
            print(
                f"[{page_number}/{total_pages}] "
                f"Extracting text...",
                flush=True
            )

        normal_text = (
            page.get_text("text")
            .strip()
        )

        quality = text_quality_score(
            normal_text
        )

        if verbose:
            print(
                f"    Text quality: "
                f"{quality:.2f}",
                flush=True
            )

        final_text = normal_text
        extraction_method = "text"

        if use_vision and page_needs_vision(
            page,
            normal_text
        ):

            if not _vision_available() and not vision_warned:
                if verbose:
                    print(
                        "    -> Vision extraction "
                        "unavailable (ollama not "
                        "installed); using normal "
                        "text extraction",
                        flush=True
                    )
                vision_warned = True

            elif _vision_available():

                if verbose:
                    print(
                        "    -> Visual extraction needed",
                        flush=True
                    )

                TEMP_DIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                image_path = (
                    TEMP_DIR / "page.png"
                )

                pixmap = render_page(
                    page
                )

                pixmap.save(
                    str(image_path)
                )

                try:

                    # Imported lazily so that text-only
                    # use never requires the vision
                    # model or a running Ollama server.
                    from vision import (
                        extract_page_content
                    )

                    vision_text = (
                        extract_page_content(
                            str(image_path),
                            page_number,
                            source_name
                        )
                    )

                    if vision_text.strip():

                        final_text = vision_text
                        extraction_method = "vision"

                        if verbose:
                            print(
                                "    -> Vision "
                                "extraction "
                                "complete",
                                flush=True
                            )

                except Exception as exc:

                    if verbose:
                        print(
                            f"    -> Vision failed: "
                            f"{exc}",
                            flush=True
                        )

                        print(
                            "    -> Using normal "
                            "extraction",
                            flush=True
                        )

                finally:

                    if image_path.exists():

                        image_path.unlink()

        else:

            if verbose:
                print(
                    "    -> Normal extraction "
                    "sufficient",
                    flush=True
                )

        if final_text.strip():

            pages.append({
                "text": final_text,
                "page": page_number,
                "source": source_name,
                "extraction_method":
                    extraction_method
            })

    document.close()

    if verbose:
        print(f"\nFinished: {source_name}")

    return pages


def load_txt(path: str):

    text = Path(path).read_text(
        encoding="utf-8"
    )

    return [{
        "text": text,
        "page": 1,
        "source": Path(path).name,
        "extraction_method": "text"
    }]


def load_document(path: str, use_vision: bool = True, verbose: bool = True):
    """Load a document as a list of page dicts.

    Each page dict has: text, page, source, extraction_method.
    """

    extension = Path(path).suffix.lower()

    if extension == ".pdf":
        return load_pdf(path, use_vision=use_vision, verbose=verbose)

    if extension == ".txt":
        return load_txt(path)

    raise ValueError(
        f"Unsupported file type: {extension}"
                )
