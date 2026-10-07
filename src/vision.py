from pathlib import Path

import ollama


VISION_MODEL = "qwen3-vl:4b-instruct"


def extract_page_content(
    image_path: str,
    page_number: int,
    source_name: str
) -> str:
    """
    Extract useful academic content from a PDF page
    using Qwen3-VL.
    """

    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(
            f"Image not found: {image_file}"
        )

    prompt = f"""
Extract the important academic content from this
physics textbook page for a RAG knowledge base.

Document: {source_name}
Page: {page_number}

Rules:
- Preserve formulas and equations accurately.
- Preserve subscripts, superscripts, Greek symbols,
  numerical values, and units.
- Use readable mathematical notation or LaTeX.
- Preserve definitions, headings, and important text.
- Put equations on separate lines.
- Do not summarize.
- Do not explain.
- Do not invent missing information.
- Return only the extracted content.

Example:
I = mr²

Example:
I = \\sum_i m_i r_i^2
"""

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_file)]
                }
            ],
            options={
                "num_ctx": 8192
            }
        )

    except Exception as exc:
        raise RuntimeError(
            f"Vision model failed while processing "
            f"page {page_number}: {exc}"
        ) from exc

    result = response["message"]["content"].strip()

    if not result:
        raise RuntimeError(
            f"Vision model returned empty content "
            f"for page {page_number}."
        )

    return result