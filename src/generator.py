"""Answer generation from retrieved context using a local Ollama model."""

import os

# The project default is the vision-capable Qwen model
# the author has installed (seen in his `ollama list`,
# November 2026): one model then covers answer
# generation and page-image extraction.
# scripts/check_setup.py detects the installed tags on
# the machine and prints the right RAG_MODEL value.
DEFAULT_MODEL = "qwen3-vl:4b-instruct"


def build_prompt(question: str, documents) -> str:
    """Build the generation prompt from retrieved chunks."""

    context_parts = []

    for i, document in enumerate(
        documents,
        start=1
    ):

        metadata = document.get("metadata", {})

        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")

        context_parts.append(
            f"SOURCE {i}\n"
            f"File: {source}\n"
            f"Page: {page}\n\n"
            f"{document['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a study assistant that answers
questions strictly from the CONTEXT below.

CONTEXT:

{context}

RULES:

1. Use only information from the CONTEXT.
2. If the CONTEXT does not contain the
   answer, do not use your own knowledge.
   Write exactly: NOT IN DOCUMENTS
3. For formulas, copy the formula exactly
   as it appears in the CONTEXT.
4. Ignore parts of the CONTEXT that do
   not relate to the question.
5. Copy the exact sentence(s) from the
   CONTEXT that your answer relies on.
   Do not change any words.

QUESTION: {question}

Answer in exactly this format:

ANSWER: <your answer, or NOT IN DOCUMENTS>
QUOTE: <exact sentence(s) copied from the
        CONTEXT, or NONE>
SOURCE: <file name and page number from
         the CONTEXT, or NONE>
"""

    return prompt


def parse_answer(raw: str):
    """Split a model reply into (answer, citation).

    Returns the raw text as the answer when the model
    did not follow the requested format.
    """

    citation = None

    not_found_markers = (
        "not in documents",
        "not in the documents",
        "not in the document",
        "not in document",
    )

    if "SOURCE:" in raw:
        answer_part, _, source_part = raw.partition(
            "SOURCE:"
        )
        source_lines = source_part.strip().splitlines()
        citation = source_lines[0].strip() if source_lines else None

        if citation and citation.upper() in ("NONE", "N/A", "-"):
            citation = None
    elif "ANSWER:" in raw:
        answer_part = raw.partition("ANSWER:")[2]
    else:
        answer_part = raw

    # The displayed answer never keeps the model's
    # scaffolding: the ANSWER: label comes off, and the
    # QUOTE: section is shown separately by the caller,
    # under the evidence label, only when it verifies.
    answer = answer_part.partition("QUOTE:")[0]
    answer = answer.replace("ANSWER:", "").strip()

    # When the model reports the answer is missing,
    # normalise the message and never show a source:
    # small models sometimes invent one here.
    lowered = answer.lower()
    if any(marker in lowered for marker in not_found_markers):
        answer = (
            "I could not find this information "
            "in the indexed documents."
        )
        citation = None

    if not answer:
        answer = (
            "The model returned no answer. "
            "Check the retrieved passages directly."
        )
        citation = None

    return answer, citation


def extract_quote(raw: str):
    """Pull the QUOTE line out of a model reply."""

    if "QUOTE:" not in raw:
        return None

    quote_part = raw.partition("QUOTE:")[2]

    # The quote ends where SOURCE: begins, if present.
    quote_part = quote_part.partition("SOURCE:")[0]

    quote = quote_part.strip()

    # Models often wrap the copied sentence in
    # quotation marks; strip one matching pair so
    # the exact-match check sees the copied text.
    if len(quote) >= 2 and (
        quote[0] == quote[-1]
    ) and quote[0] in ("'", '"', "\u201c"):
        quote = quote[1:-1].strip()

    if not quote or quote.upper() in (
        "NONE", "N/A", "-"
    ):
        return None

    return quote


def verify_quote(quote, documents, min_chars=30):
    """Check a quote against the retrieved chunks.

    Returns the chunk dict the quote was found in, or
    None. Whitespace is normalised on both sides so
    line wrapping does not matter; anything else must
    match exactly, because the instruction is to copy.
    A very short quote proves nothing, so it never
    verifies.
    """

    if not quote:
        return None

    normalised = " ".join(quote.split())

    if len(normalised) < min_chars:
        return None

    for document in documents:
        text = " ".join(
            document["text"].split()
        )

        if normalised in text:
            return document

    return None


def validate_citation(citation, documents):
    """Keep a citation only if it matches retrieval.

    Small local models sometimes invent a plausible-
    looking source. A citation is kept only when the
    file it names was actually retrieved AND, when it
    states a page number, that page was retrieved for
    that file. Anything else is dropped rather than
    shown.
    """

    if not citation:
        return None

    lowered = citation.lower()

    import re

    page_match = re.search(
        r"page[:\s]*([0-9]+)", lowered
    )
    cited_page = (
        int(page_match.group(1))
        if page_match
        else None
    )

    for document in documents:
        metadata = document.get("metadata", {})

        source = str(
            metadata.get("source", "")
        ).lower()

        if not source or source not in lowered:
            continue

        if cited_page is None:
            return citation

        if cited_page == metadata.get("page"):
            return citation

    # Either no retrieved file matches the cited
    # name, or the cited page was not among the
    # retrieved pages: not trustworthy.
    return None


class Generator:

    def __init__(
        self,
        model: str = None
    ):

        self.model = (
            model
            or os.environ.get("RAG_MODEL")
            or DEFAULT_MODEL
        )

    def generate(
        self,
        question: str,
        documents
    ):
        """
        Generate an answer. documents is a list of
        retrieved chunk dicts (each with text and
        metadata).
        """

        import ollama

        prompt = build_prompt(
            question,
            documents
        )

        try:

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        except Exception as exc:
            raise RuntimeError(
                f"Could not generate an answer with "
                f"Ollama model '{self.model}'. Make "
                f"sure Ollama is running and the "
                f"model is pulled: "
                f"`ollama pull {self.model}`. "
                f"Original error: {exc}"
            ) from exc

        raw = response["message"]["content"]

        answer, citation = parse_answer(raw)

        quote = extract_quote(raw)

        quote_doc = verify_quote(quote, documents)

        citation = validate_citation(
            citation, documents
        )

        if quote_doc:
            # The model copied a passage that really is
            # in the retrieved context. This verifies the
            # quote's presence, not that it supports the
            # answer, so the answer still asks to be
            # checked against it.
            metadata = quote_doc.get("metadata", {})
            shown = " ".join(quote.split())
            return (
                f"{answer}\n\n"
                f"Evidence (verified quote from "
                f"{metadata.get('source', 'unknown')}"
                f", page "
                f"{metadata.get('page', '?')} "
                f"- check the answer against it):\n"
                f'"{shown}"'
            )

        if citation:
            # The model cited a retrieved file and
            # page; the citation was verified against
            # the retrieved set.
            return f"{answer}\n\nSource: {citation}"

        if documents:

            # The model gave an answer but no usable
            # citation. Point at the top-ranked
            # retrieved chunk as context. This is NOT
            # a verified citation: the chunk was the
            # most relevant context retrieved, but it
            # may not be the passage the answer
            # actually relies on.
            abstain_markers = (
                "could not find",
                "not in documents",
                "not in the documents",
                "not in the document",
            )

            if not any(
                m in answer.lower()
                for m in abstain_markers
            ):
                top = documents[0].get(
                    "metadata", {}
                )
                return (
                    f"{answer}\n\n"
                    f"Reference (top retrieved "
                    f"context, unverified): "
                    f"{top.get('source', 'unknown')}"
                    f", page "
                    f"{top.get('page', '?')}"
                )

        return answer
