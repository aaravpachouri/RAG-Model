"""Evaluate retrieval and generation on the indexed documents.

Run after ingestion:

    python scripts/evaluate.py                  # retrieval only
    python scripts/evaluate.py --generate       # also run the LLM
    python scripts/evaluate.py --questions my_questions.json

Two question sources:

- Default: discover Q&A-style headings in the indexed chunks
  (lines like "39 HOW MANY MEMBERS ARE REQUIRED?"). The page a
  heading sits on is the expected page. This suits FAQ/handbook
  notes; other material should use a questions file.
- --questions FILE: JSON list of {"question": ..., "page": N}
  entries with pages you verified yourself. This is the
  stricter check.

Retrieval is scored exactly: did the expected page rank first /
in the top 4? Auto-discovered questions come from the same
document being searched, so that score is a same-document
sanity check of retrieval, not independent held-out accuracy.
Generation cannot be auto-marked, so the script reports
mechanical signals only: how often the answer carried a
verified quote, a validated citation, an unverified reference,
or an abstention. A verified quote is not a correctness mark.
Read the printed answers to judge quality.
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embeddings import EmbeddingModel
from vectorstore import VectorStore
from retriever import Retriever
from reranker import Reranker
from generator import Generator


HEADING = re.compile(
    r"(?:^|\s)(\d{1,3})\s+([A-Z][A-Z0-9 ,/&()'’-]{8,}"
    r"\?(?:[A-Z0-9 ,/&()'’-]*[A-Z?])?)"
)


def discover_questions(documents):
    """Pull (question, page) pairs from Q&A-style headings."""

    pairs = []

    seen = set()

    for document in documents:
        text = " ".join(document["text"].split())

        page = document["metadata"]["page"]

        for match in HEADING.finditer(text):
            question = match.group(2).strip()

            if question in seen:
                continue

            seen.add(question)

            pairs.append({
                "question": question.title(),
                "page": page,
                "source": document["metadata"]["source"],
            })

    return pairs


def load_store_documents(vector_store):
    """Read every chunk back out of the index."""

    data = vector_store.collection.get(
        include=["documents", "metadatas"]
    )

    return [
        {"text": text, "metadata": metadata}
        for text, metadata in zip(
            data["documents"], data["metadatas"]
        )
    ]


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Measure retrieval accuracy and generation "
            "signals on the indexed documents."
        )
    )

    parser.add_argument(
        "--questions",
        help=(
            "JSON file of verified "
            '{"question": ..., "page": N} entries'
        )
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also run the generation model per question"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions"
    )

    args = parser.parse_args()

    vector_store = VectorStore()

    if vector_store.count() == 0:
        print(
            "The index is empty. Run "
            "`python src/main.py ingest` first."
        )
        sys.exit(1)

    if args.questions:
        pairs = json.loads(
            Path(args.questions).read_text(
                encoding="utf-8"
            )
        )
    else:
        pairs = discover_questions(
            load_store_documents(vector_store)
        )

    if args.limit:
        pairs = pairs[: args.limit]

    if not pairs:
        print("No questions found to evaluate.")
        sys.exit(1)

    print(f"Evaluating {len(pairs)} questions.\n")

    embedding_model = EmbeddingModel()

    retriever = Retriever(
        vector_store,
        embedding_model,
        Reranker()
    )

    generator = Generator() if args.generate else None

    top1 = 0

    top4 = 0

    signals = {
        "verified quote": 0,
        "validated citation": 0,
        "unverified reference": 0,
        "abstained": 0,
    }

    misses = []

    for i, pair in enumerate(pairs, start=1):
        question = pair["question"]

        expected = pair["page"]

        documents = retriever.retrieve(
            question,
            n_results=8,
            top_k=4
        )

        pages = [
            d["metadata"].get("page")
            for d in documents
        ]

        hit1 = pages[:1] == [expected]

        hit4 = expected in pages

        top1 += hit1

        top4 += hit4

        if not hit4:
            misses.append((question, expected, pages))

        line = (
            f"[{i}/{len(pairs)}] page {expected}: "
            f"top-1 {'OK ' if hit1 else 'miss'} "
            f"top-4 {'OK' if hit4 else 'MISS'}"
        )

        if generator is not None:
            answer = generator.generate(
                question, documents
            )

            if "Evidence (verified quote" in answer:
                signals["verified quote"] += 1
            elif "\nSource:" in answer:
                signals["validated citation"] += 1
            elif "unverified" in answer:
                signals["unverified reference"] += 1
            elif "could not find" in answer:
                signals["abstained"] += 1

            line += "\n" + answer

        print(line)

    total = len(pairs)

    source_note = (
        "user-verified questions file"
        if args.questions
        else "same-document sanity check, not held-out accuracy"
    )

    print(
        f"\nRetrieval: top-1 {top1}/{total} "
        f"({top1 / total:.0%}), "
        f"top-4 {top4}/{total} ({top4 / total:.0%}) "
        f"[{source_note}]"
    )

    if generator is not None:
        for name, count in signals.items():
            print(f"Generation {name}: {count}/{total}")

    if misses:
        print("\nQuestions whose page missed the top 4:")

        for question, expected, pages in misses:
            print(
                f"- {question} (expected page "
                f"{expected}, got {pages})"
            )


if __name__ == "__main__":
    main()
