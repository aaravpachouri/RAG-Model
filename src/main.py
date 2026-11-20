"""Command line entry point for the RAG system.

Usage:
    python src/main.py ingest            Index every PDF/TXT in data/documents
    python src/main.py ask "question"  Answer one question
    python src/main.py chat            Interactive question loop
    python src/main.py                 Ingest if needed, then chat
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embeddings import EmbeddingModel
from vectorstore import VectorStore
from retriever import Retriever
from reranker import Reranker
from generator import Generator
from ingest import ingest_documents


def build_qa_stack(use_reranker=True):
    """Load the models needed to answer questions."""

    embedding_model = EmbeddingModel()

    vector_store = VectorStore()

    reranker = Reranker() if use_reranker else None

    retriever = Retriever(
        vector_store,
        embedding_model,
        reranker
    )

    generator = Generator()

    return retriever, vector_store, generator


NOT_FOUND_MESSAGE = (
    "I could not find this information "
    "in the indexed documents."
)


def retrieval_is_confident(documents) -> bool:
    """Abstain when the reranker says nothing matched.

    Reranker scores separate cleanly: on-topic chunks
    score clearly positive, off-topic questions score
    strongly negative. When no candidate clears the
    threshold, the system should say so instead of
    letting the LLM answer from its own knowledge.
    """

    scored = [
        d for d in documents
        if "rerank_score" in d
    ]

    if not scored:
        # No reranker in use: no confidence signal,
        # so let the generator decide.
        return True

    threshold = float(
        os.environ.get("RAG_ABSTAIN_THRESHOLD", "0")
    )

    best = max(
        d["rerank_score"] for d in scored
    )

    return best >= threshold


def format_passages(documents, max_chars=None):
    """Render the retrieved chunks for display.

    These are indexed passages with whitespace
    normalized, shown so the reader can check the
    generated answer against the source text.
    """

    lines = [
        "RETRIEVED PASSAGES "
        "(extracted text, whitespace normalized):"
    ]

    for i, document in enumerate(
        documents,
        start=1
    ):

        metadata = document.get("metadata", {})

        text = " ".join(
            document["text"].split()
        )

        if max_chars is not None and len(text) > max_chars:
            text = (
                text[:max_chars].rstrip()
                + " ..."
            )

        lines.append(
            f"\n[{i}] "
            f"{metadata.get('source', 'unknown')}"
            f", page "
            f"{metadata.get('page', '?')}\n"
            f"{text}"
        )

    return "\n".join(lines)


def print_passages(documents):
    """Show retrieved passages before the answer."""

    print("\n" + format_passages(documents))

    print(
        "\nGENERATED ANSWER "
        "(unverified, check it against the "
        "passages above):"
    )


def answer_question(
    question,
    retriever,
    generator,
    n_results=8,
    top_k=4,
    on_passages=None,
    retrieve_only=False
):
    """Retrieve context and generate an answer.

    on_passages, when given, is called with the
    retrieved chunks after the confidence gate
    passes, so callers can show the exact source
    text next to the generated answer.
    """

    documents = retriever.retrieve(
        question,
        n_results=n_results,
        top_k=top_k
    )

    if not documents:
        return NOT_FOUND_MESSAGE

    if not retrieval_is_confident(documents):
        return NOT_FOUND_MESSAGE

    if on_passages is not None:
        on_passages(documents)

    if retrieve_only:
        return format_passages(documents)

    return generator.generate(
        question,
        documents
    )


def run_chat(retriever, generator):

    print("\nRAG system ready.")
    print("Type 'exit' to quit.\n")

    while True:

        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in [
            "exit",
            "quit"
        ]:
            print("\nGoodbye!")
            break

        print("\nSearching documents...")

        documents = retriever.retrieve(
            question,
            n_results=8,
            top_k=4
        )

        if not documents or (
            not retrieval_is_confident(documents)
        ):

            print(
                "\n" + NOT_FOUND_MESSAGE + "\n"
            )

            continue

        print_passages(documents)

        try:
            answer = generator.generate(
                question,
                documents
            )
        except RuntimeError as exc:
            print(f"\n{exc}\n")
            continue

        print(answer)

        print("\n" + "-" * 50)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Ask questions over your PDF/TXT "
            "notes using local RAG."
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="auto",
        choices=[
            "auto",
            "ingest",
            "ask",
            "chat"
        ]
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question for the 'ask' command"
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the reranking stage"
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
        help=(
            "Disable vision-based extraction "
            "during ingestion"
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Clear the whole index before "
            "ingesting (plain ingestion already "
            "prunes deleted files and re-indexes "
            "edited ones)"
        )
    )

    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Show extracted passages without calling Ollama (ask only)"
    )

    args = parser.parse_args()
    if args.retrieve_only and args.command != "ask":
        parser.error("--retrieve-only requires the ask command")

    if args.command == "ingest":
        ingest_documents(
            use_vision=not args.no_vision,
            reset=args.reset
        )
        return

    if args.command == "ask" and not args.question:
        parser.error(
            "the 'ask' command needs a question"
        )

    retriever, vector_store, generator = (
        build_qa_stack(
            use_reranker=not args.no_rerank
        )
    )

    if args.command == "ask":

        try:
            print(
                answer_question(
                    args.question,
                    retriever,
                    generator,
                    on_passages=(None if args.retrieve_only else print_passages),
                    retrieve_only=args.retrieve_only
                )
            )
        except RuntimeError as exc:
            print(exc)
            sys.exit(1)

        return

    # "chat" and "auto" both end in the chat
    # loop; "auto" ingests first when the store
    # is empty.
    if (
        args.command == "auto"
        and vector_store.count() == 0
    ):

        print(
            "Vector store is empty, "
            "running ingestion first.\n"
        )

        stored = ingest_documents(
            use_vision=not args.no_vision
        )

        if stored == 0:
            return

    run_chat(retriever, generator)


if __name__ == "__main__":
    main()
