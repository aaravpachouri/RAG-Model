"""Interactive chat over the already-ingested documents.

Kept for convenience; equivalent to `python src/main.py chat`.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import build_qa_stack, run_chat


def main():

    print("=" * 50)
    print("RAG AI")
    print("=" * 50)

    print("\nLoading AI components...")

    retriever, vector_store, generator = (
        build_qa_stack()
    )

    if vector_store.count() == 0:
        print(
            "\nThe vector store is empty. Run "
            "`python src/main.py ingest` first."
        )
        return

    run_chat(retriever, generator)


if __name__ == "__main__":
    main()
