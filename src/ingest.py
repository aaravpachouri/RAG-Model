"""Ingest documents from data/documents into the vector store."""

import sys
from pathlib import Path

from loader import load_document
from chunker import chunk_pages
from embeddings import EmbeddingModel
from vectorstore import VectorStore


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

DOCUMENTS_PATH = (
    PROJECT_ROOT /
    "data" /
    "documents"
)


def find_documents(documents_path=None):
    """List ingestible files in the documents directory."""

    directory = Path(
        documents_path or DOCUMENTS_PATH
    )

    if not directory.exists():
        return []

    files = sorted(
        directory.glob("*.pdf")
    )

    files += sorted(
        directory.glob("*.txt")
    )

    return files


def ingest_documents(
    documents_path=None,
    embedding_model=None,
    vector_store=None,
    use_vision=True,
    verbose=True,
    reset=False
):
    """
    Load every PDF/TXT in the documents directory,
    chunk, embed and store them. Returns the number
    of chunks stored.
    """

    files = find_documents(documents_path)

    if not files:
        if reset:
            if vector_store is None:
                vector_store = VectorStore()
            vector_store.reset()
        if verbose:
            print("No PDF or TXT files found.")
            if not reset:
                print(
                    "Empty folder: index unchanged. "
                    "Use --reset to explicitly clear it."
                )
        return 0

    if verbose:
        print(f"Found {len(files)} documents.")

    all_chunks = []

    for file_path in files:

        if verbose:
            print(
                f"Loading: {file_path.name}"
            )

        pages = load_document(
            str(file_path),
            use_vision=use_vision,
            verbose=verbose
        )

        chunks = chunk_pages(
            pages
        )

        if not chunks:
            raise RuntimeError(
                f"No text extracted from {file_path.name}. "
                "Index unchanged. Check extraction before "
                "re-ingesting; no stale chunks were pruned."
            )

        all_chunks.extend(chunks)

        if verbose:
            print(
                f"  Pages: {len(pages)}"
            )

            print(
                f"  Chunks: {len(chunks)}"
            )

    if not all_chunks:
        if verbose:
            print(
                "\nNo text could be extracted "
                "from the documents."
            )
        return 0

    if verbose:
        print(
            f"\nTotal chunks: "
            f"{len(all_chunks)}"
        )

        print(
            "\nCreating embeddings..."
        )

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    if verbose:
        print(
            "Embeddings created."
        )

        print(
            "\nSaving to ChromaDB..."
        )

    if vector_store is None:
        vector_store = VectorStore()

    if reset:
        if verbose:
            print(
                "\nClearing existing index "
                "(--reset)..."
            )
        vector_store.reset()
    vector_store.add_documents(
        all_chunks,
        embeddings,
        synchronize=True
    )

    if verbose:
        print(
            "\nINGESTION COMPLETE"
        )

    return len(all_chunks)


def main():

    print("=" * 50)
    print("RAG DOCUMENT INGESTION")
    print("=" * 50)

    ingest_documents(
        reset="--reset" in sys.argv
    )


if __name__ == "__main__":
    main()
