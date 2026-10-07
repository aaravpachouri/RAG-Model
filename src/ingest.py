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


def main():

    print("=" * 50)
    print("RAG DOCUMENT INGESTION")
    print("=" * 50)

    files = list(
        DOCUMENTS_PATH.glob("*.pdf")
    )

    files += list(
        DOCUMENTS_PATH.glob("*.txt")
    )

    if not files:

        print(
            "No PDF or TXT files found."
        )

        return

    print(
        f"\nFound {len(files)} documents.\n"
    )

    all_documents = []

    for file_path in files:

        print(
            f"Loading: {file_path.name}"
        )

        pages = load_document(
            str(file_path)
        )

        chunks = chunk_pages(
            pages
        )

        all_documents.extend(
            chunks
        )

        print(
            f"  Pages: {len(pages)}"
        )

        print(
            f"  Chunks: {len(chunks)}"
        )

    print(
        f"\nTotal chunks: "
        f"{len(all_documents)}"
    )

    print(
        "\nCreating embeddings..."
    )

    embedding_model = EmbeddingModel()

    texts = [
        document["text"]
        for document in all_documents
    ]

    embeddings = embedding_model.encode(
        texts
    )

    print(
        "Embeddings created."
    )

    print(
        "\nSaving to ChromaDB..."
    )

    vector_store = VectorStore()

    vector_store.add_documents(
        all_documents,
        embeddings
    )

    print(
        "\n" + "=" * 50
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 50
    )


if __name__ == "__main__":
    main()