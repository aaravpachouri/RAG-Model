"""End-to-end retrieval tests over the generated sample PDF.

These download the small embedding/reranking models from
Hugging Face on first run and cache them afterwards.
"""

import pytest

from chunker import chunk_pages
from embeddings import EmbeddingModel
from ingest import ingest_documents
from loader import load_pdf
from reranker import Reranker
from retriever import Retriever
from vectorstore import VectorStore


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel()


@pytest.fixture(scope="module")
def indexed_store(sample_pdf, embedding_model):
    """Sample PDF chunked, embedded and stored in memory."""

    pages = load_pdf(
        str(sample_pdf),
        use_vision=False,
        verbose=False,
    )

    chunks = chunk_pages(pages)

    store = VectorStore(persist=False)

    store.add_documents(
        chunks,
        embedding_model.encode(
            [c["text"] for c in chunks]
        ),
    )

    return store, chunks


def test_ingest_documents_end_to_end(
    tmp_path, embedding_model
):

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    # Reuse the sample-PDF generator against a new dir.
    import pymupdf
    from conftest import SAMPLE_PAGES

    pdf_path = docs_dir / "notes.pdf"
    document = pymupdf.open()
    for text in SAMPLE_PAGES:
        page = document.new_page()
        page.insert_text(
            pymupdf.Point(72, 72), text, fontsize=11
        )
    document.save(str(pdf_path))
    document.close()

    store = VectorStore(persist=False)

    stored = ingest_documents(
        documents_path=docs_dir,
        embedding_model=embedding_model,
        vector_store=store,
        use_vision=False,
        verbose=False,
    )

    assert stored > 0
    assert store.count() == stored


def test_retrieval_finds_relevant_page(
    indexed_store, embedding_model
):

    store, _ = indexed_store

    retriever = Retriever(
        store, embedding_model
    )

    results = retriever.retrieve(
        "What is the moment of inertia of a "
        "point mass?",
        n_results=5,
        top_k=3,
    )

    assert len(results) > 0

    top = results[0]

    assert "moment of inertia" in top["text"].lower()
    assert top["metadata"]["page"] == 1
    assert top["metadata"]["source"] == "sample_notes.pdf"


def test_retrieval_ranks_by_topic(
    indexed_store, embedding_model
):

    store, _ = indexed_store

    retriever = Retriever(
        store, embedding_model
    )

    results = retriever.retrieve(
        "conservation of angular momentum",
        n_results=5,
        top_k=2,
    )

    assert results
    assert results[0]["metadata"]["page"] == 3
    assert "angular momentum" in results[0]["text"].lower()


def test_reranker_improves_order(
    indexed_store, embedding_model
):

    store, _ = indexed_store

    reranker = Reranker()

    retriever = Retriever(
        store, embedding_model, reranker
    )

    results = retriever.retrieve(
        "define torque and give its SI unit",
        n_results=6,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["metadata"]["page"] == 2
    assert "rerank_score" in results[0]


def test_empty_store_returns_no_results(
    embedding_model
):

    store = VectorStore(persist=False)

    retriever = Retriever(
        store, embedding_model
    )

    assert retriever.retrieve("anything") == []
