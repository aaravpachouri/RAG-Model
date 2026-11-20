"""Edge cases: awkward PDFs and mixed folders."""

import pymupdf

from ingest import find_documents, ingest_documents
from loader import load_pdf
from embeddings import EmbeddingModel
from vectorstore import VectorStore


def _blank_pdf(path):
    """One page with a drawing but no text layer."""

    document = pymupdf.open()
    page = document.new_page()
    page.draw_circle(pymupdf.Point(200, 200), 50)
    document.save(str(path))
    document.close()


def test_image_only_page_is_dropped_without_vision(
    tmp_path,
):

    path = tmp_path / "scan.pdf"
    _blank_pdf(path)

    pages = load_pdf(
        str(path), use_vision=False, verbose=False
    )

    # No crash, and no empty-text page enters the
    # pipeline.
    assert pages == []


def test_ingest_mixed_directory(tmp_path):

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    (docs_dir / "notes.txt").write_text(
        "Angular momentum L equals I times omega. "
        "It is conserved when no external torque "
        "acts on the system.",
        encoding="utf-8",
    )

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        pymupdf.Point(72, 72),
        "Torque tau equals r cross F. "
        "The SI unit of torque is the newton metre.",
        fontsize=11,
    )
    document.save(str(docs_dir / "physics.pdf"))
    document.close()

    # An unsupported file must be ignored.
    (docs_dir / "readme.md").write_text(
        "ignore me", encoding="utf-8"
    )

    found = find_documents(docs_dir)
    assert sorted(p.name for p in found) == [
        "notes.txt",
        "physics.pdf",
    ]

    store = VectorStore(persist=False)

    stored = ingest_documents(
        documents_path=docs_dir,
        embedding_model=EmbeddingModel(),
        vector_store=store,
        use_vision=False,
        verbose=False,
    )

    assert stored > 0
    assert store.count() == stored

    sources = {
        item["source"]
        for item in store.collection.get(
            include=["metadatas"]
        )["metadatas"]
    }

    assert sources == {"notes.txt", "physics.pdf"}


def test_ingest_prunes_deleted_files(tmp_path):
    """Re-ingesting after deleting a file removes its
    chunks automatically, no reset needed."""

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    first = docs_dir / "first.txt"
    first.write_text(
        "Alpha document about torque.",
        encoding="utf-8",
    )

    second = docs_dir / "second.txt"
    second.write_text(
        "Beta document about momentum.",
        encoding="utf-8",
    )

    store = VectorStore(persist=False)
    model = EmbeddingModel()

    ingest_documents(
        documents_path=docs_dir,
        embedding_model=model,
        vector_store=store,
        use_vision=False,
        verbose=False,
    )

    assert store.count() == 2

    # Delete one file and re-ingest normally.
    first.unlink()

    ingest_documents(
        documents_path=docs_dir,
        embedding_model=model,
        vector_store=store,
        use_vision=False,
        verbose=False,
    )

    assert store.count() == 1

    sources = {
        item["source"]
        for item in store.collection.get(
            include=["metadatas"]
        )["metadatas"]
    }

    assert sources == {"second.txt"}


def test_ingest_replaces_edited_files(tmp_path):
    """Editing a document and re-ingesting replaces
    its old chunks instead of keeping both."""

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    notes = docs_dir / "notes.txt"
    notes.write_text(
        "Original text about angular velocity.",
        encoding="utf-8",
    )

    store = VectorStore(persist=False)
    model = EmbeddingModel()

    ingest_documents(
        documents_path=docs_dir,
        embedding_model=model,
        vector_store=store,
        use_vision=False,
        verbose=False,
    )

    assert store.count() == 1

    notes.write_text(
        "Rewritten text about harmonic motion.",
        encoding="utf-8",
    )

    ingest_documents(
        documents_path=docs_dir,
        embedding_model=model,
        vector_store=store,
        use_vision=False,
        verbose=False,
    )

    assert store.count() == 1

    texts = store.collection.get()["documents"]

    assert any(
        "harmonic motion" in text
        for text in texts
    )

    assert not any(
        "angular velocity" in text
        for text in texts
    )


def test_reset_removes_stale_chunks(tmp_path):
    """--reset clears the whole index before
    re-ingesting, so nothing stale survives."""

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    first = docs_dir / "first.txt"
    first.write_text(
        "Alpha document about torque.",
        encoding="utf-8",
    )

    second = docs_dir / "second.txt"
    second.write_text(
        "Beta document about momentum.",
        encoding="utf-8",
    )

    store = VectorStore(persist=False)
    model = EmbeddingModel()

    ingest_documents(
        documents_path=docs_dir,
        embedding_model=model,
        vector_store=store,
        use_vision=False,
        verbose=False,
    )

    assert store.count() == 2

    # Delete one file and re-ingest with reset.
    first.unlink()

    ingest_documents(
        documents_path=docs_dir,
        embedding_model=model,
        vector_store=store,
        use_vision=False,
        verbose=False,
        reset=True,
    )

    assert store.count() == 1

    sources = {
        item["source"]
        for item in store.collection.get(
            include=["metadatas"]
        )["metadatas"]
    }

    assert sources == {"second.txt"}


def test_ingest_empty_directory(tmp_path, capsys):

    stored = ingest_documents(
        documents_path=tmp_path,
        embedding_model=None,
        vector_store=None,
        use_vision=False,
        verbose=False,
    )

    assert stored == 0


def test_sync_removes_trailing_chunks_and_failed_upsert_keeps_index(monkeypatch):
    import numpy as np
    import pytest
    store = VectorStore(persist=False)
    chunks = [
        {"text": f"chunk {i}", "source": "n.txt", "page": 1, "chunk": i}
        for i in range(3)
    ]
    store.add_documents(chunks, np.ones((3, 4)), synchronize=True)
    assert store.count() == 3
    store.add_documents(chunks[:1], np.ones((1, 4)), synchronize=True)
    assert store.count() == 1

    def fail(**kwargs):
        raise RuntimeError("write failed")
    monkeypatch.setattr(type(store.collection), "upsert", lambda *a, **k: fail(**k))
    with pytest.raises(RuntimeError):
        store.add_documents(chunks[1:], np.ones((2, 4)), synchronize=True)
    assert store.count() == 1


def test_empty_folder_preserves_index_unless_reset(tmp_path):
    import numpy as np
    store = VectorStore(persist=False)
    store.add_documents(
        [{"text": "original", "source": "n.txt", "page": 1, "chunk": 0}],
        np.ones((1, 4))
    )
    ingest_documents(tmp_path, vector_store=store, verbose=False)
    assert store.count() == 1
    ingest_documents(tmp_path, vector_store=store, verbose=False, reset=True)
    assert store.count() == 0


def test_extraction_failure_preserves_index(tmp_path, monkeypatch):
    import numpy as np
    import pytest
    import ingest
    (tmp_path / "n.txt").write_text("empty extraction test")
    store = VectorStore(persist=False)
    store.add_documents(
        [{"text": "original", "source": "n.txt", "page": 1, "chunk": 0}],
        np.ones((1, 4))
    )
    monkeypatch.setattr(ingest, "load_document", lambda *a, **kw: [])
    with pytest.raises(RuntimeError, match="Index unchanged"):
        ingest_documents(tmp_path, vector_store=store, verbose=False)
    assert store.collection.get()["documents"] == ["original"]
