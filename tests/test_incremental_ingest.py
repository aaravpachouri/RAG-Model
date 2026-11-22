"""Incremental ingestion: skip unchanged, resume, prune, force."""

import json

import pytest

from embeddings import EmbeddingModel
from ingest import (
    _load_manifest,
    _sha256,
    ingest_documents,
)
from vectorstore import VectorStore


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


def _make_docs(tmp_path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Alpha notes about torque and rotation.",
        encoding="utf-8",
    )
    (docs_dir / "beta.txt").write_text(
        "Beta notes about momentum and impulse.",
        encoding="utf-8",
    )
    return docs_dir


def _run(docs_dir, model, store, manifest, **kwargs):
    return ingest_documents(
        documents_path=docs_dir,
        embedding_model=model,
        vector_store=store,
        use_vision=False,
        verbose=False,
        manifest_path=manifest,
        **kwargs,
    )


def test_unchanged_files_are_skipped(tmp_path, model):
    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"

    first = _run(docs_dir, model, store, manifest)
    assert first == 2
    assert store.count() == 2
    assert manifest.exists()

    class Bomb:
        def encode(self, texts, **kwargs):
            raise AssertionError(
                "encode called for unchanged files"
            )

    second = _run(docs_dir, Bomb(), store, manifest)
    assert second == 0
    assert store.count() == 2


def test_only_edited_file_is_re_embedded(tmp_path, model):
    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"

    _run(docs_dir, model, store, manifest)

    (docs_dir / "alpha.txt").write_text(
        "Alpha rewritten about harmonic motion.",
        encoding="utf-8",
    )

    embedded_texts = []
    real_encode = model.encode

    def spy(texts, **kwargs):
        embedded_texts.extend(texts)
        return real_encode(texts, **kwargs)

    model.encode = spy
    try:
        stored = _run(docs_dir, model, store, manifest)
    finally:
        model.encode = real_encode

    assert stored == 1
    assert embedded_texts == [
        "Alpha rewritten about harmonic motion."
    ]
    assert store.count() == 2

    texts = store.collection.get()["documents"]
    assert any("harmonic motion" in t for t in texts)
    assert not any("torque" in t for t in texts)


def test_deleted_file_is_pruned_with_manifest(
    tmp_path, model
):
    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"

    _run(docs_dir, model, store, manifest)
    assert store.count() == 2

    (docs_dir / "beta.txt").unlink()

    stored = _run(docs_dir, model, store, manifest)
    assert stored == 0
    assert store.count() == 1

    sources = {
        m["source"]
        for m in store.collection.get(
            include=["metadatas"]
        )["metadatas"]
    }
    assert sources == {"alpha.txt"}

    data = json.loads(manifest.read_text())
    assert list(data["files"]) == ["alpha.txt"]


def test_interrupted_run_resumes_from_manifest(
    tmp_path, model
):
    """A crash mid-run keeps completed files indexed;
    the next run embeds only the rest."""

    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"

    real_encode = model.encode
    calls = {"n": 0}

    def fail_on_second(texts, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real_encode(texts, **kwargs)

    model.encode = fail_on_second
    try:
        with pytest.raises(RuntimeError):
            _run(docs_dir, model, store, manifest)
    finally:
        model.encode = real_encode

    # First file completed and was checkpointed.
    assert store.count() == 1
    data = json.loads(manifest.read_text())
    assert list(data["files"]) == ["alpha.txt"]

    # Retry: the completed file is skipped, the
    # remaining one is embedded.
    stored = _run(docs_dir, model, store, manifest)
    assert stored == 1
    assert store.count() == 2


def test_force_re_embeds_unchanged_files(tmp_path, model):
    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"

    _run(docs_dir, model, store, manifest)

    stored = _run(
        docs_dir, model, store, manifest, force=True
    )
    assert stored == 2
    assert store.count() == 2


def test_reset_clears_manifest_and_index(tmp_path, model):
    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"

    _run(docs_dir, model, store, manifest)
    assert store.count() == 2

    stored = _run(
        docs_dir, model, store, manifest, reset=True
    )
    assert stored == 2
    assert store.count() == 2


def test_corrupt_manifest_reindexes(tmp_path, model):
    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{ not json", encoding="utf-8")

    assert _load_manifest(manifest)["files"] == {}

    stored = _run(docs_dir, model, store, manifest)
    assert stored == 2


def test_extraction_failure_keeps_file_and_checkpoints(
    tmp_path, model
):
    """A file with no extractable text raises; files
    completed earlier in the run stay indexed."""

    docs_dir = _make_docs(tmp_path)
    store = VectorStore(persist=False)
    manifest = tmp_path / "manifest.json"

    import ingest

    real_load = ingest.load_document

    def fail_beta(path, **kwargs):
        if path.endswith("beta.txt"):
            return []
        return real_load(path, **kwargs)

    ingest.load_document = fail_beta
    try:
        with pytest.raises(
            RuntimeError, match="Index unchanged"
        ):
            _run(docs_dir, model, store, manifest)
    finally:
        ingest.load_document = real_load

    # alpha.txt (processed first) was committed.
    assert store.count() == 1
    data = json.loads(manifest.read_text())
    assert list(data["files"]) == ["alpha.txt"]

    # Fixing beta and re-running completes the job.
    stored = _run(docs_dir, model, store, manifest)
    assert stored == 1
    assert store.count() == 2


def test_sha256_detects_content_change(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("one", encoding="utf-8")
    first = _sha256(path)
    path.write_text("two", encoding="utf-8")
    assert _sha256(path) != first
