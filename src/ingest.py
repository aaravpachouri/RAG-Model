"""Ingest documents from data/documents into the vector store.

Ingestion is incremental: a small manifest (data/ingest_manifest.json,
gitignored along with the rest of data/) records the SHA-256 of every
indexed file. Re-running ``ingest`` skips files whose content has not
changed, re-indexes edited files, and prunes chunks of files that were
deleted from the folder. Each file is loaded, chunked, embedded and
stored one at a time, so memory stays bounded by the largest single
document, and an interrupted run keeps every file that completed -
the next run resumes by skipping them.
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
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

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT /
    "data" /
    "ingest_manifest.json"
)

MANIFEST_VERSION = 1


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


def chunk_id(chunk) -> str:
    """Deterministic ID for one chunk (matches VectorStore)."""

    return (
        f"{chunk['source']}_"
        f"page_{chunk['page']}_"
        f"chunk_{chunk['chunk']}"
    )


def _sha256(path) -> str:
    """Content hash, streamed so huge PDFs stay memory-cheap."""

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def _empty_manifest():
    return {"version": MANIFEST_VERSION, "files": {}}


def _load_manifest(manifest_path):
    """Read the ingest manifest; tolerate a missing/corrupt file."""

    if manifest_path is None:
        return _empty_manifest()

    try:
        data = json.loads(
            Path(manifest_path).read_text(
                encoding="utf-8"
            )
        )
        if (
            isinstance(data, dict)
            and isinstance(data.get("files"), dict)
        ):
            return data
    except (OSError, ValueError):
        pass

    return _empty_manifest()


def _save_manifest(manifest_path, manifest):
    """Persist the manifest (resume checkpoint)."""

    if manifest_path is None:
        return

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )


def ingest_documents(
    documents_path=None,
    embedding_model=None,
    vector_store=None,
    use_vision=True,
    verbose=True,
    reset=False,
    force=False,
    manifest_path=None
):
    """
    Load every new or changed PDF/TXT in the documents
    directory, chunk, embed and store it. Files whose
    content hash matches the manifest are skipped, and
    chunks of files removed from the folder are pruned.
    Returns the number of chunks stored this run (0 when
    everything was already up to date).
    """

    files = find_documents(documents_path)

    if not files:
        if reset:
            if vector_store is None:
                vector_store = VectorStore()
            vector_store.reset()
            if (
                manifest_path is not None
                or getattr(vector_store, "persist", False)
            ):
                _save_manifest(
                    manifest_path or DEFAULT_MANIFEST_PATH,
                    _empty_manifest()
                )
        if verbose:
            print("No PDF or TXT files found.")
            if not reset:
                print(
                    "Empty folder: index unchanged. "
                    "Use --reset to explicitly clear it."
                )
        return 0

    if vector_store is None:
        vector_store = VectorStore()

    # The manifest only makes sense for a persistent
    # store; tests with ephemeral stores can pass an
    # explicit manifest_path to exercise resume logic.
    if (
        manifest_path is None
        and getattr(vector_store, "persist", False)
    ):
        manifest_path = DEFAULT_MANIFEST_PATH

    manifest = _load_manifest(manifest_path)

    if reset:
        if verbose:
            print(
                "Clearing existing index "
                "(--reset)..."
            )
        vector_store.reset()
        manifest = _empty_manifest()
        _save_manifest(manifest_path, manifest)

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    if verbose:
        print(f"Found {len(files)} documents.")
        print(
            f"Embedding device: "
            f"{getattr(embedding_model, 'device', '?')} "
            f"(batch size "
            f"{getattr(embedding_model, 'batch_size', '?')})"
        )

    disk_names = {path.name for path in files}

    total_new_chunks = 0
    indexed_files = 0
    skipped_files = 0

    for file_path in files:

        name = file_path.name
        digest = _sha256(file_path)
        entry = manifest["files"].get(name)

        if (
            not force
            and entry is not None
            and entry.get("sha256") == digest
        ):
            skipped_files += 1
            if verbose:
                print(
                    f"Unchanged, skipping: {name}"
                )
            continue

        if verbose:
            print(f"Loading: {name}")

        started = time.perf_counter()

        pages = load_document(
            str(file_path),
            use_vision=use_vision,
            verbose=verbose
        )

        chunks = chunk_pages(pages)

        if not chunks:
            raise RuntimeError(
                f"No text extracted from {name}. "
                "Index unchanged for this file: its "
                "previously stored chunks (if any) were "
                "kept and no stale chunks were pruned. "
                "Files processed earlier in this run "
                "were already updated and will be "
                "skipped on the next run. Fix or remove "
                "the file, then re-run ingest."
            )

        loaded_in = time.perf_counter() - started

        if verbose:
            print(f"  Pages: {len(pages)}")
            print(f"  Chunks: {len(chunks)}")

        started = time.perf_counter()

        embeddings = embedding_model.encode(
            [chunk["text"] for chunk in chunks]
        )

        embedded_in = time.perf_counter() - started

        new_ids = {
            chunk_id(chunk) for chunk in chunks
        }

        old_ids = set()
        if entry is not None:
            old_ids.update(
                entry.get("chunk_ids", [])
            )
        # Also cover chunks stored before the manifest
        # existed (pre-incremental ingests).
        old_ids.update(
            vector_store.ids_for_source(name)
        )

        vector_store.add_documents(
            chunks,
            embeddings
        )

        stale_for_file = sorted(old_ids - new_ids)
        if stale_for_file:
            vector_store.delete_ids(stale_for_file)

        manifest["files"][name] = {
            "sha256": digest,
            "pages": len(pages),
            "chunks": len(chunks),
            "chunk_ids": sorted(new_ids),
            "ingested_at": datetime.now(
                timezone.utc
            ).isoformat(timespec="seconds"),
        }

        # Checkpoint after every file: an interrupted
        # run keeps completed files indexed, and the
        # next run skips them.
        _save_manifest(manifest_path, manifest)

        total_new_chunks += len(chunks)
        indexed_files += 1

        if verbose:
            print(
                f"  Indexed in "
                f"{loaded_in + embedded_in:.1f}s "
                f"(extract {loaded_in:.1f}s, "
                f"embed {embedded_in:.1f}s)"
            )

    # Prune chunks whose source file left the folder,
    # then reconcile the store with the manifest so no
    # orphaned chunks survive (including trailing
    # chunks of shortened files from older runs).
    removed_files = [
        name
        for name in manifest["files"]
        if name not in disk_names
    ]

    for name in removed_files:
        del manifest["files"][name]
        if verbose:
            print(
                f"Removed from folder, pruning: {name}"
            )

    expected_ids = set()
    for entry in manifest["files"].values():
        expected_ids.update(
            entry.get("chunk_ids", [])
        )

    stale = sorted(
        set(vector_store.all_ids()) - expected_ids
    )

    if stale:
        vector_store.delete_ids(stale)

    _save_manifest(manifest_path, manifest)

    if verbose:
        print(
            f"\nIndexed {indexed_files} file(s) "
            f"({total_new_chunks} chunks), "
            f"skipped {skipped_files} unchanged, "
            f"pruned {len(stale)} stale chunk(s)."
        )
        print("\nINGESTION COMPLETE")

    return total_new_chunks


def main():

    print("=" * 50)
    print("RAG DOCUMENT INGESTION")
    print("=" * 50)

    ingest_documents(
        reset="--reset" in sys.argv,
        force="--force" in sys.argv
    )


if __name__ == "__main__":
    main()
