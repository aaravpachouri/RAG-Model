"""ChromaDB-backed vector store for document chunks."""

from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "knowledge"


class VectorStore:

    def __init__(self, persist: bool = True):
        """
        persist=True stores the index on disk under
        data/chroma (normal use). persist=False uses an
        ephemeral in-memory store (tests).
        """

        self.persist = persist

        if persist:
            CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_PATH)
            )
            collection_name = COLLECTION_NAME
        else:
            self.client = chromadb.EphemeralClient()
            # Ephemeral clients share one in-memory
            # system per process, so use a unique
            # collection to keep stores independent.
            import uuid
            collection_name = (
                f"{COLLECTION_NAME}_"
                f"{uuid.uuid4().hex[:12]}"
            )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Drop every stored chunk.

        Plain ingestion already prunes chunks of
        deleted files and re-indexes edited ones;
        reset remains for forcing a full rebuild,
        for example after changing chunking
        settings.
        """

        self.client.delete_collection(
            name=self.collection.name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection.name
            )
        )

    def add_documents(
        self,
        documents,
        embeddings,
        synchronize=False
    ):
        """
        documents: list of chunk dicts with keys
        text, source, page, chunk.
        embeddings: numpy array aligned with documents.
        """

        ids = []

        texts = []

        metadatas = []

        for document in documents:

            source = document["source"]
            page = document["page"]
            chunk = document["chunk"]

            doc_id = (
                f"{source}_"
                f"page_{page}_"
                f"chunk_{chunk}"
            )

            ids.append(doc_id)

            texts.append(
                document["text"]
            )

            metadatas.append({
                "source": source,
                "page": page,
                "chunk": chunk
            })

        if not ids:
            print("No chunks to store.")
            return

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        if synchronize:
            # Delete stale IDs only after all new chunks
            # were stored successfully. This also removes
            # trailing chunks when a document gets shorter.
            existing = self.collection.get(include=[])["ids"]
            stale = sorted(set(existing) - set(ids))
            if stale:
                self.collection.delete(ids=stale)

        print(
            f"Stored {len(documents)} chunks."
        )

    def all_ids(self):
        """Every stored chunk ID."""
        return self.collection.get(include=[])["ids"]

    def ids_for_source(self, source: str):
        """Stored chunk IDs belonging to one document."""
        return self.collection.get(
            where={"source": source},
            include=[]
        )["ids"]

    def delete_ids(self, ids):
        """Delete specific chunk IDs (no-op on empty input)."""
        ids = list(ids)
        if ids:
            self.collection.delete(ids=ids)

    def search(
        self,
        query_embedding,
        n_results=8
    ):

        if self.collection.count() == 0:
            return []

        n_results = min(
            n_results,
            self.collection.count()
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = []

        for i in range(
            len(results["documents"][0])
        ):

            documents.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return documents
