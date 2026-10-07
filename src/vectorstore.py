from pathlib import Path
import chromadb


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="knowledge"
            )
        )

    def add_documents(
        self,
        documents,
        embeddings
    ):

        ids = []

        texts = []

        metadatas = []

        for i, document in enumerate(documents):

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

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(
            f"Stored {len(documents)} chunks."
        )

    def search(
        self,
        query_embedding,
        n_results=8
    ):

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