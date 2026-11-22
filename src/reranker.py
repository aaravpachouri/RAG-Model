import os

from sentence_transformers import CrossEncoder

from embeddings import detect_device, default_batch_size


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
    ):

        self.device = (
            device
            or os.environ.get("RAG_RERANK_DEVICE")
            or detect_device()
        )

        self.batch_size = int(
            batch_size
            or os.environ.get("RAG_RERANK_BATCH_SIZE")
            or default_batch_size(self.device)
        )

        self.model = CrossEncoder(
            model_name or DEFAULT_RERANKER_MODEL,
            device=self.device,
        )

    def rerank(
        self,
        query,
        documents,
        top_k=4
    ):

        if not documents:
            return []

        pairs = []

        for document in documents:

            pairs.append([
                query,
                document["text"]
            ])

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
        )

        for document, score in zip(
            documents,
            scores
        ):

            document["rerank_score"] = float(
                score
            )

        documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return documents[:top_k]
