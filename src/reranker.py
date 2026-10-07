from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
            pairs
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