"""Retrieve the most relevant chunks for a query."""


class Retriever:

    def __init__(
        self,
        vector_store,
        embedding_model,
        reranker=None
    ):
        """
        reranker is optional. Without it, results come
        straight from the vector store ordered by
        embedding distance.
        """

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker

    def retrieve(
        self,
        query,
        n_results=8,
        top_k=4
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )[0]
        )

        documents = (
            self.vector_store.search(
                query_embedding,
                n_results=n_results
            )
        )

        if self.reranker is not None:

            documents = self.reranker.rerank(
                query,
                documents,
                top_k=top_k
            )

        else:

            documents = documents[:top_k]

        return documents
