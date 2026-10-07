class Retriever:

    def __init__(
        self,
        vector_store,
        embedding_model,
        reranker
    ):

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

        documents = self.reranker.rerank(
            query,
            documents,
            top_k=top_k
        )

        return documents