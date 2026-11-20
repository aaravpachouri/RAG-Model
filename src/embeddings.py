"""Sentence-Transformer embedding model for semantic search."""

import os

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self, model_name: str = None):

        self.model = SentenceTransformer(
            model_name
            or os.environ.get("RAG_EMBEDDING_MODEL")
            or DEFAULT_MODEL
        )

    def encode(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
