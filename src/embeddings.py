"""Sentence-Transformer embedding model for semantic search."""

import os

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


def detect_device():
    """Pick the best available compute device.

    Order: explicit RAG_DEVICE override, then CUDA
    (an NVIDIA GPU, if torch can use it), else CPU.
    """

    override = os.environ.get("RAG_DEVICE")

    if override:
        return override

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def default_batch_size(device: str) -> int:
    """Larger batches pay off on a GPU; keep CPU batches small."""

    return 128 if device == "cuda" else 32


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
    ):

        self.device = device or detect_device()

        self.batch_size = int(
            batch_size
            or os.environ.get("RAG_EMBED_BATCH_SIZE")
            or default_batch_size(self.device)
        )

        self.model = SentenceTransformer(
            model_name
            or os.environ.get("RAG_EMBEDDING_MODEL")
            or DEFAULT_MODEL,
            device=self.device,
        )

    def encode(self, texts, show_progress: bool = False):

        options = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }

        if show_progress:
            options["show_progress_bar"] = True

        return self.model.encode(texts, **options)
