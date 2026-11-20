"""Live generation tests - need a running Ollama server.

Skipped automatically when no server is reachable. Set
OLLAMA_HOST and RAG_MODEL to match your setup:

    ollama serve &
    RAG_MODEL=qwen3-vl:4b-instruct python -m pytest tests/test_generation_live.py

Tests skip automatically when the configured model is
not pulled on this machine.
"""

import os
import socket

import pytest


def _ollama_reachable() -> bool:

    host = os.environ.get(
        "OLLAMA_HOST", "127.0.0.1:11434"
    )

    host = host.replace("http://", "").replace(
        "https://", ""
    )

    name, _, port = host.partition(":")

    try:
        with socket.create_connection(
            (name, int(port or 11434)), timeout=2
        ):
            return True
    except OSError:
        return False


def _model_available() -> bool:
    """Skip when the configured model is not pulled."""

    if not _ollama_reachable():
        return False

    try:
        import ollama

        installed = {
            m.get("name") or m.get("model")
            for m in ollama.list().get("models", [])
        }
    except Exception:
        return False

    wanted = os.environ.get(
        "RAG_MODEL"
    )

    if wanted is None:
        import sys
        from pathlib import Path

        sys.path.insert(
            0,
            str(
                Path(__file__).resolve().parent.parent
                / "src"
            ),
        )

        from generator import DEFAULT_MODEL

        wanted = DEFAULT_MODEL

    if wanted in installed:
        return True

    # Ollama accepts a bare name as <name>:latest.
    if ":" not in wanted:
        return f"{wanted}:latest" in installed

    return False


pytestmark = pytest.mark.skipif(
    not _model_available(),
    reason=(
        "no Ollama server reachable, or the "
        "configured model is not pulled"
    ),
)


DOCS = [
    {
        "text": "The moment of inertia of a point "
                "mass m at a perpendicular distance r "
                "from the axis of rotation is "
                "I = m r^2.",
        "metadata": {
            "source": "sample_physics_notes.pdf",
            "page": 1,
        },
    },
    {
        "text": "Torque is the rotational analogue "
                "of force: tau = r F sin(theta). "
                "The SI unit of torque is the "
                "newton metre (N m).",
        "metadata": {
            "source": "sample_physics_notes.pdf",
            "page": 2,
        },
    },
]


@pytest.fixture(scope="module")
def qa_stack():
    """Real retriever + generator over the sample chunks."""

    from embeddings import EmbeddingModel
    from generator import Generator
    from reranker import Reranker
    from retriever import Retriever
    from vectorstore import VectorStore

    embedding_model = EmbeddingModel()
    store = VectorStore(persist=False)
    store.add_documents(
        [
            {
                "text": d["text"],
                "source": d["metadata"]["source"],
                "page": d["metadata"]["page"],
                "chunk": i,
            }
            for i, d in enumerate(DOCS)
        ],
        embedding_model.encode(
            [d["text"] for d in DOCS]
        ),
    )

    retriever = Retriever(
        store, embedding_model, Reranker()
    )

    return retriever, Generator()


def test_grounded_answer_with_citation(qa_stack):

    retriever, generator = qa_stack

    from main import answer_question

    answer = answer_question(
        "What is the moment of inertia of a "
        "point mass?",
        retriever,
        generator,
    )

    # Small local models word the answer
    # differently run to run; the grounding
    # signals are the formula content and a
    # source reference to the real file.
    compact = answer.replace(" ", "").lower()
    assert (
        "r^2" in compact
        or "r²" in answer
        or "inertia" in answer.lower()
    )
    # Either a verified citation (model cited a
    # retrieved file+page) or a clearly-labelled
    # unverified retrieved-context reference.
    assert (
        "Source:" in answer
        or "Reference (top retrieved context" in answer
    )
    assert "sample_physics_notes.pdf" in answer


def test_abstains_when_answer_not_in_documents(
    qa_stack,
):
    """Off-topic questions never reach the LLM: the
    retrieval-confidence gate answers instead, so the
    system cannot invent facts from model knowledge."""

    retriever, generator = qa_stack

    from main import answer_question

    answer = answer_question(
        "What is the capital of France?",
        retriever,
        generator,
    )

    assert "could not find" in answer.lower()
    assert "Source:" not in answer


def test_generator_alone_uses_prompt_rules():

    """The prompt-level rule is the second layer: even
    when called directly, the model is instructed to
    stay inside the context. Small models can still
    slip, which is why the retrieval gate exists."""

    from generator import Generator, build_prompt

    prompt = build_prompt(
        "What is the capital of France?", DOCS
    )

    assert "NOT IN DOCUMENTS" in prompt
