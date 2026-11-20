"""Near-topic unsupported questions must be abstained.

These questions are about rotational motion - the same
subject as the sample notes - but the notes do not
contain their answers. They are the hard case for a
RAG system: embeddings retrieve topically close chunks,
so the system must still refuse to answer rather than
improvise from model knowledge.

Uses the real embedding model and reranker (no Ollama
server needed): the generator is a fake that records
whether it was called.
"""

import pytest

from embeddings import EmbeddingModel
from main import NOT_FOUND_MESSAGE, answer_question
from reranker import Reranker
from retriever import Retriever
from vectorstore import VectorStore


CHUNKS = [
    "The moment of inertia of a point mass m at a "
    "perpendicular distance r from the axis of "
    "rotation is I = m r^2. For a system of "
    "particles, I = sum of m_i r_i^2 over all "
    "particles.",
    "Torque is the rotational analogue of force: "
    "tau = r F sin(theta). The SI unit of torque "
    "is the newton metre (N m). A net torque "
    "changes angular momentum: tau = dL/dt.",
    "A rigid body rotating with angular velocity "
    "omega has angular momentum L = I omega. When "
    "no external torque acts, total angular "
    "momentum stays constant.",
]


class RecordingGenerator:
    """Stands in for the LLM; must never be called
    for unsupported questions."""

    def __init__(self):
        self.calls = []

    def generate(self, question, documents):
        self.calls.append(question)
        return "generated answer"


@pytest.fixture(scope="module")
def stack():

    embedding_model = EmbeddingModel()
    store = VectorStore(persist=False)

    store.add_documents(
        [
            {
                "text": text,
                "source": "notes.pdf",
                "page": i + 1,
                "chunk": 0,
            }
            for i, text in enumerate(CHUNKS)
        ],
        embedding_model.encode(CHUNKS),
    )

    retriever = Retriever(
        store, embedding_model, Reranker()
    )

    return retriever


NEAR_TOPIC_UNSUPPORTED = [
    "What is the kinetic energy of a rotating body?",
    "What is the radius of gyration?",
    "State the parallel axis theorem.",
]


@pytest.mark.parametrize(
    "question", NEAR_TOPIC_UNSUPPORTED
)
def test_near_topic_unsupported_question_abstains(
    stack, question
):

    generator = RecordingGenerator()

    answer = answer_question(
        question, stack, generator
    )

    assert answer == NOT_FOUND_MESSAGE
    assert generator.calls == []


def test_supported_question_reaches_generator(stack):

    generator = RecordingGenerator()

    answer = answer_question(
        "What is the moment of inertia of a "
        "point mass?",
        stack,
        generator,
    )

    assert answer == "generated answer"
    assert len(generator.calls) == 1
