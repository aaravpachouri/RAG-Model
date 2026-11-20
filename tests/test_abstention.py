"""Retrieval-confidence gate tests - no models required."""

import os

from main import (
    NOT_FOUND_MESSAGE,
    answer_question,
    retrieval_is_confident,
)


def doc(score):
    return {
        "text": "chunk",
        "metadata": {"source": "n.pdf", "page": 1},
        "rerank_score": score,
    }


def test_confident_when_best_score_positive():

    assert retrieval_is_confident(
        [doc(8.5), doc(-10.0)]
    )


def test_not_confident_when_all_scores_negative():

    assert not retrieval_is_confident(
        [doc(-10.9), doc(-11.2)]
    )


def test_confident_without_reranker_scores():

    plain = {
        "text": "chunk",
        "metadata": {"source": "n.pdf", "page": 1},
    }

    assert retrieval_is_confident([plain])


def test_threshold_is_configurable(monkeypatch):

    monkeypatch.setenv("RAG_ABSTAIN_THRESHOLD", "5")

    assert not retrieval_is_confident([doc(2.5)])
    assert retrieval_is_confident([doc(6.0)])


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, question, n_results, top_k):
        return self.documents


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, question, documents):
        self.calls += 1
        return "generated answer"


def test_answer_question_abstains_without_calling_llm():

    generator = FakeGenerator()

    answer = answer_question(
        "off topic?",
        FakeRetriever([doc(-11.0)]),
        generator,
    )

    assert answer == NOT_FOUND_MESSAGE
    assert generator.calls == 0


def test_answer_question_generates_when_confident():

    generator = FakeGenerator()

    answer = answer_question(
        "on topic?",
        FakeRetriever([doc(8.0)]),
        generator,
    )

    assert answer == "generated answer"
    assert generator.calls == 1


def test_answer_question_empty_results():

    answer = answer_question(
        "anything?",
        FakeRetriever([]),
        FakeGenerator(),
    )

    assert answer == NOT_FOUND_MESSAGE


def test_on_passages_called_only_when_confident():

    seen = []

    answer = answer_question(
        "on topic?",
        FakeRetriever([doc(8.0)]),
        FakeGenerator(),
        on_passages=seen.append,
    )

    assert answer == "generated answer"
    assert len(seen) == 1

    seen.clear()

    answer = answer_question(
        "off topic?",
        FakeRetriever([doc(-11.0)]),
        FakeGenerator(),
        on_passages=seen.append,
    )

    assert answer == NOT_FOUND_MESSAGE
    assert seen == []


def test_format_passages_labels_source_and_page():

    from main import format_passages

    output = format_passages([
        {
            "text": "Some   spaced\nout   text.",
            "metadata": {
                "source": "notes.pdf",
                "page": 7,
            },
        }
    ])

    assert "RETRIEVED PASSAGES" in output
    assert "notes.pdf, page 7" in output
    assert "Some spaced out text." in output


def test_retrieval_only_never_calls_generator():
    generator = FakeGenerator()
    output = answer_question(
        "on topic?", FakeRetriever([doc(8.0)]), generator,
        retrieve_only=True
    )
    assert "RETRIEVED PASSAGES" in output
    assert "n.pdf, page 1" in output
    assert generator.calls == 0
