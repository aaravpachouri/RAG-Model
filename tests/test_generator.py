"""Generator prompt and parsing tests - no Ollama server required."""

from generator import (
    Generator,
    build_prompt,
    parse_answer,
    validate_citation,
)


DOCS = [
    {
        "text": "The moment of inertia of a point "
                "mass is I = m r^2.",
        "metadata": {"source": "notes.pdf", "page": 1},
    },
    {
        "text": "Torque is tau = r F sin(theta).",
        "metadata": {"source": "notes.pdf", "page": 2},
    },
]


def test_build_prompt_contains_context_and_question():

    prompt = build_prompt(
        "What is the moment of inertia of a point mass?",
        DOCS,
    )

    assert "I = m r^2" in prompt
    assert "tau = r F sin(theta)" in prompt
    assert "notes.pdf" in prompt
    assert (
        "What is the moment of inertia of a point mass?"
        in prompt
    )


def test_build_prompt_instructs_grounded_answers():

    prompt = build_prompt("q", DOCS)

    assert "do not use your own knowledge" in prompt
    assert "NOT IN DOCUMENTS" in prompt
    assert "SOURCE:" in prompt


def test_generator_model_override():

    generator = Generator(model="test-model")

    assert generator.model == "test-model"


def test_parse_answer_extracts_answer_and_citation():

    raw = (
        "ANSWER: I = m r^2\n"
        "SOURCE: notes.pdf, Page: 1"
    )

    answer, citation = parse_answer(raw)

    assert answer == "I = m r^2"
    assert citation == "notes.pdf, Page: 1"


def test_parse_answer_normalises_abstention():

    raw = (
        "ANSWER: Not in the document\n"
        "SOURCE: made_up.pdf\n"
    )

    answer, citation = parse_answer(raw)

    assert "could not find" in answer
    assert citation is None


def test_parse_answer_handles_unformatted_reply():

    answer, citation = parse_answer(
        "Just a plain answer."
    )

    assert answer == "Just a plain answer."
    assert citation is None


def test_parse_answer_strips_quote_scaffolding():

    raw = (
        "ANSWER: I = m r^2\n"
        "QUOTE: The moment of inertia is I = m r^2."
    )

    answer, citation = parse_answer(raw)

    assert answer == "I = m r^2"
    assert citation is None


def test_parse_answer_strips_quote_before_source():

    raw = (
        "ANSWER: I = m r^2\n"
        "QUOTE: The moment of inertia is I = m r^2.\n"
        "SOURCE: notes.pdf, Page: 1"
    )

    answer, citation = parse_answer(raw)

    assert answer == "I = m r^2"
    assert citation == "notes.pdf, Page: 1"


def test_generate_verified_quote_hides_scaffolding(
    monkeypatch,
):
    """End-to-end formatting: the reply's ANSWER:/QUOTE:
    lines never leak into the shown answer, and a quote
    that verifies earns the Evidence label."""

    import sys
    import types

    raw = (
        "ANSWER: I = m r^2\n"
        "QUOTE: The moment of inertia of a point "
        "mass is I = m r^2."
    )

    fake_ollama = types.SimpleNamespace(
        chat=lambda **kwargs: {
            "message": {"content": raw}
        }
    )
    monkeypatch.setitem(
        sys.modules, "ollama", fake_ollama
    )

    result = Generator(model="m").generate("q", DOCS)

    assert "ANSWER:" not in result
    assert "QUOTE:" not in result
    assert "Evidence (verified quote" in result
    assert "notes.pdf" in result


def test_validate_citation_keeps_real_source():

    citation = validate_citation(
        "notes.pdf, Page: 1", DOCS
    )

    assert citation == "notes.pdf, Page: 1"


def test_validate_citation_drops_invented_source():

    assert (
        validate_citation(
            "made_up_file.pdf, Page: 9", DOCS
        )
        is None
    )


def test_validate_citation_drops_wrong_page():

    # notes.pdf was retrieved, but page 7 was not.
    assert (
        validate_citation(
            "notes.pdf, Page: 7", DOCS
        )
        is None
    )


def test_validate_citation_keeps_retrieved_page():

    assert (
        validate_citation(
            "notes.pdf, Page: 2", DOCS
        )
        == "notes.pdf, Page: 2"
    )


def test_validate_citation_keeps_filename_only():

    assert (
        validate_citation("notes.pdf", DOCS)
        == "notes.pdf"
    )


def test_validate_citation_none_passthrough():

    assert validate_citation(None, DOCS) is None


def test_parse_empty_answer_and_source():
    for raw in ("", "ANSWER: \nSOURCE:", "ANSWER: \nSOURCE: notes.pdf"):
        answer, citation = parse_answer(raw)
        assert "returned no answer" in answer
        assert citation is None


def test_extract_and_verify_quote():
    from generator import extract_quote, verify_quote

    raw = (
        "ANSWER: Three members.\n"
        "QUOTE: A minimum of three members must be present.\n"
        "SOURCE: notes.pdf, page 20\n"
    )
    assert extract_quote(raw) == "A minimum of three members must be present."

    docs = [{
        "text": "Filler words. A minimum of three\nmembers must be present. More.",
        "metadata": {"source": "notes.pdf", "page": 20},
    }]
    found = verify_quote(extract_quote(raw), docs)
    assert found is docs[0]

    # Not copied from context: no verification.
    assert verify_quote("An invented sentence that is not in any chunk.", docs) is None
    # Too short to prove anything.
    assert verify_quote("three members", docs) is None
    assert extract_quote("ANSWER: x\nSOURCE: notes.pdf") is None
    assert extract_quote("ANSWER: x\nQUOTE: NONE\nSOURCE: notes.pdf") is None


def test_extract_quote_strips_wrapping_marks():
    from generator import extract_quote, verify_quote
    raw = 'ANSWER: x\nQUOTE: "A copied sentence long enough to count here."\nSOURCE: n.pdf'
    docs = [{"text": "Lead in. A copied sentence long enough to count here. Tail.",
             "metadata": {"source": "n.pdf", "page": 1}}]
    assert verify_quote(extract_quote(raw), docs) is docs[0]
