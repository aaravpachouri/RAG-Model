from loader import (
    load_document,
    load_pdf,
    text_quality_score,
)


def test_quality_score_empty_text():

    assert text_quality_score("") == 0.0
    assert text_quality_score("   ") == 0.0


def test_quality_score_good_text_scores_high():

    text = (
        "The moment of inertia of a rigid body "
        "depends on how its mass is distributed "
        "relative to the axis of rotation. "
    ) * 3

    score = text_quality_score(text)

    assert score >= 0.85


def test_quality_score_tiny_text_scores_low():

    score = text_quality_score("I = mr^2")

    assert score < 0.65


def test_load_txt(tmp_path):

    path = tmp_path / "notes.txt"
    path.write_text("Some study notes.", encoding="utf-8")

    pages = load_document(str(path))

    assert len(pages) == 1
    assert pages[0]["text"] == "Some study notes."
    assert pages[0]["source"] == "notes.txt"


def test_load_pdf_extracts_all_pages(sample_pdf):

    pages = load_pdf(
        str(sample_pdf),
        use_vision=False,
        verbose=False,
    )

    assert len(pages) == 3

    assert pages[0]["page"] == 1
    assert "moment of inertia" in pages[0]["text"].lower()
    assert "torque" in pages[1]["text"].lower()
    assert "angular momentum" in pages[2]["text"].lower()

    for page in pages:
        assert page["source"] == "sample_notes.pdf"
        assert page["extraction_method"] == "text"


def test_load_document_rejects_unknown_type(tmp_path):

    path = tmp_path / "notes.docx"
    path.write_text("x", encoding="utf-8")

    try:
        load_document(str(path))
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError")
