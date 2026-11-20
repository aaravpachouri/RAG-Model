from chunker import chunk_text, chunk_pages


def test_chunk_text_splits_into_chunks():

    text = " ".join(f"word{i}" for i in range(1000))

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1

    for chunk in chunks:
        assert 0 < len(chunk.split()) <= 100


def test_chunk_text_overlap_shares_words():

    text = " ".join(f"w{i}" for i in range(200))

    chunks = chunk_text(text, chunk_size=100, overlap=50)

    # With 50 words of overlap, the second chunk starts
    # at word 50, so its first word appears in chunk 1.
    first_words = set(chunks[0].split())
    assert chunks[1].split()[0] in first_words


def test_chunk_text_empty_input():

    assert chunk_text("") == []


def test_chunk_pages_preserves_metadata():

    pages = [
        {
            "text": "alpha beta gamma " * 5,
            "page": 3,
            "source": "notes.pdf",
            "extraction_method": "text",
        }
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) >= 1

    for i, chunk in enumerate(chunks):
        assert chunk["page"] == 3
        assert chunk["source"] == "notes.pdf"
        assert chunk["chunk"] == i
        assert chunk["text"]
