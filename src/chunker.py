def chunk_text(
    text,
    chunk_size=700,
    overlap=100
):
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_pages(pages):

    all_chunks = []

    for page in pages:

        chunks = chunk_text(
            page["text"]
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": page["source"],
                "chunk": index
            })

    return all_chunks