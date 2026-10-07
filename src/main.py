from loader import load_document
from chunker import chunk_text
from embeddings import EmbeddingModel
from vectorstore import VectorStore
from retriever import Retriever
from generator import Generator


def main():

    # Load document
    text = load_document(
        "data/documents/XI_PHY_ROTATIONAL_NOTES.pdf"
    )

    # Split into chunks
    chunks = chunk_text(text)

    print(f"Created {len(chunks)} chunks.")

    # Embeddings
    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode(chunks)

    # Vector database
    vector_store = VectorStore()

    vector_store.add_documents(
        chunks,
        embeddings
    )

    # Retriever
    retriever = Retriever(
        vector_store,
        embedding_model
    )

    # LLM
    generator = Generator()

    print("\nRAG system ready.")

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        documents = retriever.retrieve(
            question,
            n_results=5
        )

        context = "\n\n".join(documents)

        answer = generator.generate(
            question,
            context
        )

        print("\nAI:", answer)


if __name__ == "__main__":
    main()