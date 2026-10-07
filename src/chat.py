from embeddings import EmbeddingModel
from vectorstore import VectorStore
from retriever import Retriever
from reranker import Reranker
from generator import Generator


def main():

    print("=" * 50)
    print("RAG AI")
    print("=" * 50)

    print("\nLoading AI components...")

    embedding_model = EmbeddingModel()

    vector_store = VectorStore()

    reranker = Reranker()

    retriever = Retriever(
        vector_store,
        embedding_model,
        reranker
    )

    generator = Generator()

    print("\nRAG system ready.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nGoodbye!"
            )

            break

        print(
            "\nSearching documents..."
        )

        documents = retriever.retrieve(
            question,
            n_results=8,
            top_k=4
        )

        if not documents:

            print(
                "\nNo relevant information "
                "was found.\n"
            )

            continue

        print(
            f"Retrieved {len(documents)} "
            f"relevant chunks."
        )

        print(
            "\nGenerating answer..."
        )

        answer = generator.generate(
            question,
            documents
        )

        print(
            "\nAI:\n"
        )

        print(answer)

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()