import ollama


class Generator:

    def __init__(
        self,
        model="llama3.2"
    ):

        self.model = model

    def generate(
        self,
        question,
        documents
    ):

        context_parts = []

        for i, document in enumerate(
            documents,
            start=1
        ):

            metadata = document["metadata"]

            source = metadata["source"]
            page = metadata["page"]

            context_parts.append(
                f"""
SOURCE {i}
File: {source}
Page: {page}

{document["text"]}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        prompt = f"""
You are a highly accurate RAG-based
AI study assistant.

Answer the user's question using
the provided sources.

IMPORTANT RULES:

1. Use the provided context first.
2. Do not invent information.
3. If the answer is not available,
   say that the information was not
   found in the documents.
4. For formulas, reproduce the formula
   clearly when it exists in the source.
5. Explain formulas in simple language
   when appropriate.
6. At the end, provide the source file
   and page number.
7. Ignore irrelevant retrieved context.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]