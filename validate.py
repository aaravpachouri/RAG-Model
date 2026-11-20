"""Validation workflow for the RAG system - checks all components.

Uses an in-memory vector store, so it never touches the real index
under data/chroma. The generation check needs a running Ollama server
and is reported as SKIP when Ollama is not available.

Run from the project root:
    python validate.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chunker import chunk_pages
from embeddings import EmbeddingModel
from generator import Generator
from loader import load_document, text_quality_score
from reranker import Reranker
from retriever import Retriever
from vectorstore import VectorStore


def make_chunk(text, page=1, source="sample.txt", chunk=0):
    """Build a chunk dict in the shape the pipeline uses."""
    return {
        "text": text,
        "page": page,
        "source": source,
        "chunk": chunk,
    }


SAMPLE_CHUNKS = [
    make_chunk(
        "The moment of inertia of a point mass m at "
        "distance r from the axis is I = m r^2.",
        page=1,
    ),
    make_chunk(
        "Torque is the rotational analogue of force: "
        "tau = r F sin(theta), measured in N m.",
        page=2,
    ),
    make_chunk(
        "When no external torque acts, total angular "
        "momentum L = I omega is conserved.",
        page=3,
    ),
]


class ValidationWorkflow:
    """Automated validation workflow for the RAG system."""

    def __init__(self):
        self.results = []
        self.start_time = None

    def log_step(self, step_name, status, details=""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbol = {
            "PASS": "PASS",
            "FAIL": "FAIL",
            "SKIP": "SKIP",
        }[status]
        print(f"[{timestamp}] {symbol} {step_name}")
        if details:
            print(f"  -> {details}")
        self.results.append((step_name, status, details))

    def validate_embeddings(self):
        print("\nSTEP 1: Embedding model")

        try:
            model = EmbeddingModel()
            self.log_step("Embedding model loads", "PASS")

            texts = [
                "Hello world",
                "This is a test",
                "RAG system validation",
            ]
            embeddings = model.encode(texts)

            if len(embeddings) == len(texts):
                self.log_step(
                    "Encoding multiple texts", "PASS",
                    f"{len(embeddings)} embeddings, "
                    f"dimension {len(embeddings[0])}"
                )
            else:
                self.log_step(
                    "Encoding multiple texts", "FAIL",
                    f"Expected {len(texts)}, "
                    f"got {len(embeddings)}"
                )

        except Exception as e:
            self.log_step("Embedding model", "FAIL", str(e))

    def validate_vectorstore(self):
        print("\nSTEP 2: Vector store")

        try:
            model = EmbeddingModel()
            store = VectorStore(persist=False)
            self.log_step("Vector store initialises", "PASS")

            embeddings = model.encode(
                [c["text"] for c in SAMPLE_CHUNKS]
            )
            store.add_documents(SAMPLE_CHUNKS, embeddings)

            if store.count() == len(SAMPLE_CHUNKS):
                self.log_step(
                    "Adding chunk documents", "PASS",
                    f"Stored {store.count()} chunks"
                )
            else:
                self.log_step(
                    "Adding chunk documents", "FAIL",
                    f"Count is {store.count()}"
                )

            query_emb = model.encode(
                ["moment of inertia formula"]
            )[0]
            results = store.search(query_emb, n_results=2)

            if (
                results
                and "moment of inertia"
                in results[0]["text"].lower()
            ):
                self.log_step(
                    "Semantic search", "PASS",
                    "Relevant chunk ranked first"
                )
            else:
                self.log_step(
                    "Semantic search", "FAIL",
                    "Unexpected top result"
                )

        except Exception as e:
            self.log_step("Vector store", "FAIL", str(e))

    def validate_retriever(self):
        print("\nSTEP 3: Retriever and reranker")

        try:
            model = EmbeddingModel()
            store = VectorStore(persist=False)
            reranker = Reranker()
            retriever = Retriever(store, model, reranker)
            self.log_step(
                "Retriever and reranker initialise", "PASS"
            )

            embeddings = model.encode(
                [c["text"] for c in SAMPLE_CHUNKS]
            )
            store.add_documents(SAMPLE_CHUNKS, embeddings)

            results = retriever.retrieve(
                "what is the SI unit of torque",
                n_results=3,
                top_k=2,
            )

            if (
                results
                and results[0]["metadata"]["page"] == 2
                and "rerank_score" in results[0]
            ):
                self.log_step(
                    "Retrieve and rerank", "PASS",
                    "Torque chunk ranked first"
                )
            else:
                self.log_step(
                    "Retrieve and rerank", "FAIL",
                    "Unexpected ranking"
                )

        except Exception as e:
            self.log_step(
                "Retriever validation", "FAIL", str(e)
            )

    def validate_loader(self):
        print("\nSTEP 4: Document loader")

        try:
            good = (
                "The moment of inertia of a rigid body "
                "depends on the mass distribution "
                "relative to the axis of rotation. "
            ) * 3

            if (
                text_quality_score("") == 0.0
                and text_quality_score(good) >= 0.85
                and text_quality_score("I = mr^2") < 0.65
            ):
                self.log_step(
                    "Text quality scoring", "PASS"
                )
            else:
                self.log_step(
                    "Text quality scoring", "FAIL",
                    "Unexpected scores"
                )

            pages = [
                {
                    "text": "alpha beta gamma " * 10,
                    "page": 7,
                    "source": "x.pdf",
                    "extraction_method": "text",
                }
            ]
            chunks = chunk_pages(pages)

            if (
                chunks
                and chunks[0]["page"] == 7
                and chunks[0]["source"] == "x.pdf"
            ):
                self.log_step(
                    "Page chunking keeps metadata", "PASS"
                )
            else:
                self.log_step(
                    "Page chunking keeps metadata", "FAIL"
                )

        except Exception as e:
            self.log_step("Loader validation", "FAIL", str(e))

    def validate_generator(self):
        print("\nSTEP 5: Generator (needs Ollama)")

        try:
            generator = Generator()
            self.log_step(
                "Generator initialises", "PASS",
                f"model: {generator.model}"
            )

            try:
                answer = generator.generate(
                    "What is the moment of inertia "
                    "of a point mass?",
                    SAMPLE_CHUNKS[:1],
                )
            except RuntimeError:
                self.log_step(
                    "Text generation", "SKIP",
                    "Ollama is not running or the "
                    "model is not pulled; install "
                    "Ollama and run "
                    f"`ollama pull {generator.model}`"
                )
                return

            if answer and answer.strip():
                self.log_step(
                    "Text generation", "PASS",
                    f"{len(answer)} characters"
                )
            else:
                self.log_step(
                    "Text generation", "FAIL",
                    "Empty response"
                )

        except Exception as e:
            self.log_step(
                "Generator validation", "FAIL", str(e)
            )

    def generate_report(self):
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        passed = sum(
            1 for _, s, _ in self.results if s == "PASS"
        )
        failed = sum(
            1 for _, s, _ in self.results if s == "FAIL"
        )
        skipped = sum(
            1 for _, s, _ in self.results if s == "SKIP"
        )
        total = len(self.results)

        print(f"Total checks: {total}")
        print(f"Passed:       {passed}")
        print(f"Failed:       {failed}")
        print(f"Skipped:      {skipped}")
        print("=" * 60)

        report_path = (
            Path(__file__).parent
            / "validation_report.txt"
        )
        with open(report_path, "w") as f:
            f.write("RAG SYSTEM VALIDATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(
                "Generated: "
                + datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                + "\n\n"
            )
            for step, status, details in self.results:
                f.write(f"[{status}] {step}\n")
                if details:
                    f.write(f"  Details: {details}\n")
            f.write(
                f"\nPassed {passed}/{total}, "
                f"failed {failed}, skipped {skipped}\n"
            )

        print(f"Report saved to: {report_path}\n")

    def run(self):
        self.start_time = time.time()

        print("=" * 60)
        print("RAG SYSTEM VALIDATION")
        print("=" * 60)

        self.validate_embeddings()
        self.validate_vectorstore()
        self.validate_retriever()
        self.validate_loader()
        self.validate_generator()

        self.generate_report()

        duration = time.time() - self.start_time
        print(f"Completed in {duration:.1f} seconds\n")

        failed = sum(
            1 for _, s, _ in self.results if s == "FAIL"
        )
        return failed == 0


def main():
    workflow = ValidationWorkflow()
    success = workflow.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
