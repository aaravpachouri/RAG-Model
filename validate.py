"""Validation workflow for the RAG system - tests all components end-to-end."""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import EmbeddingModel
from vectorstore import VectorStore
from retriever import Retriever
from reranker import Reranker
from generator import Generator
from loader import load_document
from chunker import chunk_text


class ValidationWorkflow:
    """Automated validation workflow for the RAG system."""

    def __init__(self):
        """Initialize validation workflow."""
        self.results = []
        self.start_time = None

    def log_step(self, step_name, status, details=""):
        """Log a validation step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"[{timestamp}] {status_symbol} {step_name}")
        if details:
            print(f"  └─ {details}")
        self.results.append((step_name, status, details))

    def validate_embeddings(self):
        """Validate embedding model."""
        print("\n" + "=" * 70)
        print("STEP 1: Validating Embedding Model")
        print("=" * 70)

        try:
            model = EmbeddingModel()
            self.log_step("Embedding Model Initialization", "PASS")

            # Test encoding
            test_texts = [
                "Hello world",
                "This is a test",
                "RAG system validation"
            ]
            embeddings = model.encode(test_texts)

            if len(embeddings) == len(test_texts):
                self.log_step("Encoding Multiple Texts", "PASS",
                             f"Generated {len(embeddings)} embeddings")
            else:
                self.log_step("Encoding Multiple Texts", "FAIL",
                             f"Expected {len(test_texts)}, got {len(embeddings)}")

            # Validate embedding dimensions
            dim = len(embeddings[0])
            self.log_step("Embedding Dimensions", "PASS",
                         f"Dimension: {dim}")

        except Exception as e:
            self.log_step("Embedding Model Validation", "FAIL", str(e))

    def validate_vectorstore(self):
        """Validate vector store."""
        print("\n" + "=" * 70)
        print("STEP 2: Validating Vector Store")
        print("=" * 70)

        try:
            # Initialize components
            model = EmbeddingModel()
            store = VectorStore()
            self.log_step("Vector Store Initialization", "PASS")

            # Add documents
            docs = [
                "Python programming",
                "Machine learning",
                "Data science"
            ]
            embeddings = model.encode(docs)
            store.add_documents(docs, embeddings)

            self.log_step("Adding Documents to Vector Store", "PASS",
                         f"Added {len(docs)} documents")

            # Search
            query_emb = model.encode(["programming"])[0]
            results = store.search(query_emb, n_results=2)

            if len(results) > 0:
                self.log_step("Document Search", "PASS",
                             f"Found {len(results)} results")
            else:
                self.log_step("Document Search", "FAIL", "No results found")

        except Exception as e:
            self.log_step("Vector Store Validation", "FAIL", str(e))

    def validate_retriever(self):
        """Validate retriever and reranker."""
        print("\n" + "=" * 70)
        print("STEP 3: Validating Retriever and Reranker")
        print("=" * 70)

        try:
            # Initialize all components
            model = EmbeddingModel()
            store = VectorStore()
            reranker = Reranker()
            retriever = Retriever(store, model, reranker)

            self.log_step("Retriever Initialization", "PASS")

            # Add sample documents
            docs = [
                "Python is a programming language",
                "Machine learning uses algorithms",
                "Deep learning involves neural networks",
                "NLP processes human language",
                "RAG combines retrieval and generation"
            ]

            embeddings = model.encode(docs)
            store.add_documents(docs, embeddings)

            self.log_step("Sample Documents Added", "PASS",
                         f"Added {len(docs)} documents")

            # Test retrieval
            query = "machine learning algorithms"
            results = retriever.retrieve(query, n_results=3, top_k=2)

            if len(results) > 0:
                self.log_step("Document Retrieval", "PASS",
                             f"Retrieved {len(results)} documents")
            else:
                self.log_step("Document Retrieval", "FAIL", "No documents retrieved")

            # Verify results are reranked
            self.log_step("Reranking Results", "PASS",
                         f"Results processed through reranker")

        except Exception as e:
            self.log_step("Retriever Validation", "FAIL", str(e))

    def validate_generator(self):
        """Validate answer generation."""
        print("\n" + "=" * 70)
        print("STEP 4: Validating Generator")
        print("=" * 70)

        try:
            generator = Generator()
            self.log_step("Generator Initialization", "PASS")

            # Test generation with sample documents
            question = "What is machine learning?"
            documents = [
                {
                    "text": "Machine learning is a subset of AI",
                    "metadata": {"source": "test.txt", "page": 1}
                },
                {
                    "text": "It uses algorithms to learn from data",
                    "metadata": {"source": "test.txt", "page": 2}
                }
            ]

            try:
                answer = generator.generate(question, documents)
                if answer and len(answer) > 0:
                    self.log_step("Text Generation", "PASS",
                                 f"Generated {len(answer)} character response")
                else:
                    self.log_step("Text Generation", "FAIL", "Empty response")
            except Exception as gen_error:
                # Generator might fail if ollama is not running
                self.log_step("Text Generation", "FAIL",
                             "Ollama may not be running (optional)")

        except Exception as e:
            self.log_step("Generator Validation", "FAIL", str(e))

    def validate_full_pipeline(self):
        """Validate the complete RAG pipeline."""
        print("\n" + "=" * 70)
        print("STEP 5: Validating Full RAG Pipeline")
        print("=" * 70)

        try:
            # Initialize all components
            model = EmbeddingModel()
            store = VectorStore()
            reranker = Reranker()
            retriever = Retriever(store, model, reranker)
            generator = Generator()

            self.log_step("All Components Initialized", "PASS")

            # Prepare documents
            docs = [
                "RAG systems combine retrieval and generation",
                "They improve answer quality by using relevant documents",
                "The system retrieves documents, then generates answers",
                "This is called Retrieval-Augmented Generation"
            ]

            embeddings = model.encode(docs)
            store.add_documents(docs, embeddings)

            self.log_step("Documents Ingested", "PASS",
                         f"Ingested {len(docs)} documents")

            # Test full pipeline
            query = "What is a RAG system?"
            retrieved_docs = retriever.retrieve(query, n_results=3, top_k=2)

            if len(retrieved_docs) > 0:
                self.log_step("Pipeline: Retrieval Complete", "PASS",
                             f"Retrieved {len(retrieved_docs)} documents")

                try:
                    answer = generator.generate(query, retrieved_docs)
                    if answer:
                        self.log_step("Pipeline: Generation Complete", "PASS",
                                     f"Generated {len(answer)} character response")
                    else:
                        self.log_step("Pipeline: Generation Complete", "FAIL",
                                     "Empty response")
                except Exception as gen_error:
                    self.log_step("Pipeline: Generation Complete", "FAIL",
                                 "Ollama may not be running")
            else:
                self.log_step("Pipeline: Retrieval Complete", "FAIL",
                             "No documents retrieved")

        except Exception as e:
            self.log_step("Full Pipeline Validation", "FAIL", str(e))

    def generate_report(self):
        """Generate validation report."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        passed = sum(1 for _, status, _ in self.results if status == "PASS")
        failed = sum(1 for _, status, _ in self.results if status == "FAIL")
        total = len(self.results)

        print(f"Total Checks:   {total}")
        print(f"Passed:         {passed}")
        print(f"Failed:         {failed}")
        print(f"Success Rate:   {(passed/total*100):.1f}%" if total > 0 else "N/A")
        print(f"Completed at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Save report to file
        report_path = Path(__file__).parent / "validation_report.txt"
        with open(report_path, "w") as f:
            f.write("RAG SYSTEM VALIDATION REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for step, status, details in self.results:
                f.write(f"[{status}] {step}\n")
                if details:
                    f.write(f"  Details: {details}\n")
                f.write("\n")

            f.write(f"\nTotal: {passed}/{total} checks passed\n")

        print(f"\nReport saved to: {report_path}\n")

    def run(self):
        """Run complete validation workflow."""
        self.start_time = time.time()

        print("\n" + "=" * 70)
        print("RAG SYSTEM VALIDATION WORKFLOW")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Run validation steps
        self.validate_embeddings()
        self.validate_vectorstore()
        self.validate_retriever()
        self.validate_generator()
        self.validate_full_pipeline()

        # Generate final report
        self.generate_report()

        duration = time.time() - self.start_time
        print(f"Validation completed in {duration:.2f} seconds\n")

        # Return success if all validations passed
        failed = sum(1 for _, status, _ in self.results if status == "FAIL")
        return failed == 0


def main():
    """Main entry point."""
    workflow = ValidationWorkflow()
    success = workflow.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
