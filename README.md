# RAG-Model

> A modular Retrieval-Augmented Generation system for question answering over documents, with support for document ingestion, chunking, embeddings, retrieval, reranking, OCR, formula processing, and vision-assisted analysis.

## Overview

This project is an experimental Retrieval-Augmented Generation (RAG) system designed to answer questions using information retrieved from user-provided documents.

Instead of relying only on a language model's internal knowledge, the system processes documents, creates searchable representations, retrieves relevant information, and uses the retrieved context to generate an answer.

The project also explores document OCR, mathematical formula handling, and vision-based processing for more difficult PDF content.

## Pipeline

```text
Documents
    │
    ▼
Document Loader
    │
    ▼
Text / OCR Extraction
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector Store
    │
    ▼
Retriever
    │
    ▼
Reranker
    │
    ▼
Relevant Context
    │
    ▼
Generator / LLM
    │
    ▼
Answer

Core Components
Document Ingestion
Loads documents and prepares them for downstream processing.
Chunking
Splits extracted content into smaller searchable sections so relevant information can be retrieved efficiently.
Embeddings
Converts document chunks into numerical vector representations for semantic search.
Vector Store
Stores and searches document embeddings.
Retrieval
Finds document chunks that are most relevant to a user's query.
Reranking
Provides an additional relevance stage after initial retrieval to improve the quality of the context passed to the generator.
OCR
Processes document pages and image-based content when normal text extraction is insufficient.
Formula OCR
Includes experimentation with extracting mathematical expressions from document images.
Vision
Provides image-based analysis for document content that cannot be handled reliably through plain text extraction.
Generation
Uses retrieved context to generate answers based on the indexed documents.
Project Structure
RAG-Model/
│
├── src/
│   ├── chat.py
│   ├── chunker.py
│   ├── debug_pdf.py
│   ├── embeddings.py
│   ├── formula_ocr.py
│   ├── generator.py
│   ├── ingest.py
│   ├── loader.py
│   ├── main.py
│   ├── ocr.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── vectorstore.py
│   ├── vision.py
│   └── tests/
│
├── AUTOMATION.md
├── TESTING.md
├── README.md
├── requirements.txt
├── validate.py
└── .env.example

Technologies
- Python
- Retrieval-Augmented Generation
- Semantic embeddings
- Vector search
- Reranking
- PDF processing
- OCR
- Computer vision
- Large language models
- PyTorch / ML tooling where applicable
Local Development
1. Clone the repository
git clone https://github.com/aaravpachouri/RAG-Model.git
cd RAG-Model

2. Create a virtual environment
Windows PowerShell:
python -m venv .venv

3. Activate the environment
.venv\Scripts\Activate.ps1

4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

5. Configure environment variables
Create a local .env file from .env.example.
.env.example
     ↓
.env

Add the credentials required by your selected model or external services.
Never commit .env or other secrets to GitHub.
6. Add your own documents
Place documents you have permission to process inside your local data directory.
The project's development documents and generated artifacts are intentionally excluded from the public repository.
7. Run the system
The exact entry point may depend on the current experiment. The primary source entry point is:
python src/main.py

Testing
The project includes testing and validation utilities.
Run the validation script with:
python validate.py

Additional testing documentation is available in:
TESTING.md

Experiments
The project explores several extensions beyond basic text-only RAG:
- PDF extraction
- OCR-assisted document processing
- Mathematical formula extraction
- Vision-assisted document analysis
- Semantic retrieval
- Reranking
- Local model experimentation
These components are being developed iteratively and may change between versions.
Limitations
This is a research and development project rather than a production-grade document intelligence platform.
Potential limitations include:
- OCR errors on difficult documents
- Retrieval failures
- Incorrect ranking of relevant passages
- Hallucinations from the generation model
- Complex mathematical expressions
- Highly visual document layouts
- Dependence on the capabilities of the selected language model
Results should therefore be verified when accuracy is important.
Data and Privacy
The public repository does not include the private development documents used during local testing.
Do not upload:
- private documents
- confidential information
- copyrighted material that you do not have permission to redistribute
- API keys
- credentials
- personal information
Use your own documents or datasets that you are legally permitted to use.
Project Status
Work in Progress
This repository documents an ongoing exploration of Retrieval-Augmented Generation, document processing, OCR, retrieval, reranking, and multimodal document understanding.
Future Work
- Improve retrieval quality
- Improve reranking
- Improve OCR reliability
- Improve formula extraction
- Improve vision-assisted document understanding
- Add stronger evaluation benchmarks
- Improve reproducibility
- Support additional document formats
- Explore more capable local models
- Reduce unnecessary external dependencies
Author
Aarav Pachouri
Student exploring Artificial Intelligence, Machine Learning, Retrieval-Augmented Generation, and multimodal AI.
GitHub: @aaravpachouri