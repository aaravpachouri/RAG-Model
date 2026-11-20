# RAG-Model

A Retrieval-Augmented Generation system for asking questions over your own study notes. Point it at a folder of PDFs (for example class notes), and it indexes them, finds the passages relevant to your question, and offers an optional, unverified generated explanation alongside the retrieved text and page references. Everything runs locally: no cloud API keys, no usage costs.

## How it works

```text
PDF / TXT notes
      │
      ▼
Loader        text extraction per page (PyMuPDF), with optional
              vision-model extraction for pages where normal
              extraction fails
      │
      ▼
Chunker       splits pages into overlapping word-based chunks
      │
      ▼
Embeddings    all-MiniLM-L6-v2 sentence embeddings
      │
      ▼
Vector store  ChromaDB, persisted under data/chroma
      │
      ▼
Retriever     semantic search for the question
      │
      ▼
Reranker      cross-encoder (ms-marco-MiniLM-L-6-v2) reorders
              the candidates for relevance
      │
      ▼
Generator     local LLM via Ollama answers using only the
              retrieved sources, and must copy the exact
              sentence(s) it relies on as a quote
```

The reranker matters: with embedding-only search, short note chunks from the same document often score within a few percent of each other. The cross-encoder reranker reads the question and each candidate together and is much better at picking the right page.

The system is designed to abstain rather than improvise. When the reranker reports that no retrieved chunk actually matches the question, the system answers that the information was not found and does not call the LLM. This matters because small local models will otherwise answer from their own training knowledge even when told not to. On the sample notes, reranker confidence separates cleanly (on-topic questions score clearly positive; off-topic and near-topic-but-unsupported questions score negative). The threshold is configurable with `RAG_ABSTAIN_THRESHOLD` (default 0).

Two caveats keep this honest: the separation margin depends on your notes, so a question whose answer is missing but which is very close to the covered material can still pass the gate, and then the answer is only as reliable as the model. Citations and evidence are enforced by the system rather than left to model compliance, and the kinds are labelled differently. The prompt asks the model to copy the exact sentence(s) its answer relies on. When that quote is found verbatim in a retrieved chunk, the answer shows `Evidence (verified quote from <file>, page N - check the answer against it)`. Verified means the words really exist in the retrieved context; it still does not prove they support the answer, so the label says to check. When no quote verifies, the citation rules apply: A line starting `Source:` means the model's citation was verified against the retrieved set: it names a file that was actually retrieved and, if it states a page, a page that was actually retrieved (invented files and pages are dropped). When the model omits a citation or cites something that fails verification, the answer instead carries `Reference (top retrieved context, unverified)`: the most relevant passage retrieval found, which may not be the passage the answer actually relies on. Even a verified citation only proves the passage was retrieved, not that it supports the answer, so check the cited page when accuracy matters.

Because generation is the weak link with small local models, `ask` and `chat` print the full retrieved passages themselves, extracted text with whitespace normalized and file and page, above every answer, and label the answer as unverified. The passages show what retrieval found, not a guarantee of relevance, completeness or extraction fidelity. The generated text is a convenience layer to check against the document. Use `ask "question" --retrieve-only` for passages without loading a generation model or requiring Ollama.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) for answer generation (optional for ingestion/retrieval). The default model is qwen3-vl:4b-instruct, a vision-capable Qwen that also covers image pages below; it needs Ollama 0.12.7 or newer:
  ```bash
  ollama pull qwen3-vl:4b-instruct
  ```
  Any other Ollama chat model works for generation instead (llama3.2 needs Ollama 0.3.12 or newer); set `RAG_MODEL` to switch.
- Optional, for pages where text extraction fails, a separate vision model (qwen3-vl or qwen2.5vl) if you do not use the default for both jobs.
- Optional, for the OCR experiments: [Tesseract](https://github.com/tesseract-ocr/tesseract)

## Setup

```bash
git clone https://github.com/aaravpachouri/RAG-Model.git
cd RAG-Model
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Configuration is optional. Copy `.env.example` to `.env` (or export the variables) to change the models or the Tesseract path. No API keys are needed.

## Usage

Put your PDF or TXT notes in `data/documents/`, then:

```bash
# Index your notes
python src/main.py ingest

# Ask one question
python src/main.py ask "What is the moment of inertia of a point mass?"

# Interactive session
python src/main.py chat
```

Running `python src/main.py` with no arguments ingests first if the store is empty, then starts the chat.

Flags: `--no-vision` disables vision extraction during ingestion (faster, text-only), `--no-rerank` skips the reranking stage, `--reset` forces a full rebuild of the index. After successful extraction and embedding, a plain `ingest` upserts the current chunks and prunes IDs no longer in the folder, including trailing chunks of shortened files. Extraction failures abort before index writes. An empty folder preserves the old index unless `--reset` explicitly clears it. Synchronization is not transactional: an interrupted write can leave a mixed index; rerun ingestion before querying it. Use `--reset` after changing chunking settings or if the index seems corrupt.

### Try it without your own notes

```bash
python scripts/create_sample_pdf.py   # writes sample notes to data/documents/
python src/main.py ingest
python src/main.py ask "how is torque defined and what is its SI unit?"
```

The sample notes are short original physics notes generated for this project.

## Already have a Qwen installed?

A vision-capable Qwen (qwen3-vl, qwen2.5vl) can do both jobs in this project: answer questions and read page images. The default configuration already assumes qwen3-vl:4b-instruct for both. If `ollama list` shows a different tag on your machine, set both variables to that tag and nothing new downloads:

```bash
# bash
export RAG_MODEL=qwen3-vl:4b-instruct
export RAG_VISION_MODEL=qwen3-vl:4b-instruct

# Windows PowerShell
$env:RAG_MODEL="qwen3-vl:4b-instruct"
$env:RAG_VISION_MODEL="qwen3-vl:4b-instruct"
```

Not sure what is installed or which tag to use? Run the read-only checker; it lists your models, flags vision-capable ones, warns if qwen3-vl needs a newer Ollama, and prints the exact settings to copy:

```bash
python scripts/check_setup.py
```

Vision model download sizes: qwen3-vl:2b 1.9 GB, qwen3-vl:4b-instruct 3.3 GB, qwen3-vl:8b-instruct 6.2 GB, qwen2.5vl:3b 3.2 GB, qwen2.5vl:7b 6.0 GB (ollama.com library pages). The same trial advice from the evaluation section applies: download size is not runtime memory, and vision models carry an extra image encoder, so start smaller and watch `ollama ps`.

## Evaluating on your own notes

```bash
python src/main.py ingest
python scripts/evaluate.py              # retrieval accuracy
python scripts/evaluate.py --generate   # plus generation signals
```

The evaluator pulls Q&A-style headings out of your indexed notes (FAQ/handbook material), treats each heading's page as the expected answer page, and measures how often retrieval ranks that page first or in the top 4. Because those questions come from the same document being searched, this is a same-document sanity check of retrieval, not independent held-out accuracy. For other material, pass `--questions file.json` with `{"question": ..., "page": N}` entries you verified yourself. Generation cannot be auto-marked: `--generate` reports mechanical signals only (verified quote, validated citation, unverified reference, abstention) and prints every answer for you to read. A verified quote is not a correctness mark; it only proves the words exist in the retrieved context. On a 52-page Q&A handbook used during development, the same-document check scored 11/12 top-1 and 12/12 top-4.

Model choice matters for the generation step. Download sizes: llama3.2:3b is 2.0 GB, qwen2.5:3b 1.9 GB, qwen2.5:7b 4.7 GB (ollama.com library pages). Download size is not runtime memory: a running model also needs working memory for its context, and this app loads the embedding and reranking models alongside it. There is no published fit guarantee for your machine, so treat sizing as a trial. A reasonable starting point on 16 GB RAM is a 3B model (qwen2.5:3b or llama3.2:3b). If answers are good but slow, try qwen2.5:7b and run `ollama ps` while it generates to see whether it sits on the GPU or spills to CPU. If the machine slows badly or the model process dies, go back to the smaller one. More GPU memory gives a 7B more room; on a 4 GB card, expect a 3B to be the practical choice. Only sub-2B models were tested during development (see below), and they misread real passages, so run the evaluator with your chosen model before trusting answers.

## Testing

```bash
# Unit + end-to-end tests (64 with Ollama running)
python -m pytest

# Component-by-component validation with a written report
python validate.py
```

The test suite generates its own sample PDF, so it never needs private notes. Tests that need a running Ollama server (grounded answer with citation, abstention on off-topic questions) are skipped automatically when no server is reachable. See TESTING.md for details.

Tested with: Python 3.10; Ollama 0.40.16 with qwen2.5:0.5b as the generation model the configured default is qwen3-vl:4b-instruct and follows the same code path; a clean virtual environment install of `requirements.txt` (latest releases: torch 2.14, sentence-transformers 6.3, chromadb 1.5) and an older set (torch 2.2, sentence-transformers 3.4, chromadb 1.3). 64 tests pass with an Ollama server running, 61 pass and 3 live tests skip without one.

Real-document checks used a private 52-page PDF and three grounded questions. In a roughly 2 GB RAM environment, qwen2.5:0.5b was tested end-to-end; retrieval ranked the relevant pages first, but generation misread some passages. llama3.2:1b ran alone but failed for lack of memory alongside the retrieval stack. When retrieval and generation were run in separate processes, its first run answered the quorum question incorrectly and falsely abstained on two other answerable questions. A later repeat of the quorum question returned the correct count, showing that a single successful run is not evidence of reliable generation. A separate generation-only qwen2.5:1.5b check also answered the wrong sub-question. These are diagnostic checks, not a benchmark or evidence that those models are adequate. The configured default qwen3-vl:4b-instruct was not validated here (it could not run in the 2 GB sandbox at all) and its answer quality remains untested.

## Project structure

```
RAG-Model/
├── src/
│   ├── main.py          # CLI: ingest / ask / chat
│   ├── ingest.py        # document ingestion pipeline
│   ├── loader.py        # PDF/TXT extraction, quality scoring, vision fallback
│   ├── chunker.py       # overlapping word-based chunking
│   ├── embeddings.py    # sentence-transformer embeddings
│   ├── vectorstore.py   # ChromaDB store
│   ├── retriever.py     # semantic retrieval
│   ├── reranker.py      # cross-encoder reranking
│   ├── generator.py     # grounded answer generation via Ollama
│   ├── vision.py        # optional vision-model page extraction
│   ├── ocr.py           # optional Tesseract helpers
│   └── chat.py          # interactive chat (shortcut for main.py chat)
├── tests/               # pytest suite (unit + end-to-end retrieval)
├── scripts/
│   ├── create_sample_pdf.py
│   ├── check_setup.py     # read-only Ollama setup checker
│   └── evaluate.py        # retrieval/generation evaluation harness
├── experiments/         # OCR/vision experiments (not part of the pipeline)
├── validate.py          # component validation with written report
├── requirements.txt
└── .env.example
```

## Limitations

Honest ones, so results are not over-trusted:

- Answer quality depends on the local LLM. The retrieval-confidence gate blocks questions retrieval judges unsupported (off-topic and, on the sample notes, near-topic gaps), but it is a heuristic: margins vary by corpus, and a weak model can still misread or over-extend a retrieved chunk that did pass the gate. Verify answers against the cited page.
- The generation model only sees the retrieved chunks. If retrieval misses the right page, the model may abstain or give an unsupported answer.
- Chunking is word-based and does not understand document structure (headings, columns, tables).
- Vision extraction quality depends on the local vision model and page rendering.
- Scanned/image-only PDFs need the vision model or OCR; plain text extraction returns nothing for them.

## Data and privacy

Notes are ingested and indexed entirely on your own machine; nothing is sent to a third-party API. The `data/` directory is gitignored, so your documents and the vector index are never committed. Do not commit private documents or copyrighted material you may not redistribute.

## Author

Aarav Pachouri
GitHub: [@aaravpachouri](https://github.com/aaravpachouri)

Contribution note: the original design and code are the author's. The November 2026 repair work (wiring fixes, CLI, test suite, abstention gate, docs rewrite) was done with an AI coding assistant and is attributed as such in the commit history.
