# Testing

Two levels of automated checks, both runnable from the project root.

## 1. Test suite (pytest)

```bash
python -m pytest
```

76 tests covering:

| Area | What is checked |
|------|-----------------|
| Chunker | chunk size limits, overlap, empty input, metadata preservation |
| Loader | text-quality scoring, TXT loading, multi-page PDF extraction, unsupported types |
| Edge cases | image-only PDF pages, mixed PDF/TXT folders, empty folders, automatic pruning of deleted files' chunks, re-indexing of edited files, full index reset |
| Incremental ingest | unchanged files are skipped (no re-embedding), only an edited file is re-embedded, deleted files are pruned via the manifest, an interrupted run resumes from the per-file checkpoint, `--force` re-embeds everything, `--reset` clears the manifest with the index, a corrupt manifest falls back to full re-indexing |
| Abstention | confidence gate: off-topic and near-topic-but-unsupported questions are refused without calling the LLM; threshold configurable; empty results handled; retrieved-passages display fires only when the gate passes. Note: the gate is a heuristic - margins are measured on the sample notes, not guaranteed on every corpus |
| Generator | prompt contains context, sources, grounding rules; reply parsing and citation extraction (no Ollama needed) |
| Pipeline | end-to-end ingest of a generated sample PDF, semantic retrieval ranks the right page first, reranking, empty-store behaviour |
| Live generation | with a running Ollama server: grounded answers carry source citations, off-topic questions are refused. Skipped automatically when no server is reachable |

The suite generates its own sample PDF (short original physics notes) in a temporary directory and uses an in-memory vector store, so it never touches your documents in `data/`. First run downloads the embedding and reranking models (~200 MB) from Hugging Face and caches them.

Run the live generation tests with your real model:

```bash
ollama pull qwen3-vl:4b-instruct
ollama serve &
python -m pytest tests/test_generation_live.py
```

## 2. Component validation

```bash
python validate.py
```

Checks each component in turn (embeddings, vector store, retriever + reranker, loader, generator) with realistic data, prints PASS/FAIL/SKIP per check, and writes `validation_report.txt`.

The generation check requires a running Ollama server with the configured model pulled (`ollama pull qwen3-vl:4b-instruct`, or set `RAG_MODEL` to a model you have). Without it the check is reported as SKIP rather than FAIL, since the rest of the system does not depend on it.

## Requirements

- Test dependencies are included in `requirements.txt` (`pytest`).
- Run everything from the project root so imports resolve.
