# Automated Testing and Validation for RAG System

This directory contains automated testing and validation scripts for the RAG (Retrieval-Augmented Generation) system.

## 📋 Quick Start

Run all tests and validation:

```bash
# Full comprehensive test suite
python test_all.py

# Quick mode (skip batch tests)
python test_all.py --quick
```

## 📊 Available Scripts

### 1. **test_all.py** - Master Orchestrator
Runs all testing and validation workflows in sequence.

```bash
python test_all.py          # Full test suite
python test_all.py --quick  # Quick mode
```

**Includes:**
- System validation
- Unit and integration tests
- Performance benchmarks
- Quality tests

### 2. **validate.py** - System Validation Workflow
Comprehensive validation of all RAG components end-to-end.

```bash
python validate.py
```

**Validates:**
- Embedding model initialization and encoding
- Vector store operations (add, search)
- Retriever and reranker functionality
- Generator initialization
- Full RAG pipeline integration

**Output:** `validation_report.txt`

### 3. **run_tests.py** - Unit and Integration Tests
Runs modular unit tests and integration tests.

```bash
python run_tests.py
```

**Test Suites:**
- `tests/test_embeddings.py` - Embedding model tests
- `tests/test_vectorstore.py` - Vector store tests
- `tests/test_retriever.py` - Retriever tests
- `tests/test_integration.py` - End-to-end integration tests

**Output:** `test_report.txt`

### 4. **batch_runner.py** - Performance and Quality Tests
Runs performance benchmarks and quality tests with test cases.

```bash
python batch_runner.py
```

**Tests:**
- Embedding generation performance
- Vector search performance
- Full retrieval performance
- Quality validation with predefined test cases

**Output:** `batch_results.json`

## 📁 Project Structure

```
RAG-Model/
├── test_all.py                 # Master orchestrator
├── validate.py                 # System validation
├── run_tests.py                # Unit/integration tests
├── batch_runner.py             # Performance tests
├── tests/
│   ├── __init__.py
│   ├── fixtures.py             # Test data and fixtures
│   ├── test_embeddings.py      # Embedding tests
│   ├── test_vectorstore.py     # Vector store tests
│   ├── test_retriever.py       # Retriever tests
│   ├── test_integration.py     # Integration tests
│   └── data/                   # Test data
│       ├── test_cases.json
│       ├── sample_documents.txt
│       └── benchmarks.json
└── [other source files...]
```

## 🧪 Test Categories

### Unit Tests
- Individual component validation
- Input/output verification
- Error handling

### Integration Tests
- Multi-component workflows
- End-to-end pipeline validation
- Data flow verification

### Performance Tests
- Embedding generation speed
- Vector search latency
- Retrieval time
- Generation time

### Quality Tests
- Relevance of retrieved documents
- Keyword matching in results
- Test case validation

## 📈 Performance Benchmarks

Default performance thresholds:

| Operation | Max Time |
|-----------|----------|
| Embedding per document | 0.5s |
| Vector search | 1.0s |
| Full retrieval | 2.0s |
| Text generation | 10.0s |

Edit `tests/fixtures.py` to adjust benchmarks.

## 📝 Test Data

Test data is defined in `tests/fixtures.py`:

- **TEST_CASES**: Predefined question-answer test cases
- **SAMPLE_DOCUMENTS**: Sample documents for testing
- **PERFORMANCE_BENCHMARKS**: Performance thresholds

Export test data to files:
```bash
python -c "from tests.fixtures import save_test_data; save_test_data()"
```

## 📊 Reports and Results

Generated files:

- `validation_report.txt` - Validation results
- `test_report.txt` - Unit/integration test results
- `batch_results.json` - Performance and quality test results (JSON format)

## 🔄 Continuous Integration

To run tests automatically on a schedule using Windows Task Scheduler:

```powershell
# Create a task to run tests daily
$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$action = New-ScheduledTaskAction -Execute "python" -Argument "C:\path\to\test_all.py" -WorkingDirectory "C:\path\to\RAG-Model"
Register-ScheduledTask -TaskName "RAG-SystemTests" -Trigger $trigger -Action $action
```

Or use a batch file:
```batch
@echo off
cd C:\path\to\RAG-Model
python test_all.py
pause
```

## 📋 Test Execution Flow

```
test_all.py (Master Orchestrator)
├── validate.py
│   ├── Embedding Model Validation
│   ├── Vector Store Validation
│   ├── Retriever Validation
│   ├── Generator Validation
│   └── Full Pipeline Validation
├── run_tests.py
│   ├── Embeddings Tests
│   ├── Vector Store Tests
│   ├── Retriever Tests
│   └── Integration Tests
└── batch_runner.py
    ├── Performance Tests
    └── Quality Tests
```

## 🚀 Usage Examples

### Run quick validation only
```bash
python validate.py
```

### Run specific test suite
```bash
python -m pytest tests/test_embeddings.py -v
# or
python -c "from tests.test_embeddings import run_tests; run_tests()"
```

### Run in quick mode (skip performance tests)
```bash
python test_all.py --quick
```

### Run and check results
```bash
python test_all.py
# Check generated reports
type validation_report.txt
type test_report.txt
type batch_results.json
```

## 🔧 Customization

### Add new test cases
Edit `tests/fixtures.py`:
```python
TEST_CASES.append({
    "question": "Your question here",
    "expected_keywords": ["keyword1", "keyword2"],
    "category": "category_name"
})
```

### Adjust performance benchmarks
Edit `tests/fixtures.py`:
```python
PERFORMANCE_BENCHMARKS["embedding_generation"]["max_time_per_doc"] = 0.3  # seconds
```

### Add new test suite
Create `tests/test_myfeature.py` with a `run_tests()` function, then add it to `run_tests.py`.

## 📌 Troubleshooting

### Tests fail due to missing dependencies
```bash
pip install -r requirements.txt
```

### Generator tests fail
The generator uses Ollama. Make sure Ollama is running:
```bash
# On Windows, start Ollama
ollama serve
```

### Vector store not initialized
Ensure the `data/chroma/` directory exists and has proper permissions.

### Path errors
Run scripts from the project root directory:
```bash
cd C:\path\to\RAG-Model
python test_all.py
```

## 📞 Support

For issues or questions about the test suite, check:
1. The validation report for detailed error messages
2. Test-specific output files
3. Individual test scripts for documentation

---

**Last Updated:** 2026-09-07
