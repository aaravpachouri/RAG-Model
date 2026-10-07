# RAG System Automated Testing Setup

> Comprehensive automated testing and validation framework for the RAG system

## 🎯 Features

✅ **Unit Tests** - Test individual components (embeddings, vector store, retriever)
✅ **Integration Tests** - Validate end-to-end RAG pipeline
✅ **Performance Tests** - Benchmark embedding, search, and retrieval operations
✅ **Quality Tests** - Verify result relevance with predefined test cases
✅ **Automated Reports** - Generate detailed test reports and metrics
✅ **Master Orchestrator** - Run all tests with a single command

## 🚀 Quick Start

Run the complete automated test suite:

```bash
python test_all.py
```

For quick validation (skip performance tests):
```bash
python test_all.py --quick
```

## 📦 What's Included

### Test Suites
- **unit tests** - 21 tests across 4 test modules
- **integration tests** - 5 comprehensive pipeline tests
- **performance benchmarks** - 4 operation timing tests
- **quality validation** - 5 test cases with keyword verification

### Scripts
| Script | Purpose |
|--------|---------|
| `test_all.py` | Master orchestrator - runs everything |
| `validate.py` | Comprehensive system validation |
| `run_tests.py` | Unit and integration tests |
| `batch_runner.py` | Performance and quality tests |

### Documentation
- `TESTING.md` - Complete testing guide
- `tests/fixtures.py` - Test data and benchmarks

## 📊 Test Coverage

```
RAG Components
├── Embeddings (5 tests)
│   ├── Initialization
│   ├── Single text encoding
│   ├── Multiple texts encoding
│   ├── Dimension consistency
│   └── Non-empty validation
├── Vector Store (5 tests)
│   ├── Initialization
│   ├── Document addition
│   ├── Document search
│   ├── Metadata inclusion
│   └── Empty search handling
├── Retriever (6 tests)
│   ├── Initialization
│   ├── Document retrieval
│   ├── Document structure
│   ├── Top-k respecting
│   ├── Multiple queries
│   └── Edge cases
└── Full Integration (5 tests)
    ├── Complete pipeline
    ├── Retrieval quality
    ├── Multi-query handling
    ├── Consistency
    └── Content quality
```

## 📈 Performance Benchmarks

| Operation | Target Time | Status |
|-----------|------------|--------|
| Embedding per doc | 0.5s | ✓ |
| Vector search | 1.0s | ✓ |
| Full retrieval | 2.0s | ✓ |
| Generation | 10.0s | ✓ |

## 🔄 Typical Test Run

```
$ python test_all.py

RAG SYSTEM - MASTER TEST ORCHESTRATOR
======================================================================
Started at: 2026-09-07 10:30:45
Mode: FULL

1. Running Validation Workflow...
  ✓ Embedding Model Initialization
  ✓ Encoding Multiple Texts
  ✓ Vector Store Initialization
  ✓ Document Retrieval
  ✓ Full Pipeline Validation
  [5/5 checks passed]

2. Running Unit and Integration Tests...
  ✓ Embeddings Tests: 5/5 passed
  ✓ Vector Store Tests: 5/5 passed
  ✓ Retriever Tests: 6/6 passed
  ✓ Integration Tests: 5/5 passed
  [21/21 tests passed]

3. Running Batch Performance and Quality Tests...
  ✓ Performance: All benchmarks passed
  ✓ Quality: 5/5 test cases passed
  [Results saved]

EXECUTION SUMMARY
======================================================================
✓ System Validation Workflow: PASS
✓ Unit and Integration Tests: PASS
✓ Batch Performance and Quality Tests: PASS

Total Workflows: 3
Passed: 3
Failed: 0

Completed at: 2026-09-07 10:35:22
======================================================================
```

## 📁 Project Structure

```
RAG-Model/
├── test_all.py                 # Master orchestrator
├── validate.py                 # System validation (40+ checks)
├── run_tests.py                # Unit and integration tests
├── batch_runner.py             # Performance and quality tests
├── TESTING.md                  # Complete testing guide
├── tests/
│   ├── __init__.py
│   ├── fixtures.py             # Test data and benchmarks
│   ├── test_embeddings.py      # 5 embedding tests
│   ├── test_vectorstore.py     # 5 vector store tests
│   ├── test_retriever.py       # 6 retriever tests
│   ├── test_integration.py     # 5 integration tests
│   └── data/
│       ├── test_cases.json     # Test cases
│       ├── sample_documents.txt # Sample docs
│       └── benchmarks.json     # Performance targets
├── reports/
│   ├── validation_report.txt   # Validation results
│   ├── test_report.txt         # Unit test results
│   └── batch_results.json      # Performance metrics
└── [source files...]
```

## 🎮 Usage Examples

### Run all tests
```bash
python test_all.py
```

### Quick validation only
```bash
python validate.py
```

### Run specific test suite
```bash
python run_tests.py
```

### Performance and quality tests
```bash
python batch_runner.py
```

### Check specific reports
```bash
# View validation results
type validation_report.txt

# View test results
type test_report.txt

# View performance metrics
type batch_results.json
```

## ✨ Key Features

1. **Comprehensive** - 21+ unit tests + integration tests + performance benchmarks
2. **Automated** - Single command runs entire test suite
3. **Detailed Reports** - Human-readable and machine-parseable results
4. **Performance Tracking** - Benchmark operations and track performance over time
5. **Quality Validation** - Test case-driven validation with keyword matching
6. **Easy to Extend** - Add new tests or test cases easily

## 🔧 Configuration

Edit `tests/fixtures.py` to:
- Add new test cases
- Adjust performance benchmarks
- Modify sample documents

## 📋 Requirements

- Python 3.8+
- Dependencies from `requirements.txt`
- Ollama running (optional, for generator tests)

## 🐛 Troubleshooting

**Tests failing?**
1. Check `validation_report.txt` for details
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify Ollama is running if testing generation
4. Check file permissions on `data/` directory

**Performance tests slow?**
- Adjust benchmarks in `tests/fixtures.py`
- Check system resources
- Review `batch_results.json` for timing details

**Import errors?**
- Run from project root: `cd RAG-Model && python test_all.py`
- Verify `src/` and `tests/` directories exist

## 📞 Next Steps

1. ✅ **Run tests**: `python test_all.py`
2. 📊 **Review reports**: Check generated `.txt` and `.json` files
3. 🔧 **Customize**: Edit `tests/fixtures.py` for your needs
4. 🔄 **Schedule**: Set up automated runs with Task Scheduler

---

**Ready to automate testing?** Run `python test_all.py` now!

