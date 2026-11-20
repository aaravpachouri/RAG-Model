"""Check your local Ollama setup for this project.

    python scripts/check_setup.py

Read-only: prints installed models, which can generate and
which can read page images, and the exact environment
settings to copy. Nothing is pulled, written or sent.
"""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from setup_check import main

if __name__ == "__main__":
    raise SystemExit(main())
