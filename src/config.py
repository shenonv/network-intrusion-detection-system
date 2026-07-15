"""Central configuration: paths and shared constants.

Every script imports paths from here, so the project can be run
from any working directory.
"""
import sys
from pathlib import Path

# The project path contains non-ASCII characters (OneDrive localized folder),
# which crashes print() on Windows' default cp1252 console encoding.
for _stream in (sys.stdout, sys.stderr):
    if _stream and _stream.encoding.lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

for _dir in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
