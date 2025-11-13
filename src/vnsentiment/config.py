from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
SENTIMENT_DIR = DATA_DIR / "sentiment"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_OUTPUT_PATH = SENTIMENT_DIR / "scores.jsonl"
MAX_INPUT_TOKENS = 6000  # rough safeguard when trimming article text
