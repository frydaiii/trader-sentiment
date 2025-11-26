from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
SENTIMENT_DIR = DATA_DIR / "sentiment"
SYMBOLS_PATH = Path("symbols/hose_symbols.csv")
ICB_INDUSTRIES_PATH = Path("symbols/icb_industries.csv")
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_OUTPUT_PATH = SENTIMENT_DIR / "scores.jsonl"
ENTITY_OUTPUT_DIR = SENTIMENT_DIR / "entities"
ENTITY_MAP_OUTPUT_PATH = ENTITY_OUTPUT_DIR / "symbols.json"
ENTITY_BATCH_REQUESTS_DIR = ENTITY_OUTPUT_DIR / "batch_requests"
ENTITY_BATCH_OUTPUT_PATH = ENTITY_OUTPUT_DIR / "entity_batch_results.jsonl"
MAX_INPUT_TOKENS = 6000  # rough safeguard when trimming article text
