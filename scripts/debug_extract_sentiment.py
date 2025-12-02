#!/usr/bin/env python3
"""
Helper to call vnsentiment.cli.extract_sentiment directly (bypasses Typer) for debugging.

Usage:
    source .venv/bin/activate
    python scripts/debug_extract_sentiment.py --input data/sentiment/entities/batch_requests/2024-11/requests_combined_output.jsonl --min-confidence 0.6
"""

import argparse
from pathlib import Path

from vnsentiment.cli import extract_sentiment
from vnsentiment.config import ENTITY_SENTIMENT_OUTPUT_DIR

DEFAULT_INPUT = Path("data/sentiment/entities/batch_requests/2023-01/requests_combined_output.jsonl")

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug helper for extract_sentiment.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to requests_combined_output.jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Destination directory for sentiment files (default: {ENTITY_SENTIMENT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0, help="Minimum confidence threshold for keeping a match"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or ENTITY_SENTIMENT_OUTPUT_DIR
    extract_sentiment(batch_output_file=args.input, output_dir=output_dir, min_confidence=args.min_confidence)



if __name__ == "__main__":
    main()
