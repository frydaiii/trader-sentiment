from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import orjson
import typer
from openai import OpenAI

from .analyzer import SentimentAnalyzer
from .config import DEFAULT_MODEL, DEFAULT_OUTPUT_PATH, DEFAULT_TEMPERATURE, SENTIMENT_DIR
from .models import ArticleInput


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Sentiment analysis utilities for VN news articles.")


def _load_article(path: Path) -> ArticleInput:
    payload = orjson.loads(path.read_bytes())
    article = ArticleInput.from_payload(payload)
    if "source" not in article.metadata:
        article.metadata["source"] = path.parts[-4] if len(path.parts) >= 4 else "unknown"
    article.metadata.setdefault("path", str(path))
    return article


def _append_result(result_path: Path, content: dict) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("ab") as handle:
        handle.write(orjson.dumps(content))
        handle.write(b"\n")


def _init_analyzer(model: str, temperature: float) -> SentimentAnalyzer:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set OPENAI_API_KEY in your environment before running this command.")
    client = OpenAI(api_key=api_key)
    return SentimentAnalyzer(model=model, temperature=temperature, client=client)


@app.command("score-file")
def score_file(
    article_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Destination JSONL file for scores."),
    model: str = typer.Option(DEFAULT_MODEL),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE),
) -> None:
    """
    Score a single article JSON file (as produced by the crawler).
    """
    article = _load_article(article_path)
    analyzer = _init_analyzer(model, temperature)
    result = analyzer.score_article(article)
    _append_result(output, result.to_json())
    typer.echo(f"Sentiment stored for {article_path} -> {output}")


@app.command("batch")
def batch(
    articles_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Destination JSONL file for scores."),
    model: str = typer.Option(DEFAULT_MODEL),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE),
    limit: Optional[int] = typer.Option(None, help="Process at most this many articles."),
    glob: str = typer.Option("*.json", help="Glob for article JSON files."),
) -> None:
    """
    Iterate through article JSON files in a directory tree and append sentiment scores to a JSONL file.
    """
    analyzer = _init_analyzer(model, temperature)
    files = sorted(articles_dir.rglob(glob))
    if not files:
        raise typer.BadParameter(f"No files matching {glob} under {articles_dir}")
    processed = 0
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        if limit and processed >= limit:
            break
        try:
            article = _load_article(path)
            result = analyzer.score_article(article)
            _append_result(output, result.to_json())
            processed += 1
        except Exception as exc:  # pragma: no cover - logging path
            logger.error("Failed to score %s: %s", path, exc, exc_info=True)
    typer.echo(f"Processed {processed} articles. Scores saved to {output}")
