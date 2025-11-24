from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import orjson
import typer
from openai import OpenAI

from .analyzer import SentimentAnalyzer
from .config import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TEMPERATURE,
    ENTITY_OUTPUT_DIR,
    ICB_INDUSTRIES_PATH,
    SENTIMENT_DIR,
    SYMBOLS_PATH,
)
from .entity_classifier import EntityClassifier, load_hose_symbols, load_icb_lookup
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


def _init_entity_classifier(model: str, temperature: float) -> EntityClassifier:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set OPENAI_API_KEY in your environment before running this command.")
    client = OpenAI(api_key=api_key)
    return EntityClassifier(model=model, temperature=temperature, client=client)


def _icb_rollup_codes(icb_code: str) -> Dict[str, str]:
    """
    Return hierarchy codes for subsector -> sector -> subindustry -> industry.
    """
    code = icb_code.strip()
    if len(code) != 4 or not code.isdigit():
        return {}
    return {
        "subsector": code,
        "sector": code[:3] + "0",
        "subindustry": code[:2] + "00",
        "industry": code[:1] + "000",
    }


def _publish_date(article: ArticleInput, path: Path) -> str:
    """
    Resolve publish date for grouping. Prefers metadata date_publish; falls back to folder name.
    """
    meta_date = article.metadata.get("date_publish") or article.metadata.get("date")
    if isinstance(meta_date, str) and len(meta_date) >= 10:
        return meta_date[:10]
    if len(path.parts) >= 2:
        candidate = path.parts[-2]
        if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
            return candidate
    return "unknown-date"


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


@app.command("classify-entities")
def classify_entities(
    articles_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(ENTITY_OUTPUT_DIR, help="Directory to write entity grouping JSON files."),
    symbols_csv: Path = typer.Option(SYMBOLS_PATH, exists=True, readable=True, help="Path to HOSE symbols CSV."),
    icb_csv: Path = typer.Option(ICB_INDUSTRIES_PATH, exists=True, readable=True, help="Path to ICB industries CSV."),
    model: str = typer.Option(DEFAULT_MODEL),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE),
    glob: str = typer.Option("*.json", help="Glob for article JSON files."),
    limit: Optional[int] = typer.Option(None, help="Process at most this many articles."),
    min_confidence: float = typer.Option(0.55, help="Minimum confidence to accept an entity match (0-1)."),
) -> None:
    """
    Identify which HOSE-listed entities each article discusses and map symbols to article file names.
    """
    classifier = _init_entity_classifier(model, temperature)
    icb_lookup = load_icb_lookup(icb_csv)
    entities = load_hose_symbols(symbols_csv, icb_lookup)
    files = sorted(articles_dir.rglob(glob))
    if not files:
        raise typer.BadParameter(f"No files matching {glob} under {articles_dir}")
    processed = 0
    symbol_matches: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    icb_matches: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
        lambda: {
            "subsector": defaultdict(list),
            "sector": defaultdict(list),
            "subindustry": defaultdict(list),
            "industry": defaultdict(list),
        }
    )
    macro_articles: Dict[str, List[str]] = defaultdict(list)
    base_dir = output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    entity_by_symbol = {entity.symbol: entity for entity in entities}

    for path in files:
        if limit and processed >= limit:
            break
        try:
            article = _load_article(path)
            classification = classifier.classify_article(article, entities)
            date_key = _publish_date(article, path)
            if classification.macro:
                macro_articles[date_key].append(str(path))
            for match in classification.matches:
                if match.confidence < min_confidence:
                    continue
                symbol_matches[date_key][match.symbol].append(str(path))
                entity = entity_by_symbol.get(match.symbol)
                if entity and entity.icb_code:
                    for level, code in _icb_rollup_codes(entity.icb_code).items():
                        icb_matches[date_key][level][code].append(str(path))
            processed += 1
        except Exception as exc:  # pragma: no cover - logging path
            logger.error("Failed to classify %s: %s", path, exc, exc_info=True)

    def _dedupe(mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {key: sorted(set(paths)) for key, paths in mapping.items()}

    for date_key in sorted(symbol_matches.keys() | macro_articles.keys() | icb_matches.keys()):
        date_dir = base_dir / date_key
        date_dir.mkdir(parents=True, exist_ok=True)

        icb_named = {}
        for level, mapping in icb_matches[date_key].items():
            level_map = {}
            for code, paths in mapping.items():
                icb_entry = icb_lookup.get(code)
                level_map[code] = {"name": icb_entry.name if icb_entry else "", "articles": sorted(set(paths))}
            icb_named[level] = level_map

        (date_dir / "symbols.json").write_bytes(orjson.dumps(_dedupe(symbol_matches[date_key]), option=orjson.OPT_INDENT_2))
        (date_dir / "subsectors.json").write_bytes(orjson.dumps(icb_named.get("subsector", {}), option=orjson.OPT_INDENT_2))
        (date_dir / "sectors.json").write_bytes(orjson.dumps(icb_named.get("sector", {}), option=orjson.OPT_INDENT_2))
        (date_dir / "subindustries.json").write_bytes(
            orjson.dumps(icb_named.get("subindustry", {}), option=orjson.OPT_INDENT_2)
        )
        (date_dir / "industries.json").write_bytes(
            orjson.dumps(icb_named.get("industry", {}), option=orjson.OPT_INDENT_2)
        )
        (date_dir / "macro.json").write_bytes(orjson.dumps(sorted(set(macro_articles[date_key])), option=orjson.OPT_INDENT_2))

    typer.echo(f"Processed {processed} articles. Saved grouped outputs under {base_dir}")
