from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import orjson
import typer
from openai import OpenAI

from .analyzer import SentimentAnalyzer
from .config import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TEMPERATURE,
    ENTITY_BATCH_REQUESTS_DIR,
    ENTITY_BATCH_OUTPUT_PATH,
    ENTITY_OUTPUT_DIR,
    ICB_INDUSTRIES_PATH,
    SENTIMENT_DIR,
    SYMBOLS_PATH,
)
from .entity_classifier import EntityClassifier, EntityDefinition, load_hose_symbols, load_icb_lookup
from .models import ArticleInput


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Sentiment analysis utilities for VN news articles.")


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set OPENAI_API_KEY in your environment before running this command.")
    return OpenAI(api_key=api_key)


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
    client = _get_openai_client()
    return SentimentAnalyzer(model=model, temperature=temperature, client=client)


def _init_entity_classifier(model: str, temperature: float) -> EntityClassifier:
    client = _get_openai_client()
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


def _iter_entity_batch_requests(
    classifier: EntityClassifier, entities: List[EntityDefinition], files: List[Path], limit: Optional[int], base_dir: Path
):
    """
    Stream chat.completions requests for OpenAI batch processing, respecting limit.
    """
    count = 0
    for path in files:
        if limit and count >= limit:
            break
        article = _load_article(path)
        body = classifier.build_request_body(article, entities)
        date_key = _publish_date(article, path)
        try:
            custom_id = str(path.relative_to(base_dir))
        except ValueError:
            custom_id = str(path)
        yield date_key, {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
        count += 1


def _download_file_content(client: OpenAI, file_id: str) -> bytes:
    """
    Download a file from OpenAI, normalizing to bytes for writing.
    """
    content = client.files.content(file_id)
    if hasattr(content, "read"):
        data = content.read()
    elif hasattr(content, "text"):
        data = content.text  # type: ignore[assignment]
    elif hasattr(content, "content"):
        data = content.content  # type: ignore[assignment]
    else:
        data = content
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, bytes):
        return data
    return bytes(data)


def _write_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


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
    batch_mode: Optional[Literal["create", "status"]] = typer.Option(
        None,
        help="Use the OpenAI batch API. 'create' builds/uploads a JSONL request file; 'status' retrieves batch status and downloads results when ready.",
    ),
    batch_input: Optional[Path] = typer.Option(
        None, help=f"Directory to store per-date JSONL request files for batch creation (default: {ENTITY_BATCH_REQUESTS_DIR})."
    ),
    batch_output: Optional[Path] = typer.Option(
        None, help=f"Destination for batch results when downloading (default: {ENTITY_BATCH_OUTPUT_PATH})."
    ),
) -> None:
    """
    Identify which HOSE-listed entities each article discusses and map symbols to article file names.
    """
    if batch_mode == "status":
        progress_base = batch_input or ENTITY_BATCH_REQUESTS_DIR
        progress_files = sorted(progress_base.rglob("progress.json"))
        if not progress_files:
            raise typer.BadParameter(f"No progress.json files found under {progress_base}")
        client = _get_openai_client()
        for progress_path in progress_files:
            try:
                progress_payload = orjson.loads(progress_path.read_bytes())
            except Exception:
                progress_payload = {}
            batch_id = progress_payload.get("batch_id")
            if not batch_id:
                typer.echo(f"Skipping {progress_path} (no batch_id).")
                continue
            batch_job = client.batches.retrieve(batch_id)
            meta_date = (
                progress_payload.get("date_key")
                or progress_payload.get("date")
                or (getattr(batch_job, "metadata", {}) or {}).get("date")
                or "unknown-date"
            )
            typer.echo(f"Handling publish date {meta_date}")
            typer.echo(f"Batch {batch_job.id} status: {batch_job.status}")
            output_file_id = getattr(batch_job, "output_file_id", None)
            error_file_id = getattr(batch_job, "error_file_id", None)
            if error_file_id:
                typer.echo(f"Error file id: {error_file_id}")
            target_path = batch_output or (progress_path.parent / "batch_results.jsonl")
            if batch_job.status == "completed" and output_file_id:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(_download_file_content(client, output_file_id))
                typer.echo(f"Downloaded batch output to {target_path}")
            elif output_file_id:
                typer.echo(f"Output file id: {output_file_id} (not downloaded until status=completed)")
            progress_payload.update(
                {
                    "batch_id": batch_job.id,
                    "status": batch_job.status,
                    "output_file_id": output_file_id,
                    "error_file_id": error_file_id,
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            )
            _write_progress(progress_path, progress_payload)
            typer.echo(f"Updated progress at {progress_path}")
        return

    classifier = _init_entity_classifier(model, temperature)
    icb_lookup = load_icb_lookup(icb_csv)
    entities = load_hose_symbols(symbols_csv, icb_lookup)
    files = sorted(articles_dir.rglob(glob))
    if not files:
        raise typer.BadParameter(f"No files matching {glob} under {articles_dir}")

    if batch_mode == "create":
        base_dir = batch_input or ENTITY_BATCH_REQUESTS_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        handles: Dict[str, any] = {}
        counts: Dict[str, int] = defaultdict(int)
        batch_files: Dict[str, Path] = {}
        last_announced_date: Optional[str] = None
        try:
            for date_key, request in _iter_entity_batch_requests(classifier, entities, files, limit, articles_dir):
                if date_key != last_announced_date:
                    typer.echo(f"Handling publish date {date_key}")
                    last_announced_date = date_key
                target = base_dir / date_key / "requests.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                if date_key not in handles:
                    handles[date_key] = target.open("ab")
                    batch_files[date_key] = target
                handle = handles[date_key]
                handle.write(orjson.dumps(request))
                handle.write(b"\n")
                counts[date_key] += 1
        finally:
            for handle in handles.values():
                handle.close()
        if not batch_files:
            raise typer.BadParameter("No requests generated for batch creation.")
        client = _get_openai_client()
        for date_key, path in batch_files.items():
            upload = client.files.create(file=path, purpose="batch")
            batch_job = client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"command": "classify-entities", "date": date_key},
            )
            typer.echo(
                f"Created batch {batch_job.id} for {date_key} (status={batch_job.status}) using {path} with {counts[date_key]} requests"
            )
            progress_payload = {
                "batch_id": batch_job.id,
                "status": batch_job.status,
                "requests_file": str(path),
                "request_count": counts[date_key],
                "date_key": date_key,
                "model": model,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            progress_path = path.parent / "progress.json"
            _write_progress(progress_path, progress_payload)
            typer.echo(f"Saved progress to {progress_path}")
        return

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
