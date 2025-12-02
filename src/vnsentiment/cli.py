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

from .config import (
    DEFAULT_MODEL,
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


def _month_key(date_key: str) -> str:
    """
    Coarse bucket for batching; keeps year-month when a full date is available.
    """
    if len(date_key) >= 7 and date_key[4] == "-" and date_key[5:7].isdigit():
        return date_key[:7]
    return date_key


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


@app.command("classify-entities")
def classify_entities(
    articles_dir: Optional[Path] = typer.Option(
        None, "--articles-dir", exists=True, file_okay=False, help="Directory containing article JSON files for non-batch runs."
    ),
    output_dir: Path = typer.Option(ENTITY_OUTPUT_DIR, help="Directory to write entity grouping JSON files."),
    symbols_csv: Path = typer.Option(SYMBOLS_PATH, exists=True, readable=True, help="Path to HOSE symbols CSV."),
    icb_csv: Path = typer.Option(ICB_INDUSTRIES_PATH, exists=True, readable=True, help="Path to ICB industries CSV."),
    model: str = typer.Option(DEFAULT_MODEL),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE),
    glob: str = typer.Option("*.json", help="Glob for article JSON files."),
    limit: Optional[int] = typer.Option(None, help="Process at most this many articles."),
    min_confidence: float = typer.Option(0.55, help="Minimum confidence to accept an entity match (0-1)."),
    batch_mode: Optional[Literal["create", "upload", "status"]] = typer.Option(
        None,
        help="Use the OpenAI batch API. 'create' builds JSONL request files; 'upload' sends them to OpenAI; 'status' retrieves batch status and downloads results when ready.",
    ),
    batch_input_folder: Optional[Path] = typer.Option(
        None,
        "--batch-input-folder",
        exists=True,
        file_okay=False,
        help=f"Folder of article JSON files for batch creation (default output for requests: {ENTITY_BATCH_REQUESTS_DIR}).",
    ),
    batch_input_file: Optional[Path] = typer.Option(
        None,
        "--batch-input-file",
        exists=True,
        dir_okay=False,
        help="Path to a requests.jsonl file (required for --batch-mode=upload/status).",
    ),
    batch_output: Optional[Path] = typer.Option(
        None, help=f"Destination for batch results when downloading (default: {ENTITY_BATCH_OUTPUT_PATH})."
    ),
) -> None:
    """
    Identify which HOSE-listed entities each article discusses and map symbols to article file names.
    """
    if batch_mode == "status":
        if not batch_input_file:
            raise typer.BadParameter("--batch-input-file is required when --batch-mode=status")
        progress_path = batch_input_file.parent / "progress.json"
        progress_base = batch_input_file.parent
        progress_files = [progress_path] if progress_path.exists() else []
        if not progress_files:
            raise typer.BadParameter(f"No progress.json found alongside {batch_input_file}")
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
                    "errors": [err.message for err in batch_job.errors.data] if batch_job.errors and batch_job.errors.data else [],
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            )
            _write_progress(progress_path, progress_payload)
            typer.echo(f"Updated progress at {progress_path}")
        return

    if batch_mode == "upload":
        if not batch_input_file:
            raise typer.BadParameter("--batch-input-file is required when --batch-mode=upload")
        path = batch_input_file
        base_dir = path.parent
        if not path.exists():
            raise typer.BadParameter(f"requests.jsonl not found at {path}")
        progress_path = base_dir / "progress.json"
        try:
            progress_payload = orjson.loads(progress_path.read_bytes())
        except Exception:
            progress_payload = {}
        if progress_payload.get("batch_id"):
            typer.echo(f"Skipping {path} because batch_id {progress_payload['batch_id']} already exists in progress.")
            return
        client = _get_openai_client()
        date_key = path.parent.name
        typer.echo(f"Handling publish month {date_key}")
        if "request_count" not in progress_payload:
            try:
                progress_payload["request_count"] = sum(1 for _ in path.open("rb"))
            except Exception:
                progress_payload["request_count"] = None
        upload = client.files.create(file=path, purpose="batch")
        batch_job = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"command": "classify-entities", "date": date_key},
        )
        typer.echo(
            f"Created batch {batch_job.id} for {date_key} (status={batch_job.status}) using {path} with {progress_payload.get('request_count')} requests"
        )
        progress_payload.update(
            {
                "batch_id": batch_job.id,
                "status": batch_job.status,
                "requests_file": str(path),
                "date_key": date_key,
                "model": model,
                "errors": [],
                "created_at": progress_payload.get("created_at") or datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        _write_progress(progress_path, progress_payload)
        typer.echo(f"Updated progress at {progress_path}")
        return

    source_dir: Optional[Path]
    if batch_mode == "create":
        source_dir = batch_input_folder
        if not source_dir:
            raise typer.BadParameter("--batch-input-folder is required when --batch-mode=create")
    else:
        source_dir = articles_dir

    if not source_dir:
        raise typer.BadParameter("--articles-dir is required when batch_mode is not 'status' or 'upload'")

    classifier = _init_entity_classifier(model, temperature)
    icb_lookup = load_icb_lookup(icb_csv)
    entities = load_hose_symbols(symbols_csv, icb_lookup)
    files = sorted(source_dir.rglob(glob))
    if not files:
        raise typer.BadParameter(f"No files matching {glob} under {source_dir}")

    if batch_mode == "create":
        base_dir = ENTITY_BATCH_REQUESTS_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        handles: Dict[str, any] = {}
        counts: Dict[str, int] = defaultdict(int)
        batch_files: Dict[str, Path] = {}
        last_announced_bucket: Optional[str] = None
        try:
            for date_key, request in _iter_entity_batch_requests(classifier, entities, files, limit, source_dir):
                month_key = _month_key(date_key)
                if month_key != last_announced_bucket:
                    typer.echo(f"Handling publish month {month_key}")
                    last_announced_bucket = month_key
                target = base_dir / month_key / "requests.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                if month_key not in handles:
                    handles[month_key] = target.open("ab")
                    batch_files[month_key] = target
                handle = handles[month_key]
                handle.write(orjson.dumps(request))
                handle.write(b"\n")
                counts[month_key] += 1
        finally:
            for handle in handles.values():
                handle.close()
        if not batch_files:
            raise typer.BadParameter("No requests generated for batch creation.")
        for date_key, path in batch_files.items():
            progress_payload = {
                "requests_file": str(path),
                "request_count": counts[date_key],
                "date_key": date_key,
                "model": model,
                "errors": [],
                "status": "pending-upload",
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            progress_path = path.parent / "progress.json"
            _write_progress(progress_path, progress_payload)
            typer.echo(f"Wrote {path} with {counts[date_key]} requests (progress at {progress_path})")
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
