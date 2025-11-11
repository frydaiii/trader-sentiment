from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import typer

from .collectors import ArticleURLCollector, CollectionResult, write_url_lists
from .config import (
    ARTICLES_DIR,
    DATA_DIR,
    DEFAULT_CONCURRENCY,
    DEFAULT_SINCE_YEARS,
    default_since_timedelta,
    SourceConfig,
    SOURCES,
    URL_LIST_DIR,
)
from .crawler import ArticleCrawler
from .sitemap_client import SitemapClient
from .state import (
    CollectionStateManager,
    CrawlProgressManager,
    StateManager,
    determine_since_date,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = typer.Typer(help="Vietnam news sitemap collector and crawler.")


def _select_sources(source_names: List[str]) -> List[SourceConfig]:
    if not source_names:
        return SOURCES
    wanted = {name.lower() for name in source_names}
    selected = [source for source in SOURCES if source.name.lower() in wanted]
    if not selected:
        raise typer.BadParameter(f"No matching sources found for {source_names}")
    return selected


def _parse_date_option(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - CLI parsing
        raise typer.BadParameter("Dates must be in YYYY-MM-DD format.") from exc


def _compute_collect_window(
    source: SourceConfig,
    base_since: date,
    until_override: Optional[date],
    batch_days: Optional[int],
    resume: bool,
    collection_state: CollectionStateManager,
) -> tuple[date, date]:
    until = until_override
    if until is None and resume and batch_days:
        stored = collection_state.get_until(source.name)
        if stored:
            until = stored
    if until is None:
        until = date.today()
    if batch_days:
        since = max(base_since, until - timedelta(days=batch_days - 1))
    else:
        since = base_since
    return since, until


@app.command("collect-urls")
def collect_urls(
    since_date: Optional[str] = typer.Option(None, help="Collect URLs updated on/after this YYYY-MM-DD date."),
    until_date: Optional[str] = typer.Option(None, help="Upper bound (inclusive) for collection window."),
    since_years: int = typer.Option(DEFAULT_SINCE_YEARS, help="Fallback look-back window when no date is provided."),
    sources: List[str] = typer.Option([], "--source", help="Filter to one or more source names."),
    batch_days: Optional[int] = typer.Option(
        None,
        help="Process at most this many days per run (per source). Enables resume checkpoints.",
    ),
    max_sitemaps_per_source: Optional[int] = typer.Option(
        None,
        help="Limit number of sitemap documents fetched per source during this run.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Continue from saved batch state when available."),
    reset_state: bool = typer.Option(False, help="Clear saved batch state before running."),
) -> None:
    """
    Fetch sitemap data and persist deduplicated URL lists with optional batching.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    URL_LIST_DIR.mkdir(parents=True, exist_ok=True)
    parsed_since = _parse_date_option(since_date)
    until_override = _parse_date_option(until_date)
    base_since = parsed_since or (date.today() - default_since_timedelta(since_years))
    selected_sources = _select_sources(sources)
    collection_state = CollectionStateManager()
    if reset_state:
        collection_state.reset([src.name for src in selected_sources] if selected_sources else None)
    timestamp_slug = date.today().strftime("%Y%m%d")
    results: List[CollectionResult] = []

    with SitemapClient() as sitemap_client:
        collector = ArticleURLCollector(sitemap_client)
        for source in selected_sources:
            since_window, until_window = _compute_collect_window(
                source=source,
                base_since=base_since,
                until_override=until_override,
                batch_days=batch_days,
                resume=resume,
                collection_state=collection_state,
            )
            if until_window and since_window > until_window:
                typer.echo(f"Skipping {source.name}: since {since_window} > until {until_window}")
                continue
            result = collector.collect_for_source(
                source=source,
                since=since_window,
                until=until_window,
                max_sitemaps=max_sitemaps_per_source,
            )
            results.append(result)
            if batch_days and resume:
                boundary_date = result.earliest_date or since_window
                next_cutoff = boundary_date - timedelta(days=1)
                collection_state.save_until(source.name, next_cutoff)

    combined_path = write_url_lists(results, timestamp_slug)
    typer.echo(f"Combined URL list saved to {combined_path}")


@app.command("crawl")
def crawl(
    url_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to a URL list file."),
    since_date: Optional[str] = typer.Option(None, help="Crawl only articles on/after this YYYY-MM-DD date."),
    since_years: int = typer.Option(DEFAULT_SINCE_YEARS, help="Fallback look-back window when no date is provided."),
    max_workers: int = typer.Option(DEFAULT_CONCURRENCY, help="Maximum concurrent download workers."),
    max_urls: Optional[int] = typer.Option(None, help="Process at most this many URLs during this run."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Continue from saved state when available."),
    reset_state: bool = typer.Option(False, help="Clear saved state before running."),
) -> None:
    """
    Download article content from a URL list and persist JSON payloads.
    """
    url_file = url_file.resolve()
    state_mgr = StateManager()
    progress_mgr = CrawlProgressManager()
    if reset_state:
        state_mgr.reset()
        progress_mgr.reset()
    parsed_since = _parse_date_option(since_date)
    effective_since = determine_since_date(
        since_date=parsed_since,
        since_years=since_years,
        resume=resume,
        state_manager=state_mgr if resume else None,
    )

    urls = [
        line.strip()
        for line in url_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        raise typer.BadParameter("URL file does not contain any entries.")
    start_offset = progress_mgr.get_offset(url_file) if resume else 0
    if start_offset >= len(urls):
        typer.echo("All URLs in this list have already been processed. Use --reset-state to start over.")
        return
    if start_offset:
        urls = urls[start_offset:]
    if max_urls:
        urls = urls[:max_urls]
    if not urls:
        typer.echo("No URLs selected for this batch after applying limits.")
        return

    with ArticleCrawler(SOURCES, output_dir=ARTICLES_DIR, max_workers=max_workers) as crawler:
        processed_dates = crawler.crawl(urls, since=effective_since, max_workers=max_workers)

    if processed_dates:
        latest_date = max(processed_dates)
        state_mgr.save(latest_date)
        typer.echo(f"Processed {len(processed_dates)} articles. Last date saved: {latest_date}")
    else:
        typer.echo("No articles processed; state file unchanged.")

    if resume:
        progress_mgr.save_offset(url_file, start_offset + len(urls))
