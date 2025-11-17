from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import typer
import orjson
from underthesea import sent_tokenize

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
from .utils import ensure_parent, chunked


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = typer.Typer(help="Vietnam news sitemap collector and crawler.")


@app.callback()
def main(
    ctx: typer.Context,
    quiet: bool = typer.Option(False, "--quiet", help="Deprecated; info logs are disabled by default."),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Enable info-level logs."),
) -> None:
    if quiet and verbose:
        raise typer.BadParameter("--quiet and --verbose cannot be used together.")
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)


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

    combined_paths = write_url_lists(results, timestamp_slug)
    if not combined_paths:
        typer.echo("No combined URL lists were created.")
    else:
        for slug, path in sorted(combined_paths.items()):
            typer.echo(f"Combined URL list for {slug} saved to {path}")


@app.command("crawl")
def crawl(
    data_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a URL list file or directory (auto-detects data/url_list/<source>/<date>.txt).",
    ),
    since_date: Optional[str] = typer.Option(None, help="Crawl only articles on/after this YYYY-MM-DD date."),
    since_years: int = typer.Option(DEFAULT_SINCE_YEARS, help="Fallback look-back window when no date is provided."),
    max_urls: Optional[int] = typer.Option(None, help="Process at most this many URLs during this run."),
    source_name: Optional[str] = typer.Option(
        None,
        "--source",
        help="When pointing to a date directory, crawl only this source's URL list (e.g., cafef).",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        min=1,
        help="Process URLs in batches of this size, saving resume checkpoints between batches.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Continue from saved state when available."),
    reset_state: bool = typer.Option(False, help="Clear saved state before running."),
) -> None:
    """
    Download article content from a URL list and persist JSON payloads.
    """
    data_path = data_path.resolve()
    source_label = source_name.strip() if source_name else None
    selected_source_slug = source_label.lower() if source_label else None

    def _resolve_url_list_file(path: Path) -> Path:
        if path.is_file():
            if selected_source_slug:
                raise typer.BadParameter("--source can only be used when specifying a directory of URL lists")
            return path

        def _latest_txt_file(directory: Path) -> Path:
            txt_files = sorted([candidate for candidate in directory.glob("*.txt") if candidate.is_file()])
            if not txt_files:
                raise typer.BadParameter(f"Directory {directory} does not contain any .txt URL lists")
            return txt_files[-1]

        def _resolve_legacy_layout(search_dir: Path, legacy_dirs: Optional[List[Path]] = None) -> Path:
            target_dir = search_dir
            candidate = target_dir / "all.txt"
            if not candidate.exists():
                dated_dirs = sorted(
                    legacy_dirs
                    if legacy_dirs is not None
                    else [child for child in target_dir.iterdir() if child.is_dir() and child.name.isdigit()],
                    key=lambda child: child.name,
                )
                if not dated_dirs:
                    raise typer.BadParameter(
                        f"Directory {search_dir} does not contain per-source folders or dated subdirectories"
                    )
                target_dir = dated_dirs[-1]
                candidate = target_dir / "all.txt"
                typer.echo(f"Detected latest date directory {target_dir.name}; using {candidate}")
                if not candidate.exists():
                    raise typer.BadParameter(
                        f"Directory {target_dir} does not contain all.txt; specify a valid date folder"
                    )

            if selected_source_slug:
                source_candidate = target_dir / f"{selected_source_slug}.txt"
                if not source_candidate.exists():
                    raise typer.BadParameter(
                        f"Source '{source_label}' does not have a URL list under {target_dir}"
                    )
                return source_candidate
            return candidate

        search_dir = path
        for folder_name in ("url_list", "url_lists"):
            candidate_dir = search_dir / folder_name
            if candidate_dir.is_dir():
                search_dir = candidate_dir
                break

        bucket_dirs: List[Path] = []
        if selected_source_slug:
            if search_dir.name == selected_source_slug:
                bucket_dirs.append(search_dir)
            bucket_dirs.append(search_dir / selected_source_slug)
        else:
            if search_dir.name == "all":
                bucket_dirs.append(search_dir)
            bucket_dirs.append(search_dir / "all")

        for bucket_dir in bucket_dirs:
            if bucket_dir.is_dir():
                return _latest_txt_file(bucket_dir)

        legacy_dirs = [child for child in search_dir.iterdir() if child.is_dir() and child.name.isdigit()]
        if selected_source_slug and not legacy_dirs:
            raise typer.BadParameter(
                f"Source '{source_label}' does not have a directory under {search_dir}. "
                "Expected data/url_list/<source>/<YYYYMMDD>.txt files."
            )
        return _resolve_legacy_layout(search_dir, legacy_dirs=legacy_dirs or None)

    url_file = _resolve_url_list_file(data_path)
    crawler_sources = SOURCES
    if selected_source_slug:
        matching_sources = [src for src in SOURCES if src.name.lower() == selected_source_slug]
        if not matching_sources:
            available = ", ".join(src.name for src in SOURCES)
            raise typer.BadParameter(
                f"Unknown source '{source_label}'. Available options: {available}"
            )
        crawler_sources = matching_sources
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

    processed_dates: List[date] = []
    total_urls = len(urls)
    completed_urls = 0
    with ArticleCrawler(crawler_sources, output_dir=ARTICLES_DIR) as crawler:
        with typer.progressbar(length=total_urls, label="Crawling URLs") as progress_bar:
            def advance() -> None:
                nonlocal completed_urls
                completed_urls += 1
                progress_bar.update(1)
                if resume:
                    progress_mgr.save_offset(url_file, start_offset + completed_urls)

            if batch_size:
                for chunk in chunked(urls, batch_size):
                    chunk_dates = crawler.crawl(
                        chunk,
                        since=effective_since,
                        progress_callback=advance,
                    )
                    processed_dates.extend(chunk_dates)
            else:
                processed_dates = crawler.crawl(
                    urls,
                    since=effective_since,
                    progress_callback=advance,
                )

    if processed_dates:
        latest_date = max(processed_dates)
        state_mgr.save(latest_date)
        typer.echo(f"Processed {len(processed_dates)} articles. Last date saved: {latest_date}")
    else:
        typer.echo("No articles processed; state file unchanged.")

    if resume:
        progress_mgr.save_offset(url_file, start_offset + completed_urls)


@app.command("crawl-today")
def crawl_today(
    sources: List[str] = typer.Option([], "--source", help="Filter to one or more source names."),
    target_date: Optional[str] = typer.Option(
        None,
        "--date",
        help="Override the target day to collect and crawl (YYYY-MM-DD). Defaults to today.",
    ),
    max_sitemaps_per_source: Optional[int] = typer.Option(
        None,
        help="Limit number of sitemap documents fetched per source during this run.",
    ),
    max_workers: int = typer.Option(DEFAULT_CONCURRENCY, help="Maximum concurrent download workers."),
    max_urls: Optional[int] = typer.Option(
        None,
        min=1,
        help="Process at most this many URLs after collection (per run).",
    ),
) -> None:
    """
    Collect sitemap URLs for a single day (today by default) and immediately crawl them.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    URL_LIST_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

    selected_sources = _select_sources(sources)
    target = _parse_date_option(target_date) or date.today()
    timestamp_slug = target.strftime("%Y%m%d")

    results: List[CollectionResult] = []
    with SitemapClient() as sitemap_client:
        collector = ArticleURLCollector(sitemap_client)
        for source in selected_sources:
            result = collector.collect_for_source(
                source=source,
                since=target,
                until=target,
                max_sitemaps=max_sitemaps_per_source,
            )
            results.append(result)

    collected_urls = [url for result in results for url in result.urls]
    if not collected_urls:
        typer.echo(f"No URLs found for {target.isoformat()} across the selected sources.")
        return

    combined_paths = write_url_lists(results, timestamp_slug)
    target_slug = target.strftime("%Y%m%d")
    target_path = combined_paths.get(target_slug)
    if target_path:
        typer.echo(
            f"Collected {len(collected_urls)} URLs for {target.isoformat()}. "
            f"Combined list saved to {target_path}"
        )
    else:
        typer.echo(
            f"Collected {len(collected_urls)} URLs for {target.isoformat()}, but no combined file was generated."
        )

    deduped_urls = list(dict.fromkeys(collected_urls))
    if max_urls:
        deduped_urls = deduped_urls[:max_urls]
    if not deduped_urls:
        typer.echo("No URLs selected for crawling after applying limits.")
        return

    with ArticleCrawler(selected_sources, output_dir=ARTICLES_DIR, max_workers=max_workers) as crawler:
        with typer.progressbar(length=len(deduped_urls), label="Crawling URLs") as progress_bar:
            processed_dates = crawler.crawl(
                deduped_urls,
                since=target,
                max_workers=max_workers,
                progress_callback=lambda: progress_bar.update(1),
            )

    if processed_dates:
        typer.echo(
            f"Crawled {len(processed_dates)} articles for {target.isoformat()}. "
            f"Outputs stored under {ARTICLES_DIR}"
        )
    else:
        typer.echo("No articles processed.")


@app.command("split-sentences")
def split_sentences(
    article_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to a crawler-generated JSON file."),
    field: str = typer.Option("text", help="JSON field that contains the article body."),
    output: Optional[Path] = typer.Option(
        None,
        help="Optional path to save the resulting JSON list. Defaults to printing to stdout.",
    ),
) -> None:
    """Split an article body into individual sentences using underthesea."""

    payload = orjson.loads(article_path.read_bytes())
    content = payload.get(field)
    if not isinstance(content, str) or not content.strip():
        raise typer.BadParameter(f"Field '{field}' is missing or empty in {article_path}.")

    sentences = [sentence.strip() for sentence in sent_tokenize(content) if sentence.strip()]
    if not sentences:
        raise typer.BadParameter("No sentences were produced from the provided content.")

    rendered = orjson.dumps(sentences, option=orjson.OPT_INDENT_2).decode()
    if output:
        ensure_parent(output)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(f"Wrote {len(sentences)} sentences to {output}")
    else:
        typer.echo(rendered)
