# VNNews Collector

Python 3.12 module for collecting Vietnam news article URLs via sitemaps and crawling their content.

## Features

- Discovers sitemap URLs automatically from `robots.txt`.
- Recursively parses sitemap indexes, handles `.gz` sitemaps, and filters entries by date.
- Collects deduplicated URL lists per source plus a combined list under `data/url_lists/`.
- Crawls article content with [`news-please`](https://github.com/fhamborg/news-please), saving JSON payloads to `data/articles/{source}/{YYYY}/{YYYY-MM-DD}/`.
- Persists crawl checkpoints in `data/state/last_run.json` and supports `--resume`/`--reset-state` options.
- Threaded crawling with retry/backoff, logging, and robots.txt compliance.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI Usage

```bash
vnnews --help
```

### Collect URLs

```bash
vnnews collect-urls --source cafef --source cafebiz --since-years 3 --batch-days 10
```

Outputs:

- `data/url_lists/{source}-{YYYYMMDD}.txt`
- `data/url_lists/all-{YYYYMMDD}.txt`

Options:

- `--since-date YYYY-MM-DD` – explicit start date
- `--until-date YYYY-MM-DD` – cap collection to an upper bound (default: today or resume cursor)
- `--since-years N` – fallback look-back window (default 5 years)
- `--batch-days N` – fetch at most `N` days per run and store progress in `data/state/collect_state.json`
- `--max-sitemaps-per-source N` – stop after hitting N sitemap documents per source (useful for throttling)
- `--resume/--no-resume` – continue from checkpoints (default on; required for batching)
- `--reset-state` – clear `data/state/collect_state.json` (and force a fresh batch window)

### Crawl Articles

```bash
vnnews crawl data/url_lists/cafef-20240101.txt --max-workers 8 --max-urls 500
```

Options mirror `collect-urls`, plus `--max-workers` to control concurrency. The crawler stores the newest processed article date in `data/state/last_run.json`.

Batching-specific options:

- `--max-urls N` – crawl only the next `N` URLs from the list. Combined with `--resume`, progress is tracked per file in `data/state/crawl_progress.json` so subsequent runs continue where the previous batch stopped.
- `--resume/--no-resume` – also governs whether URL offsets and batch checkpoints are honored.
- `--reset-state` – clears both the crawl date checkpoint (`last_run.json`) and the URL offset file.

## Project Layout

- `src/vnnews/config.py` – shared constants and source definitions.
- `src/vnnews/robots.py` – robots.txt caching.
- `src/vnnews/sitemap_client.py` – sitemap discovery/parsing.
- `src/vnnews/collectors.py` – URL collection and persistence helpers.
- `src/vnnews/crawler.py` – threaded article downloader using news-please.
- `src/vnnews/state.py` – checkpoint utilities.
- `src/vnnews/cli.py` – Typer-powered command-line interface.

## Notes

- Make sure outbound requests respect the target sites’ rate limits.
- `news-please` pulls many dependencies; use a virtual environment.
- Review saved JSON outputs before downstream processing.
