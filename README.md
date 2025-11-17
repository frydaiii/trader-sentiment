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
vnnews-sentiment --help
```

### Split Sentences

Use the built-in helper to convert a crawled article JSON file into a list of Vietnamese sentences via [`underthesea`](https://github.com/undertheseanlp/underthesea):

```bash
vnnews split-sentences data/articles/cafef/2025/2025-11-09/example.json --output sentences.json
```

- `--field text` – choose a different JSON field if the article body is stored elsewhere.
- `--output PATH` – persist the JSON list to a file instead of printing to stdout.

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
- `--batch-size N` – process URLs in chunks of `N`, saving resume offsets after each batch so long runs can be resumed mid-file.
- `--resume/--no-resume` – also governs whether URL offsets and batch checkpoints are honored.
- `--reset-state` – clears both the crawl date checkpoint (`last_run.json`) and the URL offset file.

### Crawl Today's Articles

Need a single command that grabs today's sitemap entries and crawls them immediately? Use:

```bash
vnnews crawl-today --source cafef --max-workers 8
```

- `--source NAME` – specify one or more sources (defaults to all defined in `vnnews.config.SOURCES`).
- `--date YYYY-MM-DD` – override the target day (defaults to the current date).
- `--max-sitemaps-per-source N` – throttle sitemap discovery per source.
- `--max-workers N` – control crawler concurrency.
- `--max-urls N` – stop after crawling the first N URLs discovered for the day.

This command stores the collected URL lists under `data/url_lists/` (one file per source plus `all-YYYYMMDD.txt`) and writes crawled articles to `data/articles/...`.

### Sentiment Scoring

Set `OPENAI_API_KEY` in your environment, then run:

```bash
export OPENAI_API_KEY=sk-...
vnnews-sentiment score-file data/articles/cafef/2025/2025-11-09/example.json --model gpt-4o-mini
```

For batch processing of all crawler outputs in a directory (default JSONL destination is `data/sentiment/scores.jsonl`):

```bash
vnnews-sentiment batch data/articles/cafef --limit 200 --glob "*.json" --model gpt-4o-mini
```

Options:

- `--model` / `--temperature` – configure OpenAI Responses API parameters (defaults in `vnsentiment.config`).
- `--output PATH` – override the JSONL file.
- `--limit N` – stop after N files when batching.
- `--glob PATTERN` – restrict which article JSON files are scored.

## Project Layout

- `src/vnnews/config.py` – shared constants and source definitions.
- `src/vnnews/robots.py` – robots.txt caching.
- `src/vnnews/sitemap_client.py` – sitemap discovery/parsing.
- `src/vnnews/collectors.py` – URL collection and persistence helpers.
- `src/vnnews/crawler.py` – threaded article downloader using news-please.
- `src/vnnews/state.py` – checkpoint utilities.
- `src/vnnews/cli.py` – Typer-powered command-line interface.
- `src/vnsentiment/*.py` – OpenAI-powered sentiment analysis helper module and CLI (`vnnews-sentiment`).

## Notes

- Make sure outbound requests respect the target sites’ rate limits.
- `news-please` pulls many dependencies; use a virtual environment.
- Review saved JSON outputs before downstream processing.
- For sentiment scoring, set `OPENAI_API_KEY` before running `vnnews-sentiment score-file ...` or `vnnews-sentiment batch ...`.
