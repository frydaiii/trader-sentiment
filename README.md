# VNNews Collector

Python 3.12 module for collecting Vietnam news article URLs via sitemaps and crawling their content.

## Features

- Discovers sitemap URLs automatically from `robots.txt`.
- Recursively parses sitemap indexes, handles `.gz` sitemaps, and filters entries by date.
- Collects deduplicated URL lists per source plus combined lists under `data/url_list/{source}/{YYYYMMDD}.txt` (including `all`).
- Crawls article content with [`news-please`](https://github.com/fhamborg/news-please), saving JSON payloads to `data/articles/{source}/{YYYY}/{YYYY-MM-DD}/`.
- Crawl commands display interactive progress bars so long runs are easy to monitor.
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
vnnews --quiet collect-urls --source cafef
```

Use `--quiet` before any command to suppress info-level log output.

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

- `data/url_list/{source}/{YYYYMMDD}.txt` for each source
- `data/url_list/all/{YYYYMMDD}.txt` with the combined deduplicated URLs

Each `YYYYMMDD` folder corresponds to an article publish date detected from sitemap `lastmod` values. Entries without a publish date fall back to the collection run's date.

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
vnnews crawl data/url_list/cafef/20240101.txt --max-urls 500
vnnews crawl data/url_list --max-urls 500               # latest combined list under all/
vnnews crawl data/url_list --source cafef --max-urls 200 # newest cafef list auto-selected
```

Options mirror `collect-urls`. The crawler runs sequentially, so the progress bar advances linearly while storing the newest processed article date in `data/state/last_run.json`. Pointing at `data/url_list/` auto-detects the newest dated file under `all/` (or the requested `--source` directory), and you can still pass an explicit `.txt` file path to override the selection.

All crawl invocations show a progress bar that advances as each URL finishes downloading.

Batching-specific options:

- `--max-urls N` – crawl only the next `N` URLs from the list. Combined with `--resume`, progress is tracked per file in `data/state/crawl_progress.json` so subsequent runs continue where the previous batch stopped.
- `--batch-size N` – process URLs in chunks of `N`, saving resume offsets after each batch so long runs can be resumed mid-file.
- `--resume/--no-resume` – also governs whether URL offsets and batch checkpoints are honored; offsets are flushed after every processed URL so you can interrupt safely and continue later.
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

This command stores the collected URL lists under `data/url_list/{source}/{YYYYMMDD}.txt` (including an `all/` bucket) and writes crawled articles to `data/articles/...`.

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

### Entity Classification

Tag crawled articles to HOSE-listed entities using the symbol and ICB reference CSVs under `symbols/`:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=sk-...
vnnews-sentiment classify-entities data/articles/vneconomy/2024 --output-dir data/sentiment/entities --min-confidence 0.6
```

Outputs six JSON files grouped by publish date under `data/sentiment/entities/<YYYY-MM-DD>/`:

- `symbols.json` – `TICKER -> [article_paths]`
- `subsectors.json`, `sectors.json`, `subindustries.json`, `industries.json` – each maps ICB code → `{ "name": ..., "articles": [...] }`
- `macro.json` – list of article paths deemed macro/market-wide (set by the model)

Defaults:

- `--symbols-csv symbols/hose_symbols.csv`
- `--icb-csv symbols/icb_industries.csv`
- `--glob "*.json"` with an optional `--limit N` to cap processed articles.
Publish-date subfolders are created automatically from article metadata (falls back to path or `unknown-date`).

## Project Layout

- `src/vnnews/config.py` – shared constants and source definitions.
- `src/vnnews/robots.py` – robots.txt caching.
- `src/vnnews/sitemap_client.py` – sitemap discovery/parsing.
- `src/vnnews/collectors.py` – URL collection and persistence helpers.
- `src/vnnews/crawler.py` – threaded article downloader using news-please.
- `src/vnnews/state.py` – checkpoint utilities.
- `src/vnnews/cli.py` – Typer-powered command-line interface.
- `src/vnsentiment/*.py` – OpenAI-powered sentiment analysis helper module and CLI (`vnnews-sentiment`).

## Symbols Utility

Fetch and store the full list of HOSE symbols using `vnstock`:

```bash
source .venv/bin/activate
python symbols/fetch_hose_symbols.py --output symbols/hose_symbols.csv
```

The script writes a CSV containing all HOSE tickers returned by `vnstock.Listing().symbols_by_exchange()`.

List the ICB industry hierarchy (industry / sub-industry / sector) via `vnstock`:

```bash
source .venv/bin/activate
python symbols/fetch_icb_industries.py --output symbols/icb_industries.csv
```

The script uses `vnstock.Listing().industries_icb()` and writes the result to CSV sorted by industry code.

## Notes

- Make sure outbound requests respect the target sites’ rate limits.
- `news-please` pulls many dependencies; use a virtual environment.
- Review saved JSON outputs before downstream processing.
- For sentiment scoring, set `OPENAI_API_KEY` before running `vnnews-sentiment score-file ...` or `vnnews-sentiment batch ...`.
