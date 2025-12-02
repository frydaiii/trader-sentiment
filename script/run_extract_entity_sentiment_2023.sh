#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Create it before running this script."
  exit 1
fi

source .venv/bin/activate

for month in $(seq -w 1 12); do
  input="data/sentiment/entities/batch_requests/2023-${month}/requests_combined_output.jsonl"
  if [[ ! -f "$input" ]]; then
    echo "Skipping $input (not found)"
    continue
  fi
  echo "Processing $input"
  vnnews-sentiment extract-sentiment "$input"
done
