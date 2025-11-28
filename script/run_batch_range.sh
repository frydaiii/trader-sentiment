#!/usr/bin/env bash
set -euo pipefail

# Run batch_split_and_submit.py for months 06-12 of 2023 with 300s polling.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT_DIR}/script/batch_split_and_submit.py"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Script not found: ${SCRIPT}" >&2
  exit 1
fi

for month in {08..12}; do
  INPUT="${ROOT_DIR}/data/sentiment/entities/batch_requests/2023-${month}/requests.jsonl"
  echo "Processing ${INPUT}..."
  if [[ ! -f "${INPUT}" ]]; then
    echo "Skipping (missing file): ${INPUT}"
    continue
  fi
  python "${SCRIPT}" "${INPUT}" --poll-seconds 60
done
