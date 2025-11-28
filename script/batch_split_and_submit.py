#!/usr/bin/env python3
"""
Split a large OpenAI batch JSONL file into <2M-token chunks, submit sequentially,
poll for completion, download each output next to this script, then concatenate
the outputs back in the original request directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI, OpenAIError

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(
        "Missing optional dependency `tiktoken`. Install it with `pip install tiktoken`."
    )


DEFAULT_MAX_TOKENS = 2_000_000
DEFAULT_POLL_SECONDS = 10
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_ENDPOINT = "/v1/chat/completions"
CHECKPOINT_SUFFIX = "_checkpoint.json"


def estimate_tokens(encoder: tiktoken.Encoding, text: str) -> int:
    """Approximate token count for a single JSONL line using cl100k_base encoding."""
    return len(encoder.encode(text))


def split_into_chunks(
    source: Path,
    dest_dir: Path,
    max_tokens: int,
    encoder: tiktoken.Encoding,
) -> List[Path]:
    """
    Split the source JSONL into token-bounded chunk files.

    Returns the list of chunk file paths in order.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[Path] = []
    current_lines: List[str] = []
    current_tokens = 0
    chunk_index = 1
    prefix = source.stem

    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            tokens = estimate_tokens(encoder, line)
            if tokens > max_tokens:
                raise ValueError(
                    f"Single line exceeds max tokens ({tokens} > {max_tokens}). "
                    "Reduce request size first."
                )

            if current_lines and current_tokens + tokens > max_tokens:
                chunk_path = dest_dir / f"{prefix}_part{chunk_index:03d}.jsonl"
                chunk_path.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
                chunks.append(chunk_path)
                chunk_index += 1
                current_lines = []
                current_tokens = 0

            current_lines.append(line)
            current_tokens += tokens

    if current_lines:
        chunk_path = dest_dir / f"{prefix}_part{chunk_index:03d}.jsonl"
        chunk_path.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
        chunks.append(chunk_path)

    return chunks


def upload_batch_file(client: OpenAI, path: Path):
    return client.files.create(file=path.open("rb"), purpose="batch")


def create_batch(client: OpenAI, file_id: str, endpoint: str, completion_window: str):
    return client.batches.create(
        input_file_id=file_id,
        endpoint=endpoint,
        completion_window=completion_window,
    )


def poll_batch(client: OpenAI, batch_id: str, interval_seconds: int):
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"Batch {batch_id} status: {status}")
        if status in {"completed", "failed", "cancelled", "canceled", "expired"}:
            return batch
        time.sleep(interval_seconds)


def download_file(client: OpenAI, file_id: str, dest: Path) -> None:
    response = client.files.content(file_id)
    with dest.open("wb") as handle:
        handle.write(response.read())


def concatenate_outputs(files: Iterable[Path], dest: Path) -> None:
    with dest.open("wb") as out:
        for file_path in files:
            with file_path.open("rb") as handle:
                out.write(handle.read())


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_checkpoint(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except IsADirectoryError:
            continue
        except PermissionError:
            print(f"Warning: could not remove {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a batch JSONL file into <2M-token pieces, submit to OpenAI batches "
            "sequentially, and concatenate outputs."
        )
    )
    parser.add_argument("jsonl_path", type=Path, help="Path to the large batch JSONL file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens per chunk (default: 2,000,000)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="Polling interval in seconds for batch status (default: 10)",
    )
    parser.add_argument(
        "--completion-window",
        default=DEFAULT_COMPLETION_WINDOW,
        help="Completion window for batches (default: 24h)",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Batch endpoint (default: /v1/chat/completions)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path: Path = args.jsonl_path

    if not source_path.exists():
        sys.exit(f"Input file not found: {source_path}")

    script_dir = Path(__file__).resolve().parent
    chunk_dir = script_dir
    combined_output_path = source_path.parent / f"{source_path.stem}_combined_output.jsonl"
    checkpoint_path = script_dir / f"{source_path.stem}{CHECKPOINT_SUFFIX}"

    encoder = tiktoken.get_encoding("cl100k_base")

    print(f"Splitting {source_path} into chunks...")
    try:
        chunk_paths = split_into_chunks(
            source=source_path,
            dest_dir=chunk_dir,
            max_tokens=args.max_tokens,
            encoder=encoder,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    if not chunk_paths:
        sys.exit("No data to submit after splitting.")

    client = OpenAI()
    completed_outputs: List[Path] = []
    checkpoint = load_checkpoint(checkpoint_path)

    for chunk_path in chunk_paths:
        chunk_key = str(chunk_path.name)
        existing_record = checkpoint.get(chunk_key)
        if existing_record:
            output_path = Path(existing_record.get("output_path", ""))
            if existing_record.get("status") == "completed" and output_path.exists():
                print(f"Skipping already completed chunk {chunk_path.name}")
                completed_outputs.append(output_path)
                continue

        print(f"Uploading {chunk_path.name}...")
        try:
            input_file = upload_batch_file(client, chunk_path)
            batch = create_batch(
                client=client,
                file_id=input_file.id,
                endpoint=args.endpoint,
                completion_window=args.completion_window,
            )
            print(f"Submitted batch {batch.id} for {chunk_path.name}")
            batch = poll_batch(client, batch.id, interval_seconds=args.poll_seconds)
        except OpenAIError as exc:
            sys.exit(f"OpenAI error while processing {chunk_path.name}: {exc}")

        if batch.status != "completed":
            sys.exit(f"Batch {batch.id} for {chunk_path.name} did not complete (status: {batch.status}).")

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            sys.exit(f"No output file for completed batch {batch.id} ({chunk_path.name}).")

        output_path = script_dir / f"{chunk_path.stem}_output.jsonl"
        print(f"Downloading output to {output_path}...")
        download_file(client, output_file_id, output_path)
        completed_outputs.append(output_path)

        checkpoint[chunk_key] = {
            "status": "completed",
            "output_path": str(output_path),
            "batch_id": batch.id,
        }
        save_checkpoint(checkpoint_path, checkpoint)

    print("Concatenating outputs...")
    concatenate_outputs(completed_outputs, combined_output_path)
    print(f"Combined output saved to {combined_output_path}")
    cleanup_paths(list(chunk_paths) + completed_outputs + [checkpoint_path])


if __name__ == "__main__":
    main()
