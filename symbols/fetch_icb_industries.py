"""Fetch and save the ICB industry hierarchy (sector/sub-industry) using vnstock."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from vnstock import Listing


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first matching column (case-insensitive) or None."""
    normalized = {col.lower(): col for col in columns}
    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]
    return None


def fetch_icb_industries(output: Path) -> Path:
    """Fetch ICB industry tree and save to CSV."""
    df = Listing().industries_icb()
    if df is None or df.empty:
        raise RuntimeError("vnstock returned no industry data.")

    # Try to keep a tidy column order if present.
    code_col = detect_column(df.columns, ("icb_code", "code", "industry_code"))
    name_col = detect_column(df.columns, ("icb_name", "name", "industry_name"))
    level_col = detect_column(df.columns, ("icb_level", "level"))
    parent_col = detect_column(df.columns, ("parent_code", "parent", "icb_parent"))

    ordered_cols = [
        col for col in (code_col, name_col, level_col, parent_col) if col
    ]
    if ordered_cols:
        df = df[ordered_cols + [c for c in df.columns if c not in ordered_cols]]

    if code_col:
        df = df.sort_values(by=code_col)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ICB industries (sector/sub-industry) using vnstock and save to CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("symbols/icb_industries.csv"),
        help="Path to the CSV file to write (default: symbols/icb_industries.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = fetch_icb_industries(args.output)
    print(f"Saved ICB industries to {output_path}")


if __name__ == "__main__":
    main()
