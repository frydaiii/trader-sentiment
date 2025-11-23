"""Fetch and save all HOSE exchange symbols using the vnstock library."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from vnstock import Listing


def detect_exchange_column(columns: Iterable[str]) -> str:
    """Return the column name that carries exchange identifiers."""
    normalized = {col.lower(): col for col in columns}
    for candidate in ("exchange", "floor", "comgroupcode", "stock_exchange"):
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "Could not find an exchange column; expected one of exchange/floor/comGroupCode"
    )


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first matching column (case-insensitive) or None."""
    normalized = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def ensure_icb_code(df, ticker_col: str) -> tuple:
    """Ensure the dataframe has an ICB code column; fetch and merge if missing."""
    icb_col = detect_column(
        df.columns,
        (
            "icb_code",
            "icbcode",
            "icb",
            "icb_code4",
            "icb_code3",
            "icb_code2",
            "icb_code1",
            "industry_code",
            "icbindustrycode",
            "icbindustry",
        ),
    )
    if icb_col:
        if icb_col != "icb_code":
            df = df.rename(columns={icb_col: "icb_code"})
            icb_col = "icb_code"
        return df, icb_col

    industries = Listing().symbols_by_industries()
    if industries is None or industries.empty:
        raise RuntimeError("vnstock returned no industry mapping to attach ICB codes.")

    industries_ticker_col = detect_column(
        industries.columns, ("ticker", "symbol", "code", "stockcode")
    )
    industries_icb_col = detect_column(
        industries.columns,
        (
            "icb_code",
            "icbcode",
            "icb",
            "icb_code4",
            "icb_code3",
            "icb_code2",
            "icb_code1",
            "industry_code",
            "icbindustrycode",
            "icbindustry",
        ),
    )
    if not industries_ticker_col or not industries_icb_col:
        raise RuntimeError(
            "Could not detect ticker/ICB columns in vnstock symbols_by_industries response."
        )

    merged = df.merge(
        industries[[industries_ticker_col, industries_icb_col]],
        left_on=ticker_col,
        right_on=industries_ticker_col,
        how="left",
    )
    # Drop duplicate ticker column from industries if it's different.
    if industries_ticker_col != ticker_col:
        merged.drop(columns=[industries_ticker_col], inplace=True)

    # Normalize the ICB column name to icb_code for clarity.
    if industries_icb_col != "icb_code":
        merged.rename(columns={industries_icb_col: "icb_code"}, inplace=True)
        icb_col = "icb_code"
    else:
        icb_col = industries_icb_col

    return merged, icb_col


def fetch_hose_symbols(output: Path) -> Path:
    """Fetch HOSE symbols and write them to CSV."""
    listing = Listing()
    df = listing.symbols_by_exchange()
    if df is None or df.empty:
        raise RuntimeError("vnstock returned no symbol data.")

    exchange_col = detect_exchange_column(df.columns)
    hose_df = df[df[exchange_col].astype(str).str.upper().isin({"HOSE", "HSX"})].copy()
    if hose_df.empty:
        raise RuntimeError("No HOSE symbols found in vnstock response.")

    ticker_col = detect_column(hose_df.columns, ("ticker", "symbol", "code", "stockcode"))
    if not ticker_col:
        raise RuntimeError("Could not find ticker column among HOSE symbols.")

    hose_df, _ = ensure_icb_code(hose_df, ticker_col)

    # Sort for deterministic output
    hose_df.sort_values(by=ticker_col, inplace=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    hose_df.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch all HOSE symbols using vnstock and save to CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hose_symbols.csv"),
        help="Path to the CSV file to write (default: hose_symbols.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = fetch_hose_symbols(args.output)
    print(f"Saved HOSE symbols to {output_path}")


if __name__ == "__main__":
    main()
