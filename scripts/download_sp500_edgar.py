"""Download quarterly EDGAR financial data for all S&P 500 constituents (last 10 years).

This script:
1. Scrapes Wikipedia for current and historical S&P 500 constituents
2. Downloads quarterly income, balance sheet, and cash flow data from SEC EDGAR
3. Saves results as pickle files + metadata CSV and JSON log

Usage:
    python scripts/download_sp500_edgar.py
"""

import json
import os
import pickle
import sys
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from financetoolkit import Toolkit
from financetoolkit.edgar_model import get_cik_for_ticker

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START_DATE = "2016-01-01"
END_DATE = "2026-02-08"
CUTOFF_DATE = pd.Timestamp("2016-02-08")
BATCH_SIZE = 50
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_quarterly")


def get_sp500_constituents():
    """Scrape Wikipedia for current and historical S&P 500 constituents.

    Returns:
        tuple: (all_tickers sorted list, constituents_df with metadata)
    """
    print("Fetching S&P 500 constituent list from Wikipedia...")
    resp = requests.get(WIKI_URL, headers={"User-Agent": "SP500DataScript/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.find_all("table", {"class": "wikitable"})

    # Table 1: Current constituents
    current_table = tables[0]
    rows = current_table.find_all("tr")[1:]  # skip header

    current = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        ticker = cols[0].get_text(strip=True).replace(".", "-")
        name = cols[1].get_text(strip=True)
        sector = cols[3].get_text(strip=True)
        date_added = cols[6].get_text(strip=True) if len(cols) > 6 else ""
        current.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                "Date Added": date_added,
                "Status": "Current",
            }
        )

    current_tickers = {row["Ticker"] for row in current}
    print(f"  Current constituents: {len(current_tickers)}")

    # Table 2: Historical changes (additions and removals)
    changes_table = tables[1]
    change_rows = changes_table.find_all("tr")[1:]  # skip header

    removed = []
    for row in change_rows:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        date_str = cols[0].get_text(strip=True)
        try:
            change_date = pd.to_datetime(date_str)
        except (ValueError, TypeError):
            continue

        if change_date < CUTOFF_DATE:
            continue

        # The removed ticker is typically in columns index 3 or 4
        # Wikipedia format: Date | Added Ticker | Added Name | Removed Ticker | Removed Name | Reason
        removed_ticker = cols[3].get_text(strip=True).replace(".", "-")
        removed_name = cols[4].get_text(strip=True) if len(cols) > 4 else ""

        if removed_ticker and removed_ticker not in current_tickers:
            removed.append(
                {
                    "Ticker": removed_ticker,
                    "Name": removed_name,
                    "Sector": "",
                    "Date Added": "",
                    "Date Removed": date_str,
                    "Status": "Removed",
                }
            )

    # Deduplicate removed tickers (keep first occurrence = most recent removal)
    seen = set()
    unique_removed = []
    for r in removed:
        if r["Ticker"] not in seen:
            seen.add(r["Ticker"])
            unique_removed.append(r)

    print(f"  Removed constituents (since {CUTOFF_DATE.date()}): {len(unique_removed)}")

    # Build combined DataFrame
    for row in current:
        row["Date Removed"] = ""

    all_rows = current + unique_removed
    constituents_df = pd.DataFrame(all_rows)

    all_tickers = sorted(constituents_df["Ticker"].unique().tolist())
    print(f"  Total unique tickers: {len(all_tickers)}")

    return all_tickers, constituents_df


def _ensure_multiindex(df, ticker):
    """Convert a single-ticker flat DataFrame to MultiIndex format.

    When the Toolkit receives a single ticker, it returns a DataFrame with a plain
    Index (metric names). Multi-ticker results use a MultiIndex (ticker, metric).
    This normalizes the single-ticker case for consistent concatenation.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        return df
    # Single ticker: index is metric names, add ticker as level 0
    df = df.copy()
    df.index = pd.MultiIndex.from_product([[ticker], df.index])
    return df


def download_batch(tickers, batch_num, total_batches):
    """Download quarterly financial data for a batch of tickers.

    Returns:
        tuple: (income_df, balance_df, cashflow_df, failed_tickers)
    """
    print(f"\n--- Batch {batch_num}/{total_batches}: {len(tickers)} tickers ---")
    print(f"  Tickers: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")

    failed = []
    try:
        toolkit = Toolkit(
            tickers=tickers,
            quarterly=True,
            enforce_source="EDGAR",
            start_date=START_DATE,
            end_date=END_DATE,
            progress_bar=True,
        )

        print("  Fetching income statements...")
        income = toolkit.get_income_statement()

        print("  Fetching balance sheets...")
        balance = toolkit.get_balance_sheet_statement()

        print("  Fetching cash flow statements...")
        cashflow = toolkit.get_cash_flow_statement()

        # Normalize single-ticker results to MultiIndex
        if len(tickers) == 1:
            income = _ensure_multiindex(income, tickers[0])
            balance = _ensure_multiindex(balance, tickers[0])
            cashflow = _ensure_multiindex(cashflow, tickers[0])

        # Identify tickers that returned empty data
        if income is not None and not income.empty:
            if isinstance(income.index, pd.MultiIndex):
                returned_tickers = set(income.index.get_level_values(0).unique())
            else:
                returned_tickers = set()
            failed = [t for t in tickers if t not in returned_tickers]
        else:
            failed = list(tickers)
            income = pd.DataFrame()
            balance = pd.DataFrame()
            cashflow = pd.DataFrame()

        if balance is None:
            balance = pd.DataFrame()
        if cashflow is None:
            cashflow = pd.DataFrame()

        print(f"  Success: {len(tickers) - len(failed)}, Failed: {len(failed)}")
        if failed:
            print(f"  Failed tickers: {', '.join(failed)}")

        return income, balance, cashflow, failed

    except Exception as e:
        print(f"  Batch failed entirely: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), list(tickers)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # Step 1: Get constituents
    all_tickers, constituents_df = get_sp500_constituents()

    constituents_path = os.path.join(OUTPUT_DIR, "sp500_constituents.csv")
    constituents_df.to_csv(constituents_path, index=False)
    print(f"\nSaved constituents to {constituents_path}")

    # Pre-filter: only keep tickers with valid SEC CIK mappings
    print("\nChecking CIK availability for all tickers...")
    valid_tickers = []
    no_cik_tickers = []
    for ticker in all_tickers:
        cik = get_cik_for_ticker(ticker)
        if cik is not None:
            valid_tickers.append(ticker)
        else:
            no_cik_tickers.append(ticker)
    print(f"  Valid CIK: {len(valid_tickers)}, No CIK: {len(no_cik_tickers)}")
    if no_cik_tickers:
        print(f"  Skipping (no CIK): {', '.join(no_cik_tickers)}")

    # Step 2: Download in batches
    batches = [
        valid_tickers[i : i + BATCH_SIZE]
        for i in range(0, len(valid_tickers), BATCH_SIZE)
    ]
    total_batches = len(batches)
    print(f"\nDownloading quarterly EDGAR data in {total_batches} batches of up to {BATCH_SIZE}...")

    all_income = []
    all_balance = []
    all_cashflow = []
    all_failed = []

    for i, batch in enumerate(batches, 1):
        income, balance, cashflow, failed = download_batch(batch, i, total_batches)

        if not income.empty:
            all_income.append(income)
        if not balance.empty:
            all_balance.append(balance)
        if not cashflow.empty:
            all_cashflow.append(cashflow)
        all_failed.extend(failed)

    # Retry failed tickers one-by-one (batch failures are often caused by
    # a single bad ticker poisoning the whole batch via date-parse errors)
    if all_failed:
        retry_tickers = list(all_failed)
        all_failed = []
        print(f"\n--- Retrying {len(retry_tickers)} failed tickers individually ---")
        for j, ticker in enumerate(retry_tickers, 1):
            income, balance, cashflow, failed = download_batch(
                [ticker], j, len(retry_tickers)
            )
            if not income.empty:
                all_income.append(income)
            if not balance.empty:
                all_balance.append(balance)
            if not cashflow.empty:
                all_cashflow.append(cashflow)
            all_failed.extend(failed)

    # Step 3: Combine and save results
    print("\n--- Combining results ---")

    def concat_results(frames, name):
        if not frames:
            print(f"  {name}: No data collected")
            return pd.DataFrame()
        combined = pd.concat(frames)
        # Remove duplicate index entries if any
        combined = combined[~combined.index.duplicated(keep="first")]
        print(f"  {name}: {combined.shape}")
        return combined

    combined_income = concat_results(all_income, "Income statements")
    combined_balance = concat_results(all_balance, "Balance sheets")
    combined_cashflow = concat_results(all_cashflow, "Cash flows")

    # Save pickle files
    for df, filename in [
        (combined_income, "income_statements.pickle"),
        (combined_balance, "balance_sheets.pickle"),
        (combined_cashflow, "cash_flows.pickle"),
    ]:
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "wb") as f:
            pickle.dump(df, f)
        print(f"  Saved {path}")

    # Step 4: Write download log
    elapsed = time.time() - start_time
    log = {
        "timestamp": datetime.now().isoformat(),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "total_tickers": len(all_tickers),
        "valid_cik_tickers": len(valid_tickers),
        "no_cik_tickers": sorted(no_cik_tickers),
        "no_cik_count": len(no_cik_tickers),
        "successful_tickers": len(valid_tickers) - len(all_failed),
        "failed_count": len(all_failed),
        "failed_tickers": sorted(all_failed),
        "batch_size": BATCH_SIZE,
        "total_batches": total_batches,
        "elapsed_seconds": round(elapsed, 1),
        "income_shape": list(combined_income.shape) if not combined_income.empty else None,
        "balance_shape": list(combined_balance.shape) if not combined_balance.empty else None,
        "cashflow_shape": list(combined_cashflow.shape) if not combined_cashflow.empty else None,
    }

    log_path = os.path.join(OUTPUT_DIR, "download_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved {log_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Download complete in {elapsed:.1f}s")
    print(f"  Total tickers: {log['total_tickers']}")
    print(f"  Successful: {log['successful_tickers']}")
    print(f"  Failed: {log['failed_count']}")
    if log["income_shape"]:
        print(f"  Income statements shape: {log['income_shape']}")
    if log["balance_shape"]:
        print(f"  Balance sheets shape: {log['balance_shape']}")
    if log["cashflow_shape"]:
        print(f"  Cash flows shape: {log['cashflow_shape']}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
