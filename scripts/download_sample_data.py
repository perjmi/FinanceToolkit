"""Download sample FMP data for testing and comparison with EDGAR data."""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from financetoolkit import Toolkit

API_KEY = "0sF7EpXPUKCR7Vaxf9CKPZqlpj0Fmrup"
TICKERS = ["AAPL", "MSFT", "GOOGL"]
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "fmp")
os.makedirs(FIXTURE_DIR, exist_ok=True)


def main():
    print(f"Downloading FMP data for {TICKERS} ({START_DATE} to {END_DATE})...")

    toolkit = Toolkit(
        tickers=TICKERS,
        api_key=API_KEY,
        start_date=START_DATE,
        end_date=END_DATE,
        quarterly=False,
        progress_bar=True,
    )

    print("Fetching income statements...")
    income = toolkit.get_income_statement()
    if income is not None:
        with open(os.path.join(FIXTURE_DIR, "income_statement.pickle"), "wb") as f:
            pickle.dump(income, f)
        print(f"  Saved income statement: {income.shape}")

    print("Fetching balance sheet statements...")
    balance = toolkit.get_balance_sheet_statement()
    if balance is not None:
        with open(os.path.join(FIXTURE_DIR, "balance_sheet.pickle"), "wb") as f:
            pickle.dump(balance, f)
        print(f"  Saved balance sheet: {balance.shape}")

    print("Fetching cash flow statements...")
    cashflow = toolkit.get_cash_flow_statement()
    if cashflow is not None:
        with open(os.path.join(FIXTURE_DIR, "cash_flow.pickle"), "wb") as f:
            pickle.dump(cashflow, f)
        print(f"  Saved cash flow: {cashflow.shape}")

    print("Done! Files saved to:", FIXTURE_DIR)


if __name__ == "__main__":
    main()
