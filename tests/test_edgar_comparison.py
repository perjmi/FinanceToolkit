"""Cross-source comparison tests: EDGAR vs FMP fixture data.

These tests validate that EDGAR data produces comparable financial metrics
to FMP data. They use pre-saved fixture data to avoid hitting live APIs.

Tests are skipped if FMP fixture data has not been downloaded yet
(run scripts/download_sample_data.py first).
"""

# ruff: noqa

import json
import os
import pickle
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from financetoolkit import edgar_model

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
EDGAR_DIR = os.path.join(FIXTURES_DIR, "edgar")
FMP_DIR = os.path.join(FIXTURES_DIR, "fmp")

HAS_FMP_FIXTURES = os.path.exists(os.path.join(FMP_DIR, "income_statement.pickle"))


@pytest.fixture(autouse=True)
def clear_edgar_cache():
    """Clear EDGAR caches before each test."""
    edgar_model.clear_cache()
    yield
    edgar_model.clear_cache()


@pytest.fixture
def aapl_facts():
    with open(os.path.join(EDGAR_DIR, "AAPL_facts.json")) as f:
        return json.load(f)


@pytest.fixture
def ticker_cik_mapping():
    with open(os.path.join(EDGAR_DIR, "ticker_cik_mapping.json")) as f:
        return json.load(f)


class TestEdgarDataConsistency:
    """Test that EDGAR data is internally consistent."""

    def test_revenue_positive(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )
        for date, val in items["Revenues"].items():
            assert val > 0, f"Revenue should be positive for {date}"

    def test_total_assets_positive(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.BALANCE_CONCEPTS, quarter=False
        )
        for date, val in items["TotalAssets"].items():
            assert val > 0, f"Total assets should be positive for {date}"

    def test_assets_equals_liabilities_plus_equity(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.BALANCE_CONCEPTS, quarter=False
        )

        for date in items["TotalAssets"]:
            if date in items.get("TotalLiabilitiesAndStockholdersEquity", {}):
                assets = items["TotalAssets"][date]
                lie = items["TotalLiabilitiesAndStockholdersEquity"][date]
                assert abs(assets - lie) < 1, (
                    f"Assets ({assets}) should equal L+E ({lie}) for {date}"
                )

    def test_gross_profit_equals_revenue_minus_cogs(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )

        for date in items.get("GrossProfit", {}):
            if date in items.get("Revenues", {}) and date in items.get("CostOfRevenue", {}):
                revenue = items["Revenues"][date]
                cogs = items["CostOfRevenue"][date]
                gp = items["GrossProfit"][date]
                assert abs(gp - (revenue - cogs)) < 1000000, (
                    f"Gross profit ({gp}) should ~= Revenue ({revenue}) - COGS ({cogs}) for {date}"
                )

    def test_operating_cash_flow_reasonable(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.CASHFLOW_CONCEPTS, quarter=False
        )

        for date, val in items.get("NetCashFromOperatingActivities", {}).items():
            # AAPL operating cash flow should be positive and in tens of billions
            assert val > 50_000_000_000, (
                f"AAPL operating cash flow should be > $50B for {date}, got {val}"
            )


class TestEdgarFullPipeline:
    """Test the full EDGAR pipeline produces valid DataFrames."""

    def test_income_statement_has_expected_rows(self, ticker_cik_mapping, aapl_facts):
        mock_cik_response = MagicMock()
        mock_cik_response.status_code = 200
        mock_cik_response.json.return_value = ticker_cik_mapping

        mock_facts_response = MagicMock()
        mock_facts_response.status_code = 200
        mock_facts_response.json.return_value = aapl_facts

        def side_effect(url, timeout=30):
            if "company_tickers" in url:
                return mock_cik_response
            return mock_facts_response

        with patch.object(edgar_model, "_rate_limited_get", side_effect=side_effect):
            df = edgar_model.get_financial_statement("AAPL", "income")

        expected_fields = [
            "Revenues",
            "CostOfRevenue",
            "GrossProfit",
            "OperatingIncomeLoss",
            "NetIncomeLoss",
            "EarningsPerShareBasic",
            "EarningsPerShareDiluted",
        ]
        for field in expected_fields:
            assert field in df.index, f"Missing expected field: {field}"

    def test_balance_sheet_has_expected_rows(self, ticker_cik_mapping, aapl_facts):
        mock_cik_response = MagicMock()
        mock_cik_response.status_code = 200
        mock_cik_response.json.return_value = ticker_cik_mapping

        mock_facts_response = MagicMock()
        mock_facts_response.status_code = 200
        mock_facts_response.json.return_value = aapl_facts

        def side_effect(url, timeout=30):
            if "company_tickers" in url:
                return mock_cik_response
            return mock_facts_response

        with patch.object(edgar_model, "_rate_limited_get", side_effect=side_effect):
            df = edgar_model.get_financial_statement("AAPL", "balance")

        expected_fields = [
            "TotalAssets",
            "TotalLiabilities",
            "TotalStockholdersEquity",
            "CashAndCashEquivalents",
        ]
        for field in expected_fields:
            assert field in df.index, f"Missing expected field: {field}"

    def test_cashflow_has_expected_rows(self, ticker_cik_mapping, aapl_facts):
        mock_cik_response = MagicMock()
        mock_cik_response.status_code = 200
        mock_cik_response.json.return_value = ticker_cik_mapping

        mock_facts_response = MagicMock()
        mock_facts_response.status_code = 200
        mock_facts_response.json.return_value = aapl_facts

        def side_effect(url, timeout=30):
            if "company_tickers" in url:
                return mock_cik_response
            return mock_facts_response

        with patch.object(edgar_model, "_rate_limited_get", side_effect=side_effect):
            df = edgar_model.get_financial_statement("AAPL", "cashflow")

        expected_fields = [
            "NetCashFromOperatingActivities",
            "NetCashFromInvestingActivities",
            "NetCashFromFinancingActivities",
            "CapitalExpenditures",
        ]
        for field in expected_fields:
            assert field in df.index, f"Missing expected field: {field}"


@pytest.mark.skipif(not HAS_FMP_FIXTURES, reason="FMP fixture data not downloaded")
class TestEdgarVsFmpComparison:
    """Compare EDGAR data against FMP fixture data.

    These tests require FMP fixture data to be downloaded first:
    python scripts/download_sample_data.py
    """

    @pytest.fixture
    def fmp_income(self):
        with open(os.path.join(FMP_DIR, "income_statement.pickle"), "rb") as f:
            return pickle.load(f)

    @pytest.fixture
    def fmp_balance(self):
        with open(os.path.join(FMP_DIR, "balance_sheet.pickle"), "rb") as f:
            return pickle.load(f)

    @pytest.fixture
    def fmp_cashflow(self):
        with open(os.path.join(FMP_DIR, "cash_flow.pickle"), "rb") as f:
            return pickle.load(f)

    def test_aapl_revenue_within_tolerance(self, fmp_income, aapl_facts):
        """AAPL revenue from EDGAR should be within 1% of FMP."""
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )

        edgar_revenue = items["Revenues"]

        if "AAPL" in fmp_income.index.get_level_values(0):
            fmp_aapl = fmp_income.loc["AAPL"]
            if "Revenue" in fmp_aapl.index:
                fmp_rev = fmp_aapl.loc["Revenue"]
                for period in fmp_rev.index:
                    year = str(period.year)
                    # Find matching EDGAR date
                    for edgar_date, edgar_val in edgar_revenue.items():
                        if edgar_date.startswith(year):
                            fmp_val = fmp_rev[period]
                            if fmp_val != 0:
                                pct_diff = abs(edgar_val - fmp_val) / abs(fmp_val)
                                assert pct_diff < 0.01, (
                                    f"Revenue mismatch for AAPL {year}: "
                                    f"EDGAR={edgar_val}, FMP={fmp_val}, diff={pct_diff:.2%}"
                                )
                            break

    def test_aapl_net_income_within_tolerance(self, fmp_income, aapl_facts):
        """AAPL net income from EDGAR should be within 1% of FMP."""
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )

        edgar_ni = items["NetIncomeLoss"]

        if "AAPL" in fmp_income.index.get_level_values(0):
            fmp_aapl = fmp_income.loc["AAPL"]
            if "Net Income" in fmp_aapl.index:
                fmp_ni = fmp_aapl.loc["Net Income"]
                for period in fmp_ni.index:
                    year = str(period.year)
                    for edgar_date, edgar_val in edgar_ni.items():
                        if edgar_date.startswith(year):
                            fmp_val = fmp_ni[period]
                            if fmp_val != 0:
                                pct_diff = abs(edgar_val - fmp_val) / abs(fmp_val)
                                assert pct_diff < 0.01, (
                                    f"Net income mismatch for AAPL {year}: "
                                    f"EDGAR={edgar_val}, FMP={fmp_val}, diff={pct_diff:.2%}"
                                )
                            break

    def test_aapl_total_assets_within_tolerance(self, fmp_balance, aapl_facts):
        """AAPL total assets from EDGAR should be within 1% of FMP."""
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.BALANCE_CONCEPTS, quarter=False
        )

        edgar_ta = items["TotalAssets"]

        if "AAPL" in fmp_balance.index.get_level_values(0):
            fmp_aapl = fmp_balance.loc["AAPL"]
            if "Total Assets" in fmp_aapl.index:
                fmp_ta = fmp_aapl.loc["Total Assets"]
                for period in fmp_ta.index:
                    year = str(period.year)
                    for edgar_date, edgar_val in edgar_ta.items():
                        if edgar_date.startswith(year):
                            fmp_val = fmp_ta[period]
                            if fmp_val != 0:
                                pct_diff = abs(edgar_val - fmp_val) / abs(fmp_val)
                                assert pct_diff < 0.01, (
                                    f"Total assets mismatch for AAPL {year}: "
                                    f"EDGAR={edgar_val}, FMP={fmp_val}, diff={pct_diff:.2%}"
                                )
                            break

    def test_aapl_operating_cash_flow_within_tolerance(self, fmp_cashflow, aapl_facts):
        """AAPL operating cash flow from EDGAR should be within 1% of FMP."""
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.CASHFLOW_CONCEPTS, quarter=False
        )

        edgar_ocf = items["NetCashFromOperatingActivities"]

        if "AAPL" in fmp_cashflow.index.get_level_values(0):
            fmp_aapl = fmp_cashflow.loc["AAPL"]
            if "Cash Flow from Operations" in fmp_aapl.index:
                fmp_ocf = fmp_aapl.loc["Cash Flow from Operations"]
                for period in fmp_ocf.index:
                    year = str(period.year)
                    for edgar_date, edgar_val in edgar_ocf.items():
                        if edgar_date.startswith(year):
                            fmp_val = fmp_ocf[period]
                            if fmp_val != 0:
                                pct_diff = abs(edgar_val - fmp_val) / abs(fmp_val)
                                assert pct_diff < 0.01, (
                                    f"Operating CF mismatch for AAPL {year}: "
                                    f"EDGAR={edgar_val}, FMP={fmp_val}, diff={pct_diff:.2%}"
                                )
                            break
