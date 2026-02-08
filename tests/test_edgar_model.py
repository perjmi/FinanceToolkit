"""EDGAR Model Tests"""

# ruff: noqa

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from financetoolkit import edgar_model

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "edgar")


@pytest.fixture(autouse=True)
def clear_edgar_cache():
    """Clear EDGAR caches before each test."""
    edgar_model.clear_cache()
    yield
    edgar_model.clear_cache()


@pytest.fixture
def aapl_facts():
    """Load AAPL company facts fixture."""
    with open(os.path.join(FIXTURES_DIR, "AAPL_facts.json")) as f:
        return json.load(f)


@pytest.fixture
def ticker_cik_mapping():
    """Load ticker-CIK mapping fixture."""
    with open(os.path.join(FIXTURES_DIR, "ticker_cik_mapping.json")) as f:
        return json.load(f)


class TestGetCikForTicker:
    def test_valid_ticker(self, ticker_cik_mapping):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ticker_cik_mapping

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            cik = edgar_model.get_cik_for_ticker("AAPL")

        assert cik == "0000320193"

    def test_valid_ticker_case_insensitive(self, ticker_cik_mapping):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ticker_cik_mapping

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            cik = edgar_model.get_cik_for_ticker("aapl")

        assert cik == "0000320193"

    def test_invalid_ticker(self, ticker_cik_mapping):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ticker_cik_mapping

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            cik = edgar_model.get_cik_for_ticker("INVALIDTICKER")

        assert cik is None

    def test_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            cik = edgar_model.get_cik_for_ticker("AAPL")

        assert cik is None

    def test_cik_caching(self, ticker_cik_mapping):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ticker_cik_mapping

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response) as mock_get:
            edgar_model.get_cik_for_ticker("AAPL")
            edgar_model.get_cik_for_ticker("MSFT")
            # Should only call the API once since second call uses cache
            assert mock_get.call_count == 1


class TestGetCompanyFacts:
    def test_valid_cik(self, aapl_facts):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = aapl_facts

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            facts = edgar_model.get_company_facts("0000320193")

        assert facts is not None
        assert facts["entityName"] == "Apple Inc."
        assert "us-gaap" in facts["facts"]

    def test_http_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            facts = edgar_model.get_company_facts("0000000000")

        assert facts is None

    def test_facts_caching(self, aapl_facts):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = aapl_facts

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response) as mock_get:
            edgar_model.get_company_facts("0000320193")
            edgar_model.get_company_facts("0000320193")
            assert mock_get.call_count == 1


class TestExtractFinancialItems:
    def test_annual_income_items(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )

        # Check revenue is extracted
        assert "Revenues" in items
        revenue = items["Revenues"]
        assert "2020-09-26" in revenue
        assert revenue["2020-09-26"] == 274515000000
        assert revenue["2023-09-30"] == 383285000000

    def test_annual_income_net_income(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )

        assert "NetIncomeLoss" in items
        net_income = items["NetIncomeLoss"]
        assert net_income["2021-09-25"] == 94680000000

    def test_annual_balance_items(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.BALANCE_CONCEPTS, quarter=False
        )

        assert "TotalAssets" in items
        assert items["TotalAssets"]["2022-09-24"] == 352755000000

        assert "TotalLiabilities" in items
        assert items["TotalLiabilities"]["2020-09-26"] == 258549000000

    def test_annual_cashflow_items(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.CASHFLOW_CONCEPTS, quarter=False
        )

        assert "NetCashFromOperatingActivities" in items
        opcf = items["NetCashFromOperatingActivities"]
        assert opcf["2021-09-25"] == 104038000000

    def test_quarterly_filter(self, aapl_facts):
        """Quarterly data should only include 10-Q entries."""
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=True
        )

        revenue = items.get("Revenues", {})
        # 10-Q entries should be present
        assert "2021-03-27" in revenue or "2023-04-01" in revenue
        # 10-K entries should NOT be present
        assert "2020-09-26" not in revenue

    def test_eps_extraction(self, aapl_facts):
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )

        assert "EarningsPerShareBasic" in items
        assert items["EarningsPerShareBasic"]["2022-09-24"] == 6.15

        assert "EarningsPerShareDiluted" in items
        assert items["EarningsPerShareDiluted"]["2023-09-30"] == 6.13

    def test_concept_fallback(self, aapl_facts):
        """Test that alternative concept names work as fallback."""
        # Revenue uses RevenueFromContractWithCustomerExcludingAssessedTax
        # (not the first name "Revenues" in INCOME_CONCEPTS)
        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.INCOME_CONCEPTS, quarter=False
        )
        assert len(items["Revenues"]) == 4  # 4 years of data

    def test_deduplication_prefers_latest_filing(self, aapl_facts):
        """When multiple filings exist for the same period, prefer the latest."""
        # Add a duplicate filing with different date
        us_gaap = aapl_facts["facts"]["us-gaap"]
        us_gaap["Assets"]["units"]["USD"].append(
            {"end": "2023-09-30", "val": 999999999999, "form": "10-K", "filed": "2023-10-01"}
        )

        items = edgar_model._extract_financial_items(
            aapl_facts, edgar_model.BALANCE_CONCEPTS, quarter=False
        )

        # Should use the filing from 2023-11-03 (later) rather than 2023-10-01
        assert items["TotalAssets"]["2023-09-30"] == 352583000000


class TestGetFinancialStatement:
    def test_invalid_statement_type(self):
        with pytest.raises(ValueError, match="Please choose either"):
            edgar_model.get_financial_statement("AAPL", "invalid")

    def test_income_statement(self, ticker_cik_mapping, aapl_facts):
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

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Revenues" in df.index or "NetIncomeLoss" in df.index
        # Check that columns are PeriodIndex
        assert isinstance(df.columns, pd.PeriodIndex)

    def test_balance_sheet(self, ticker_cik_mapping, aapl_facts):
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

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "TotalAssets" in df.index

    def test_cashflow_statement(self, ticker_cik_mapping, aapl_facts):
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

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "NetCashFromOperatingActivities" in df.index

    def test_invalid_ticker_returns_empty(self, ticker_cik_mapping):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ticker_cik_mapping

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            df = edgar_model.get_financial_statement("INVALIDXYZ", "income")

        # Should have error column but no data rows
        assert df.empty or "EDGAR NO CIK FOUND" in df.columns

    def test_no_data_cik(self, ticker_cik_mapping):
        mock_cik_response = MagicMock()
        mock_cik_response.status_code = 200
        mock_cik_response.json.return_value = ticker_cik_mapping

        mock_facts_response = MagicMock()
        mock_facts_response.status_code = 404

        def side_effect(url, timeout=30):
            if "company_tickers" in url:
                return mock_cik_response
            return mock_facts_response

        with patch.object(edgar_model, "_rate_limited_get", side_effect=side_effect):
            df = edgar_model.get_financial_statement("AAPL", "income")

        assert "EDGAR NO DATA FOUND" in df.columns

    def test_fallback_flag(self, ticker_cik_mapping):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ticker_cik_mapping

        with patch.object(edgar_model, "_rate_limited_get", return_value=mock_response):
            df = edgar_model.get_financial_statement(
                "INVALIDXYZ", "income", fallback=True
            )

        assert "EDGAR NO CIK FOUND FALLBACK" in df.columns


class TestCollectFinancialStatementsWithEdgar:
    """Test EDGAR integration in fundamentals_model.collect_financial_statements."""

    def test_edgar_enforce_source(self, ticker_cik_mapping, aapl_facts):
        from financetoolkit import fundamentals_model

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
            result, stats, no_data = fundamentals_model.collect_financial_statements(
                tickers="AAPL",
                statement="income",
                api_key="",
                enforce_source="EDGAR",
                progress_bar=False,
            )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "AAPL" in result.index.get_level_values(0)
