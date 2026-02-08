"""EDGAR Module"""

__docformat__ = "google"

import time
import threading

import pandas as pd
import requests

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# Rate limiting: SEC EDGAR allows 10 requests per second
_rate_lock = threading.Lock()
_last_request_time = 0.0

# Cache for ticker-to-CIK mapping and company facts
_ticker_cik_cache: dict[str, str] = {}
_company_facts_cache: dict[str, dict] = {}

USER_AGENT = "FinanceToolkit research@financetoolkit.com"

# XBRL concept mappings: each output field maps to ordered list of concept names to try.
# Companies may use different XBRL tags; we try each in order until one is found.

INCOME_CONCEPTS: dict[str, list[str]] = {
    "Revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenueNet",
    ],
    "CostOfRevenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ],
    "GrossProfit": [
        "GrossProfit",
    ],
    "ResearchAndDevelopmentExpense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "SellingGeneralAndAdministrativeExpense": [
        "SellingGeneralAndAdministrativeExpense",
        "SellingAndMarketingExpense",
        "GeneralAndAdministrativeExpense",
    ],
    "OperatingExpenses": [
        "OperatingExpenses",
        "CostsAndExpenses",
    ],
    "OperatingIncomeLoss": [
        "OperatingIncomeLoss",
    ],
    "InterestExpense": [
        "InterestExpense",
        "InterestExpenseDebt",
    ],
    "InterestIncome": [
        "InterestIncome",
        "InvestmentIncomeInterest",
    ],
    "IncomeTaxExpenseBenefit": [
        "IncomeTaxExpenseBenefit",
    ],
    "NetIncomeLoss": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "EarningsPerShareBasic": [
        "EarningsPerShareBasic",
    ],
    "EarningsPerShareDiluted": [
        "EarningsPerShareDiluted",
    ],
    "WeightedAverageNumberOfSharesOutstandingBasic": [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ],
    "WeightedAverageNumberOfDilutedSharesOutstanding": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "DepreciationAndAmortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ],
    "IncomeBeforeTax": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "NetIncomeFromContinuingOperations": [
        "IncomeLossFromContinuingOperations",
    ],
    "OtherIncomeExpenseNet": [
        "NonoperatingIncomeExpense",
        "OtherNonoperatingIncomeExpense",
    ],
}

BALANCE_CONCEPTS: dict[str, list[str]] = {
    "CashAndCashEquivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    ],
    "ShortTermInvestments": [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
    ],
    "AccountsReceivableNet": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
        "ReceivablesNetCurrent",
    ],
    "Inventories": [
        "InventoryNet",
        "InventoryFinishedGoods",
    ],
    "OtherCurrentAssets": [
        "OtherAssetsCurrent",
        "PrepaidExpenseAndOtherAssetsCurrent",
    ],
    "TotalCurrentAssets": [
        "AssetsCurrent",
    ],
    "PropertyPlantAndEquipmentNet": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],
    "Goodwill": [
        "Goodwill",
    ],
    "IntangibleAssetsNet": [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
    ],
    "LongTermInvestments": [
        "LongTermInvestments",
        "MarketableSecuritiesNoncurrent",
        "AvailableForSaleSecuritiesNoncurrent",
    ],
    "OtherNonCurrentAssets": [
        "OtherAssetsNoncurrent",
    ],
    "TotalNonCurrentAssets": [
        "AssetsNoncurrent",
    ],
    "TotalAssets": [
        "Assets",
    ],
    "AccountsPayable": [
        "AccountsPayableCurrent",
        "AccountsPayableAndAccruedLiabilitiesCurrent",
    ],
    "ShortTermDebt": [
        "ShortTermBorrowings",
        "CommercialPaper",
    ],
    "DeferredRevenueCurrent": [
        "DeferredRevenueCurrent",
        "ContractWithCustomerLiabilityCurrent",
    ],
    "OtherCurrentLiabilities": [
        "OtherLiabilitiesCurrent",
        "AccruedLiabilitiesCurrent",
    ],
    "TotalCurrentLiabilities": [
        "LiabilitiesCurrent",
    ],
    "LongTermDebt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "OtherNonCurrentLiabilities": [
        "OtherLiabilitiesNoncurrent",
    ],
    "TotalNonCurrentLiabilities": [
        "LiabilitiesNoncurrent",
    ],
    "TotalLiabilities": [
        "Liabilities",
    ],
    "CommonStockValue": [
        "CommonStocksIncludingAdditionalPaidInCapital",
        "CommonStockValue",
    ],
    "RetainedEarnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
    "AccumulatedOtherComprehensiveIncomeLoss": [
        "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
    ],
    "TotalStockholdersEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "TotalLiabilitiesAndStockholdersEquity": [
        "LiabilitiesAndStockholdersEquity",
    ],
    "CommonSharesOutstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    "TreasuryStock": [
        "TreasuryStockValue",
    ],
    "AdditionalPaidInCapital": [
        "AdditionalPaidInCapitalCommonStock",
        "AdditionalPaidInCapital",
    ],
}

CASHFLOW_CONCEPTS: dict[str, list[str]] = {
    "NetIncomeLoss": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "DepreciationAndAmortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ],
    "StockBasedCompensation": [
        "ShareBasedCompensation",
        "StockBasedCompensation",
    ],
    "DeferredIncomeTax": [
        "DeferredIncomeTaxExpenseBenefit",
        "DeferredIncomeTaxesAndTaxCredits",
    ],
    "ChangeInAccountsReceivable": [
        "IncreaseDecreaseInAccountsReceivable",
    ],
    "ChangeInInventories": [
        "IncreaseDecreaseInInventories",
    ],
    "ChangeInAccountsPayable": [
        "IncreaseDecreaseInAccountsPayable",
        "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",
    ],
    "OtherOperatingActivities": [
        "OtherOperatingActivitiesCashFlowStatement",
        "OtherNoncashIncomeExpense",
    ],
    "NetCashFromOperatingActivities": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "CapitalExpenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "PurchaseOfInvestments": [
        "PaymentsToAcquireInvestments",
        "PaymentsToAcquireAvailableForSaleSecuritiesDebt",
        "PaymentsToAcquireMarketableSecurities",
    ],
    "SaleOfInvestments": [
        "ProceedsFromSaleAndMaturityOfMarketableSecurities",
        "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
        "ProceedsFromSaleOfAvailableForSaleSecuritiesDebt",
    ],
    "Acquisitions": [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesAndInterestInAffiliates",
    ],
    "OtherInvestingActivities": [
        "PaymentsForProceedsFromOtherInvestingActivities",
        "OtherInvestingActivitiesCashFlowStatement",
    ],
    "NetCashFromInvestingActivities": [
        "NetCashProvidedByUsedInInvestingActivities",
    ],
    "DebtRepayment": [
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
    ],
    "DebtIssuance": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromDebtNetOfIssuanceCosts",
    ],
    "CommonStockRepurchased": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
    "DividendsPaid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ],
    "OtherFinancingActivities": [
        "ProceedsFromPaymentsForOtherFinancingActivities",
        "OtherFinancingActivitiesCashFlowStatement",
    ],
    "NetCashFromFinancingActivities": [
        "NetCashProvidedByUsedInFinancingActivities",
    ],
    "NetChangeInCash": [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
        "CashAndCashEquivalentsPeriodIncreaseDecrease",
    ],
    "CashAtEndOfPeriod": [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAtCarryingValue",
    ],
}


def _rate_limited_get(url: str, timeout: int = 30) -> requests.Response:
    """
    Perform an HTTP GET request with rate limiting (max 10 requests/sec)
    and SEC-required User-Agent header.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        requests.Response object.
    """
    global _last_request_time
    with _rate_lock:
        elapsed = time.time() - _last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        _last_request_time = time.time()

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    return response


def get_cik_for_ticker(ticker: str) -> str | None:
    """
    Resolves a stock ticker to its SEC CIK number.

    Uses the SEC company_tickers.json endpoint and caches the full mapping.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").

    Returns:
        CIK number as a zero-padded string, or None if not found.
    """
    ticker_upper = ticker.upper()

    if ticker_upper in _ticker_cik_cache:
        return _ticker_cik_cache[ticker_upper]

    if not _ticker_cik_cache:
        try:
            response = _rate_limited_get(
                "https://www.sec.gov/files/company_tickers.json"
            )
            if response.status_code != 200:
                logger.error(
                    "Failed to fetch SEC ticker-CIK mapping: HTTP %s",
                    response.status_code,
                )
                return None

            data = response.json()
            for entry in data.values():
                t = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                _ticker_cik_cache[t] = cik
        except (requests.RequestException, ValueError) as e:
            logger.error("Error fetching SEC ticker-CIK mapping: %s", e)
            return None

    return _ticker_cik_cache.get(ticker_upper)


def get_company_facts(cik: str) -> dict | None:
    """
    Fetches company facts from SEC EDGAR XBRL API.

    One API call returns all financial data for a company.
    Results are cached per CIK.

    Args:
        cik: Zero-padded CIK number.

    Returns:
        Dict of company facts JSON, or None on error.
    """
    if cik in _company_facts_cache:
        return _company_facts_cache[cik]

    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        response = _rate_limited_get(url)

        if response.status_code != 200:
            logger.error(
                "Failed to fetch EDGAR company facts for CIK %s: HTTP %s",
                cik,
                response.status_code,
            )
            return None

        facts = response.json()
        _company_facts_cache[cik] = facts
        return facts
    except (requests.RequestException, ValueError) as e:
        logger.error("Error fetching EDGAR company facts for CIK %s: %s", cik, e)
        return None


def _extract_financial_items(
    facts: dict,
    concept_mapping: dict[str, list[str]],
    quarter: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Extracts specific financial items from company facts JSON.

    Args:
        facts: Company facts JSON from SEC EDGAR.
        concept_mapping: Dict mapping output field names to ordered lists
                        of XBRL concept names to try.
        quarter: If True, extract quarterly (10-Q) data; otherwise annual (10-K).

    Returns:
        Dict mapping field names to dicts of {period_end_date: value}.
    """
    form_filter = "10-Q" if quarter else "10-K"
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    result: dict[str, dict[str, float]] = {}

    for field_name, concept_names in concept_mapping.items():
        field_data: dict[str, float] = {}

        for concept_name in concept_names:
            if concept_name not in us_gaap:
                continue

            concept_data = us_gaap[concept_name]
            units = concept_data.get("units", {})

            # Determine the right unit key
            unit_key = None
            for key in ["USD", "USD/shares", "shares", "pure"]:
                if key in units:
                    unit_key = key
                    break

            if unit_key is None:
                continue

            entries = units[unit_key]

            # Filter by form type and collect entries
            filing_entries: dict[str, list] = {}
            for entry in entries:
                form = entry.get("form", "")
                if form != form_filter:
                    continue

                end_date = entry.get("end", "")
                if not end_date:
                    continue

                # For annual data, skip entries that look quarterly
                # (duration less than ~350 days)
                start_date = entry.get("start", "")
                if not quarter and start_date and end_date:
                    try:
                        from datetime import datetime

                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        duration = (end_dt - start_dt).days
                        # Annual filings should cover roughly a year
                        if duration < 350:
                            continue
                    except ValueError:
                        pass

                filed_date = entry.get("filed", "")
                val = entry.get("val", 0)

                if end_date not in filing_entries:
                    filing_entries[end_date] = []
                filing_entries[end_date].append((filed_date, val))

            # Deduplicate: prefer most recently filed entry per period
            for end_date, filings in filing_entries.items():
                filings.sort(key=lambda x: x[0], reverse=True)
                field_data[end_date] = filings[0][1]

            if field_data:
                break  # Found data with this concept, stop trying alternatives

        result[field_name] = field_data

    return result


def get_financial_statement(
    ticker: str,
    statement: str,
    quarter: bool = False,
    fallback: bool = False,
) -> pd.DataFrame:
    """
    Retrieves a financial statement from SEC EDGAR.

    Interface matches yfinance_model.get_financial_statement().

    Args:
        ticker: Stock ticker symbol.
        statement: One of 'balance', 'income', or 'cashflow'.
        quarter: If True, retrieves quarterly data.
        fallback: If True, this is being used as a fallback source.

    Returns:
        pd.DataFrame with EDGAR concept names as index, PeriodIndex columns.
        Returns empty DataFrame on error.
    """
    if statement not in ["balance", "income", "cashflow"]:
        raise ValueError(
            "Please choose either 'balance', 'income', or "
            "cashflow' for the statement parameter."
        )

    cik = get_cik_for_ticker(ticker)
    if cik is None:
        error_code = "EDGAR NO CIK FOUND FALLBACK" if fallback else "EDGAR NO CIK FOUND"
        logger.warning("Could not resolve CIK for ticker %s", ticker)
        return pd.DataFrame(columns=[error_code])

    facts = get_company_facts(cik)
    if facts is None:
        error_code = (
            "EDGAR NO DATA FOUND FALLBACK" if fallback else "EDGAR NO DATA FOUND"
        )
        logger.warning("Could not fetch EDGAR data for ticker %s (CIK %s)", ticker, cik)
        return pd.DataFrame(columns=[error_code])

    concept_mapping = {
        "income": INCOME_CONCEPTS,
        "balance": BALANCE_CONCEPTS,
        "cashflow": CASHFLOW_CONCEPTS,
    }[statement]

    items = _extract_financial_items(facts, concept_mapping, quarter=quarter)

    # Build DataFrame
    all_dates: set[str] = set()
    for dates in items.values():
        all_dates.update(dates.keys())

    if not all_dates:
        error_code = (
            "EDGAR NO DATA FOUND FALLBACK" if fallback else "EDGAR NO DATA FOUND"
        )
        return pd.DataFrame(columns=[error_code])

    sorted_dates = sorted(all_dates)

    data = {}
    for field_name, dates_values in items.items():
        data[field_name] = {date: dates_values.get(date, 0) for date in sorted_dates}

    df = pd.DataFrame(data).T
    df.columns = pd.PeriodIndex(df.columns, freq="Q" if quarter else "Y")

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Fill NaN with 0
    if df.isna().to_numpy().any():
        df = df.infer_objects(copy=False).fillna(0)

    return df


def clear_cache():
    """Clear all cached EDGAR data (CIK mapping and company facts)."""
    _ticker_cik_cache.clear()
    _company_facts_cache.clear()
