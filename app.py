import warnings
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
from scipy import stats
import logging
import json
import os
import ssl
from dash.exceptions import PreventUpdate

# Disable SSL warnings explicitly
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to make SSL more permissive for the API request
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Constants
BASE_URL = "https://banks.data.fdic.gov/api"
DEFAULT_START_DATE = '20200630'  # June 30, 2020
DEFAULT_END_DATE = '20250930'    # Sept 30, 2025
CACHE_DIR = 'data_cache'

os.makedirs(CACHE_DIR, exist_ok=True)

# Color scheme - Grasshopper Bank colors (Phthalo Green primary)
COLOR_SCHEME = {
    'primary': '#0E3E1B',
    'secondary': '#333333',
    'accent': '#2D5F3F',
    'background': '#f5f5f5',
    'card_bg': '#ffffff',
    'highlight': '#0E3E1B',
    'text': '#333333',
    'light_text': '#666666',
    'grid': 'rgba(0, 0, 0, 0.1)',
    'grasshopper': '#0E3E1B',
    'peer': '#808080',
    'peer_opacity': 0.4,
    'good': '#2D5F3F',
    'warning': '#FF9800',
    'danger': '#F44336',
}

BANK_NAME_MAPPING = {
    "GRASSHOPPER BANK, N.A.": "Grasshopper Bank",
    "LIVE OAK BANKING COMPANY": "Live Oak Bank",
    "CELTIC BANK CORPORATION": "Celtic Bank",
    "COASTAL COMMUNITY BANK": "Coastal Community Bank",
    "CHOICE FINANCIAL GROUP": "Choice Financial Group",
    "METROPOLITAN COMMERCIAL BANK": "Metropolitan Commercial Bank",
    "CROSS RIVER BANK": "Cross River Bank",
    "AXOS BANK": "Axos Bank",
}

BANK_INFO = [
    {"cert": "59113", "name": "GRASSHOPPER BANK, N.A."},
    {"cert": "58665", "name": "LIVE OAK BANKING COMPANY"},
    {"cert": "57056", "name": "CELTIC BANK CORPORATION"},
    {"cert": "34403", "name": "COASTAL COMMUNITY BANK"},
    {"cert": "9423", "name": "CHOICE FINANCIAL GROUP"},
    {"cert": "34699", "name": "METROPOLITAN COMMERCIAL BANK"},
    {"cert": "58410", "name": "CROSS RIVER BANK"},
    {"cert": "35546", "name": "AXOS BANK"},
]

PRIMARY_BANK_DISPLAY_NAME = "Grasshopper Bank"


def normalize_bank_name(bank_name: str) -> str:
    if not bank_name:
        return bank_name
    if bank_name in BANK_NAME_MAPPING:
        return BANK_NAME_MAPPING[bank_name]
    bank_name_upper = bank_name.upper().strip()
    for official_name, display_name in BANK_NAME_MAPPING.items():
        if official_name.upper().strip() == bank_name_upper:
            return display_name
    logger.warning(f"No mapping found for bank name: '{bank_name}'")
    return bank_name


class FDICAPIClient:
    def __init__(self):
        self.base_url = BASE_URL

    def get_data(self, endpoint: str, params: Dict) -> Dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                verify=False,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error for {url}: {e}")
            return {"data": []}

    def get_institutions(self, filters: str = "", fields: str = "") -> List[Dict]:
        params = {"filters": filters, "fields": fields, "limit": 10000}
        data = self.get_data("institutions", params)
        return data.get('data', [])

    def get_financials(self, cert: str, filters: str = "", fields: str = "") -> List[Dict]:
        params = {
            "filters": f"CERT:{cert}" + (f" AND {filters}" if filters else ""),
            "fields": fields,
            "limit": 10000
        }
        data = self.get_data("financials", params)
        return data.get('data', [])


class BankDataRepository:
    def __init__(self):
        self.api_client = FDICAPIClient()
        self.dollar_format_metrics = [
            'Total Assets',
            'Total Deposits',
            'Total Loans and Leases',
            'Net Loans and Leases',
            'Total Securities',
            'Real Estate Loans',
            'Loans to Residential Properties',
            'Multifamily',
            'Farmland Real Estate Loans',
            'Loans to Nonresidential Properties',
            'Owner-Occupied Nonresidential Properties Loans',
            'Non-OOC Nonresidential Properties Loans',
            'RE Construction and Land Development',
            '1-4 Family Residential Construction and Land Development Loans',
            'Other Construction, All Land Development and Other Land Loans',
            'Commercial Real Estate Loans not Secured by Real Estate',
            'Commercial and Industrial Loans',
            'Agriculture Loans',
            'Auto Loans',
            'Credit Cards',
            'Consumer Loans',
            'Allowance for Credit Loss',
            'Past Due 30-89 Days',
            'Past Due 90+ Days',
            'Tier 1 (Core) Capital',
            'Total Charge-Offs',
            'Total Recoveries',
            'Net Income',
            'Total Loans and Leases Net Charge-Offs Quarterly',
            'Common Equity Tier 1 Before Adjustments',
            'Bank Equity Capital',
            'CECL Transition Amount',
            'Perpetual Preferred Stock'
        ]
        os.makedirs(CACHE_DIR, exist_ok=True)

    def get_cache_path(self, start_date: str, end_date: str) -> str:
        return os.path.join(CACHE_DIR, f"bank_data_{start_date}_{end_date}.json")

    def load_cached_data(self, start_date: str, end_date: str) -> Optional[Dict]:
        cache_path = self.get_cache_path(start_date, end_date)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load cache file {cache_path}: {e}")
        return None

    def save_to_cache(self, data: Dict, start_date: str, end_date: str) -> None:
        cache_path = self.get_cache_path(start_date, end_date)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            logger.error(f"Failed to save to cache file {cache_path}: {e}")

    def fetch_data(self, bank_info: List[Union[str, Dict]], start_date: str, end_date: str) -> Dict[str, Dict]:
        cached_data = self.load_cached_data(start_date, end_date)
        if cached_data:
            logger.info(f"Using cached data for {start_date} to {end_date}")
            return cached_data

        logger.info(f"Fetching fresh data from FDIC API for {start_date} to {end_date}")

        institution_fields = "NAME,CERT"
        financial_fields = (
            "CERT,REPDTE,ASSET,DEP,LNLSGR,LNLSNET,SC,LNRE,LNCI,LNAG,LNCRCD,LNCONOTH,LNATRES,"
            "P3ASSET,P9ASSET,RBCT1J,DRLNLS,CRLNLS,NETINC,ERNASTR,NPERFV,P3ASSETR,P9ASSETR,NIMY,"
            "NTLNLSR,LNATRESR,NCLNLSR,ROA,ROE,RBC1AAJ,RBCT2,RBCRWAJ,LNLSDEPR,LNLSNTV,EEFFR,"
            "LNRESNCR,ELNANTR,IDERNCVR,NTLNLSQ,LNRECONS,LNRENRES,LNRENROW,LNRENROT,LNRERES,LNREMULT,"
            "LNREAG,LNRECNFM,LNRECNOT,LNCOMRE,CT1BADJ,EQ,EQPP,LNAUTO"
        )

        institutions_data = {}
        financials_data = {}

        api_failures = 0
        max_retries = 3

        for retry_count in range(max_retries):
            try:
                if retry_count > 0:
                    institutions_data = {}
                    financials_data = {}
                    logger.info(f"Retry attempt {retry_count} for FDIC API")

                for bank_item in bank_info:
                    try:
                        if isinstance(bank_item, str):
                            institutions = self.api_client.get_institutions(f'NAME:"{bank_item}"', institution_fields)
                        elif isinstance(bank_item, dict) and 'cert' in bank_item:
                            institutions = self.api_client.get_institutions(f'CERT:{bank_item["cert"]}', institution_fields)
                        else:
                            logger.warning(f"Invalid bank info format: {bank_item}")
                            continue

                        if not institutions:
                            logger.warning(f"No data found for bank: {bank_item}")
                            continue

                        bank = institutions[0]
                        if isinstance(bank, dict) and 'data' in bank:
                            bank_data = bank['data']
                            if 'NAME' in bank_data and 'CERT' in bank_data:
                                institutions_data[bank_data['NAME']] = bank_data
                                financials = self.api_client.get_financials(
                                    bank_data['CERT'],
                                    f"REPDTE:[{start_date} TO {end_date}]",
                                    fields=financial_fields
                                )
                                financials_data[bank_data['NAME']] = [
                                    f['data'] for f in financials if isinstance(f, dict) and 'data' in f
                                ]
                                logger.info(f"Fetched {len(financials)} records for {bank_data['NAME']}")
                            else:
                                logger.warning(f"Required fields missing for bank: {bank_item}")
                        else:
                            logger.warning(f"Unexpected data structure for bank: {bank_item}")
                    except Exception as e:
                        logger.error(f"Error fetching data for bank {bank_item}: {e}")
                        api_failures += 1

                if institutions_data:
                    break

            except Exception as e:
                logger.error(f"Error in API batch fetch (attempt {retry_count+1}): {e}")
                if retry_count == max_retries - 1 and not institutions_data:
                    logger.warning("All API retries failed. Using fallback data.")
                    return self._generate_fallback_data(start_date, end_date)

        if api_failures > len(bank_info) // 2 or not institutions_data:
            logger.warning(f"Too many API failures ({api_failures}). Using fallback data.")
            return self._generate_fallback_data(start_date, end_date)

        result = {'institutions_data': institutions_data, 'financials_data': financials_data}
        self.save_to_cache(result, start_date, end_date)
        return result

    def _generate_fallback_data(self, start_date: str, end_date: str) -> Dict[str, Dict]:
        logger.info("Generating fallback data")

        institutions_data = {}
        financials_data = {}

        for bank_info in BANK_INFO:
            bank_name = bank_info["name"]
            cert = bank_info["cert"]

            institutions_data[bank_name] = {"NAME": bank_name, "CERT": cert}

            financial_records = []
            start = pd.to_datetime(start_date, format="%Y%m%d")
            end = pd.to_datetime(end_date, format="%Y%m%d")
            quarters = pd.date_range(start=start, end=end, freq='Q')

            normalized_name = normalize_bank_name(bank_name)
            is_grasshopper = normalized_name == PRIMARY_BANK_DISPLAY_NAME
            base_assets = 2_000_000_000 if is_grasshopper else np.random.uniform(1_500_000_000, 5_000_000_000)

            for date in quarters:
                date_str = date.strftime("%Y%m%d")
                years_since_start = (date - start).days / 365.25
                growth_factor = 1 + years_since_start * 0.08
                random_factor = np.random.uniform(0.95, 1.05)

                assets = base_assets * growth_factor * random_factor
                deposits = assets * 0.75 * np.random.uniform(0.9, 1.1)
                loans = assets * 0.70 * np.random.uniform(0.9, 1.1)
                tier1_capital = assets * 0.12 * np.random.uniform(0.9, 1.1)

                if is_grasshopper:
                    deposits = assets * 0.70 * np.random.uniform(0.9, 1.1)
                    loans = assets * 0.72 * np.random.uniform(0.9, 1.1)
                    tier1_capital = assets * 0.13 * np.random.uniform(0.9, 1.1)

                record = {
                    "REPDTE": date_str,
                    "ASSET": assets,
                    "DEP": deposits,
                    "LNLSGR": loans,
                    "LNLSNET": loans * 0.98,
                    "RBCT1J": tier1_capital,
                    "ROA": np.random.uniform(0.8, 1.5),
                    "ROE": np.random.uniform(8, 14),
                    "SC": assets * 0.15,
                    "LNRE": loans * 0.45,
                    "LNCI": loans * 0.35,
                    "LNAG": loans * 0.01,
                    "LNAUTO": loans * 0.08,
                    "LNCRCD": loans * 0.02,
                    "LNCONOTH": loans * 0.09,
                    "LNATRES": loans * 0.015,
                    "NIMY": np.random.uniform(2.5, 3.8),
                    "EEFFR": np.random.uniform(55, 65),
                }
                financial_records.append(record)

            financials_data[bank_name] = financial_records

        result = {'institutions_data': institutions_data, 'financials_data': financials_data}
        self.save_to_cache(result, start_date, end_date)
        return result


class BankMetricsCalculator:
    """
    Calculator for bank metrics based on FDIC data.
    """
    def __init__(self, dollar_format_metrics: List[str]):
        self.dollar_format_metrics = dollar_format_metrics
        self.metric_definitions = self._get_metric_definitions()

    def _get_metric_definitions(self) -> Dict[str, str]:
        return {
            'Total Assets': "(YTD, $) The sum of all assets owned by the entity.",
            'Total Deposits': "(YTD, $) The sum of all deposits including demand, savings, and time deposits.",
            'Total Loans and Leases': "(YTD, $) Total loans and lease financing receivables.",
            'Net Loans and Leases': "(YTD, $) Net Loans and Leases",
            'Total Securities': "(YTD, $) Sum of held-to-maturity, available-for-sale, and equity securities.",
            'Real Estate Loans': "(YTD, $) Loans primarily secured by real estate.",
            'Loans to Residential Properties': "(YTD, $) Total loans for residential properties.",
            'Multifamily': "(YTD, $) Loans for multifamily residential properties.",
            'Farmland Real Estate Loans': "(YTD, $) Loans secured by farmland.",
            '1-4 Family Residential Construction and Land Development Loans': "(YTD, $) Construction and land development loans for 1-4 family residential properties.",
            'Other Construction, All Land Development and Other Land Loans': "(YTD, $) Other construction loans, all land development and other land loans.",
            'Loans to Nonresidential Properties': "(YTD, $) Total loans for nonresidential properties.",
            'Owner-Occupied Nonresidential Properties Loans': "(YTD, $) Loans for owner-occupied nonresidential properties.",
            'Non-OOC Nonresidential Properties Loans': "(YTD, $) Loans for non-owner-occupied nonresidential properties.",
            'Commercial Real Estate Loans not Secured by Real Estate': "(YTD, $) Commercial real estate loans that are not secured by real estate.",
            'Commercial and Industrial Loans': "(YTD, $) Loans for commercial and industrial purposes, excluding real estate-secured loans.",
            'Agriculture Loans': "(YTD, $) Loans to finance agricultural production and other loans to farmers.",
            'Auto Loans': "(YTD, $) Consumer loans for automobile purchases.",
            'Credit Cards': "(YTD, $) Consumer loans extended through credit card plans.",
            'Consumer Loans': "(YTD, $) Other loans to individuals for personal expenditures, including student loans.",
            'Allowance for Credit Loss': "(YTD, $) Reserve for estimated credit losses associated with the loan and lease portfolio.",
            'Past Due 30-89 Days': "(Qtly, $) Loans and leases past due 30-89 days, in dollars.",
            'Past Due 90+ Days': "(Qtly, $) Loans and leases past due 90 days or more, in dollars.",
            'Tier 1 (Core) Capital': "(Qtly, $) Tier 1 core capital.",
            'Total Charge-Offs': "(YTD, $) Total charge-offs of loans and leases.",
            'Total Recoveries': "(YTD, $) Total recoveries of loans and leases previously charged off.",
            'Total Loans and Leases Net Charge-Offs Quarterly': "(Qtly, $) Total loans and leases net charge-offs for the quarter.",
            'Net Income': "(YTD, $) Net income earned by the entity.",
            'RE Construction and Land Development': "(YTD, $) Real estate construction and land development loans.",
            'RE Construction and Land Development to Tier 1 + ACL': "(Qtly, %) RE construction & land dev as % of Tier 1 + ACL.",
            'Common Equity Tier 1 Before Adjustments': "(YTD, $) Common Equity Tier 1 capital before adjustments.",
            'Bank Equity Capital': "(YTD, $) Total bank equity capital.",
            'Perpetual Preferred Stock': "(YTD, $) The amount of perpetual preferred stock.",
            'CECL Transition Amount': "(YTD, $) CECL transition amount (derived).",
            'Net Interest Margin': "(YTD, %) The net interest margin.",
            'Earning Assets / Total Assets': "(Qtly, %) Earning assets to total assets.",
            'Nonperforming Assets / Total Assets': "(Qtly, %) Nonperforming assets to total assets.",
            'Assets Past Due 30-89 Days / Total Assets': "(Qtly, %) Past due 30-89 to total assets.",
            'Assets Past Due 90+ Days / Total Assets': "(Qtly, %) Past due 90+ to total assets.",
            'Net Charge-Offs / Total Loans & Leases': "(YTD, %) Net charge-offs to total loans & leases.",
            'Earnings Coverage of Net Loan Charge-Offs': "(X) Earnings coverage of net loan charge-offs.",
            'Loan and Lease Loss Provision to Net Charge-Offs': "(YTD, %) Provision to net charge-offs.",
            'Loss Allowance / Total Loans & Leases': "(YTD, %) Allowance to total loans & leases.",
            'Loss Allowance to Noncurrent Loans and Leases': "(Qtly, %) Allowance to noncurrent loans.",
            'Noncurrent Loans / Total Loans': "(Qtly, %) Noncurrent loans to total loans.",
            'Net Loans and Leases to Deposits': "(YTD, %) Net loans and leases to deposits.",
            'Net Loans and Leases to Assets': "(Qtly, %) Net loans and leases to assets.",
            'Return on Assets': "(YTD, %) Return on assets.",
            'Return on Equity': "(YTD, %) Return on equity.",
            'Leverage (Core Capital) Ratio': "(Qtly, %) Leverage ratio.",
            'Total Risk-Based Capital Ratio': "(Qtly, %) Total risk-based capital ratio.",
            'Efficiency Ratio': "(YTD, %) Efficiency ratio.",
            'Real Estate Loans to Tier 1 + ACL': "(Qtly, %) Real estate loans as % of Tier 1 + ACL.",
            'Commercial RE to Tier 1 + ACL': "(Qtly, %) Commercial RE as % of Tier 1 + ACL.",
            'Non-Owner Occupied CRE 3-Year Growth Rate': "(%) 3-year annualized growth rate of Non-OOC CRE.",
            'C&I Loans to Tier 1 + ACL': "(Qtly, %) C&I loans as % of Tier 1 + ACL.",
            'Auto Loans to Tier 1 + ACL': "(Qtly, %) Auto loans as % of Tier 1 + ACL.",
            'Net Charge-Offs / Allowance for Credit Loss': "(Qtly, %) Net charge-offs to ACL.",
        }

    @staticmethod
    def safe_float(value: Any) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def calculate_metrics(self, financials_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        all_metrics = []
        for bank_name, financials in financials_data.items():
            sorted_financials = sorted(financials, key=lambda x: x['REPDTE'])
            for i, financial in enumerate(sorted_financials):
                metrics = self._extract_basic_metrics(bank_name, financial)

                cecl_transition_amount, capital_base = self._calculate_capital_base(metrics, financial)
                metrics['CECL Transition Amount'] = cecl_transition_amount

                self._calculate_capital_ratios(metrics, capital_base)
                self._calculate_cre_growth_rate(metrics, sorted_financials, i, financial)
                self._calculate_charge_off_metrics(metrics)

                all_metrics.append(metrics)

        df = pd.DataFrame(all_metrics)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        return df.sort_values('Date')

    def _extract_basic_metrics(self, bank_name: str, financial: Dict) -> Dict:
        return {
            'Bank': bank_name,
            'Date': financial.get('REPDTE'),
            'Total Assets': self.safe_float(financial.get('ASSET')),
            'Total Deposits': self.safe_float(financial.get('DEP')),
            'Total Loans and Leases': self.safe_float(financial.get('LNLSGR')),
            'Net Loans and Leases': self.safe_float(financial.get('LNLSNET')),
            'Total Securities': self.safe_float(financial.get('SC')),
            'Real Estate Loans': self.safe_float(financial.get('LNRE')),
            'Loans to Residential Properties': self.safe_float(financial.get('LNRERES')),
            'Multifamily': self.safe_float(financial.get('LNREMULT')),
            'Farmland Real Estate Loans': self.safe_float(financial.get('LNREAG')),
            'Loans to Nonresidential Properties': self.safe_float(financial.get('LNRENRES')),
            'Owner-Occupied Nonresidential Properties Loans': self.safe_float(financial.get('LNRENROW')),
            'Non-OOC Nonresidential Properties Loans': self.safe_float(financial.get('LNRENROT')),
            'RE Construction and Land Development': self.safe_float(financial.get('LNRECONS')),
            '1-4 Family Residential Construction and Land Development Loans': self.safe_float(financial.get('LNRECNFM')),
            'Other Construction, All Land Development and Other Land Loans': self.safe_float(financial.get('LNRECNOT')),
            'Commercial Real Estate Loans not Secured by Real Estate': self.safe_float(financial.get('LNCOMRE')),
            'Commercial and Industrial Loans': self.safe_float(financial.get('LNCI')),
            'Agriculture Loans': self.safe_float(financial.get('LNAG')),
            'Auto Loans': self.safe_float(financial.get('LNAUTO')),
            'Credit Cards': self.safe_float(financial.get('LNCRCD')),
            'Consumer Loans': self.safe_float(financial.get('LNCONOTH')),
            'Allowance for Credit Loss': self.safe_float(financial.get('LNATRES')),
            'Past Due 30-89 Days': self.safe_float(financial.get('P3ASSET')),
            'Past Due 90+ Days': self.safe_float(financial.get('P9ASSET')),
            'Tier 1 (Core) Capital': self.safe_float(financial.get('RBCT1J')),
            'Total Charge-Offs': self.safe_float(financial.get('DRLNLS')),
            'Total Recoveries': self.safe_float(financial.get('CRLNLS')),
            'Total Loans and Leases Net Charge-Offs Quarterly': self.safe_float(financial.get('NTLNLSQ')),
            'Net Income': self.safe_float(financial.get('NETINC')),
            'Common Equity Tier 1 Before Adjustments': self.safe_float(financial.get('CT1BADJ')),
            'Bank Equity Capital': self.safe_float(financial.get('EQ')),
            'Perpetual Preferred Stock': self.safe_float(financial.get('EQPP')),
            'Net Interest Margin': self.safe_float(financial.get('NIMY')),
            'Earning Assets / Total Assets': self.safe_float(financial.get('ERNASTR')),
            'Nonperforming Assets / Total Assets': self.safe_float(financial.get('NPERFV')),
            'Assets Past Due 30-89 Days / Total Assets': self.safe_float(financial.get('P3ASSETR')),
            'Assets Past Due 90+ Days / Total Assets': self.safe_float(financial.get('P9ASSETR')),
            'Net Charge-Offs / Total Loans & Leases': self.safe_float(financial.get('NTLNLSR')),
            'Earnings Coverage of Net Loan Charge-Offs': self.safe_float(financial.get('IDERNCVR')),
            'Loan and Lease Loss Provision to Net Charge-Offs': self.safe_float(financial.get('ELNANTR')),
            'Loss Allowance / Total Loans & Leases': self.safe_float(financial.get('LNATRESR')),
            'Loss Allowance to Noncurrent Loans and Leases': self.safe_float(financial.get('LNRESNCR')),
            'Noncurrent Loans / Total Loans': self.safe_float(financial.get('NCLNLSR')),
            'Net Loans and Leases to Deposits': self.safe_float(financial.get('LNLSDEPR')),
            'Net Loans and Leases to Assets': self.safe_float(financial.get('LNLSNTV')),
            'Return on Assets': self.safe_float(financial.get('ROA')),
            'Return on Equity': self.safe_float(financial.get('ROE')),
            'Leverage (Core Capital) Ratio': self.safe_float(financial.get('RBC1AAJ')),
            'Total Risk-Based Capital Ratio': self.safe_float(financial.get('RBCRWAJ')),
            'Efficiency Ratio': self.safe_float(financial.get('EEFFR')),
        }

    # =========================
    # UPDATED: CECL + capital base logic
    # =========================
    def _calculate_capital_base(self, metrics: Dict, financial: Dict) -> Tuple[float, float]:
        """
        Updated philosophy:
        - Allow CECL to reduce capital base.
        - Never "turn CECL off" just because it makes the denominator small/negative.
        - Instead, compute raw_capital_base and apply a FLOOR to keep ratios computable.
        - Still protect against absurdly large denominators (mostly negative CECL explosions).
        """
        ct1badj = metrics['Common Equity Tier 1 Before Adjustments']
        eq = metrics['Bank Equity Capital']
        eqpp = metrics['Perpetual Preferred Stock']
        tier1_capital = metrics['Tier 1 (Core) Capital']
        allowance_for_credit_loss = metrics['Allowance for Credit Loss']

        date = pd.to_datetime(financial.get('REPDTE'), format='%Y%m%d')

        base_tier1_plus_acl = tier1_capital + allowance_for_credit_loss

        cecl_transition_amount = 0.0
        if date >= pd.to_datetime('2019-01-01'):
            cecl_transition_amount = ct1badj - eq + eqpp

        raw_capital_base = base_tier1_plus_acl - cecl_transition_amount

        # If CECL inputs are likely missing (common with API), don’t pretend it’s meaningful
        if date >= pd.to_datetime('2019-01-01'):
            if ct1badj == 0 and eq == 0 and eqpp == 0:
                cecl_transition_amount = 0.0
                raw_capital_base = base_tier1_plus_acl

        # Cap absurdly large denominators (protects negative CECL making capital_base huge)
        max_reasonable = max(tier1_capital * 100, 1.0)
        if raw_capital_base > max_reasonable:
            logger.warning(
                f"Capital base unusually high ({raw_capital_base:,.0f}) for {metrics.get('Bank', 'Unknown')} "
                f"on {date.date()} (CECL={cecl_transition_amount:,.0f}). Capping to {max_reasonable:,.0f}."
            )
            raw_capital_base = max_reasonable

        # FLOOR: keep CECL effect but prevent divide-by-zero / negative denominators
        floor_abs = 1.0
        floor_pct_tier1 = 0.01  # 1% of Tier1 floor (tune as desired)
        floor_value = max(floor_abs, tier1_capital * floor_pct_tier1)

        if raw_capital_base <= 0:
            logger.warning(
                f"CECL reduced capital base <= 0 for {metrics.get('Bank', 'Unknown')} on {date.date()} "
                f"(Tier1+ACL={base_tier1_plus_acl:,.0f}, CECL={cecl_transition_amount:,.0f}, raw={raw_capital_base:,.0f}). "
                f"Applying floor={floor_value:,.0f} while keeping CECL."
            )

        capital_base = max(raw_capital_base, floor_value)

        return float(cecl_transition_amount), float(capital_base)

    # =========================
    # UPDATED: capital ratios logic (allow small denominators)
    # =========================
    def _calculate_capital_ratios(self, metrics: Dict, capital_base: float) -> None:
        """
        Updated:
        - Do NOT require capital_base > 100 (that can block CECL-driven reductions).
        - Only require capital_base > 0 (floored already).
        - Warn if capital_base is very small (ratios may be extreme).
        """
        if capital_base is None or capital_base <= 0:
            logger.warning(
                f"Capital base invalid ({capital_base}) for {metrics.get('Bank', 'Unknown')} on {metrics.get('Date', 'Unknown')}"
            )
            metrics['Real Estate Loans to Tier 1 + ACL'] = None
            metrics['RE Construction and Land Development to Tier 1 + ACL'] = None
            metrics['C&I Loans to Tier 1 + ACL'] = None
            metrics['Auto Loans to Tier 1 + ACL'] = None
            metrics['Commercial RE to Tier 1 + ACL'] = None
            return

        if capital_base < 100:
            logger.warning(
                f"Capital base is very small ({capital_base:,.2f}) for {metrics.get('Bank', 'Unknown')} "
                f"on {metrics.get('Date', 'Unknown')}. Ratios may be extreme."
            )

        metrics['Real Estate Loans to Tier 1 + ACL'] = (metrics['Real Estate Loans'] / capital_base) * 100
        metrics['RE Construction and Land Development to Tier 1 + ACL'] = (metrics['RE Construction and Land Development'] / capital_base) * 100
        metrics['C&I Loans to Tier 1 + ACL'] = (metrics['Commercial and Industrial Loans'] / capital_base) * 100
        metrics['Auto Loans to Tier 1 + ACL'] = (metrics['Auto Loans'] / capital_base) * 100

        commercial_re = (
            metrics['RE Construction and Land Development'] +
            metrics['Multifamily'] +
            metrics['Loans to Nonresidential Properties'] +
            metrics['Commercial Real Estate Loans not Secured by Real Estate']
        )
        metrics['Commercial RE to Tier 1 + ACL'] = (commercial_re / capital_base) * 100

    def _calculate_cre_growth_rate(self, metrics: Dict, sorted_financials: List[Dict], i: int, current_financial: Dict) -> None:
        non_owner_occupied_cre = (
            self.safe_float(current_financial.get('LNRECONS')) +
            self.safe_float(current_financial.get('LNREMULT')) +
            self.safe_float(current_financial.get('LNRENROT')) +
            self.safe_float(current_financial.get('LNCOMRE'))
        )

        if i >= 12:
            three_years_ago = sorted_financials[i - 12]
            old_non_owner_occupied_cre = (
                self.safe_float(three_years_ago.get('LNRECONS')) +
                self.safe_float(three_years_ago.get('LNREMULT')) +
                self.safe_float(three_years_ago.get('LNRENROT')) +
                self.safe_float(three_years_ago.get('LNCOMRE'))
            )
            if old_non_owner_occupied_cre > 1000:
                growth_rate = (non_owner_occupied_cre / old_non_owner_occupied_cre) - 1
                metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = growth_rate * 100
            else:
                metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = None
        else:
            metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = None

    # =========================
    # UPDATED: charge-off metric (avoid divide by tiny ACL)
    # =========================
    def _calculate_charge_off_metrics(self, metrics: Dict) -> None:
        acl = metrics.get('Allowance for Credit Loss', 0.0)
        acl_floor = 1.0  # protect divide-by-near-zero

        if acl is None or acl <= acl_floor:
            metrics['Net Charge-Offs / Allowance for Credit Loss'] = None
            return

        metrics['Net Charge-Offs / Allowance for Credit Loss'] = (
            metrics['Total Loans and Leases Net Charge-Offs Quarterly'] / acl
        ) * 100


class BankDataService:
    def __init__(self):
        self.repository = BankDataRepository()
        self.calculator = BankMetricsCalculator(self.repository.dollar_format_metrics)

    def get_metrics_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:

        data = self.repository.fetch_data(BANK_INFO, start_date, end_date)

        if not data['institutions_data']:
            logger.error("No institution data was fetched.")
            return pd.DataFrame(), self.repository.dollar_format_metrics, self.calculator.metric_definitions

        metrics_df = self.calculator.calculate_metrics(data['financials_data'])

        logger.info("Applying bank name normalization...")
        metrics_df['Bank'] = metrics_df['Bank'].apply(normalize_bank_name)

        unique_banks_after = metrics_df['Bank'].unique()
        logger.info(f"Banks after normalization: {sorted(unique_banks_after)}")

        metric_order = [
            'Bank', 'Date',
            'Real Estate Loans to Tier 1 + ACL',
            'RE Construction and Land Development to Tier 1 + ACL',
            'Commercial RE to Tier 1 + ACL',
            'Non-Owner Occupied CRE 3-Year Growth Rate',
            'C&I Loans to Tier 1 + ACL',
            'Auto Loans to Tier 1 + ACL',
            'Net Charge-Offs / Allowance for Credit Loss',
            'Net Charge-Offs / Total Loans & Leases',
            'Earnings Coverage of Net Loan Charge-Offs',
            'Loan and Lease Loss Provision to Net Charge-Offs',
            'Loss Allowance / Total Loans & Leases',
            'Loss Allowance to Noncurrent Loans and Leases',
            'Nonperforming Assets / Total Assets',
            'Assets Past Due 30-89 Days / Total Assets',
            'Assets Past Due 90+ Days / Total Assets',
            'Noncurrent Loans / Total Loans',
            'Net Loans and Leases to Deposits',
            'Net Loans and Leases to Assets',
            'Return on Assets',
            'Return on Equity',
            'Leverage (Core Capital) Ratio',
            'Total Risk-Based Capital Ratio',
            'Efficiency Ratio',
            'Earning Assets / Total Assets',
            'Net Interest Margin'
        ] + self.repository.dollar_format_metrics

        available_columns = [col for col in metric_order if col in metrics_df.columns]
        extra_columns = [col for col in metrics_df.columns if col not in available_columns]
        metrics_df = metrics_df[available_columns + extra_columns]

        return metrics_df, self.repository.dollar_format_metrics, self.calculator.metric_definitions


class DashboardBuilder:
    def __init__(self, df: pd.DataFrame, dollar_format_metrics: List[str], metric_definitions: Dict[str, str]):
        self.df = df
        self.dollar_format_metrics = dollar_format_metrics
        self.metric_definitions = metric_definitions

        unique_banks = sorted(self.df['Bank'].unique())
        logger.info(f"Found {len(unique_banks)} unique banks in data: {', '.join(unique_banks)}")

        self.unique_dates = sorted(df['Date'].unique())

        self.metric_order = [
            'Real Estate Loans to Tier 1 + ACL',
            'RE Construction and Land Development to Tier 1 + ACL',
            'Commercial RE to Tier 1 + ACL',
            'Non-Owner Occupied CRE 3-Year Growth Rate',
            'C&I Loans to Tier 1 + ACL',
            'Auto Loans to Tier 1 + ACL',
            'Net Charge-Offs / Allowance for Credit Loss',
            'Net Charge-Offs / Total Loans & Leases',
            'Earnings Coverage of Net Loan Charge-Offs',
            'Loan and Lease Loss Provision to Net Charge-Offs',
            'Loss Allowance / Total Loans & Leases',
            'Loss Allowance to Noncurrent Loans and Leases',
            'Nonperforming Assets / Total Assets',
            'Assets Past Due 30-89 Days / Total Assets',
            'Assets Past Due 90+ Days / Total Assets',
            'Noncurrent Loans / Total Loans',
            'Net Loans and Leases to Deposits',
            'Net Loans and Leases to Assets',
            'Return on Assets',
            'Return on Equity',
            'Leverage (Core Capital) Ratio',
            'Total Risk-Based Capital Ratio',
            'Efficiency Ratio',
            'Earning Assets / Total Assets',
            'Net Interest Margin'
        ] + dollar_format_metrics

        self.available_metrics = [metric for metric in self.metric_order if metric in df.columns]

    def create_dashboard(self) -> dash.Dash:
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        app.title = "Grasshopper Bank Metrics Dashboard"
        app.config.suppress_callback_exceptions = True

        server = app.server
        app.index_string = f'''
        <!DOCTYPE html>
        <html>
            <head>
                {{%metas%}}
                <title>{{%title%}}</title>
                {{%favicon%}}
                {{%css%}}
                <style>{self._get_custom_css()}</style>
            </head>
            <body>
                {{%app_entry%}}
                <footer>
                    {{%config%}}
                    {{%scripts%}}
                    {{%renderer%}}
                </footer>
            </body>
        </html>
        '''

        app.layout = self._create_layout()
        self._register_callbacks(app)
        return app

    def _create_layout(self) -> html.Div:
        sidebar = self._create_sidebar()
        content = self._create_content()
        return html.Div([
            html.Div([sidebar, content], id="app-container"),
            dcc.Store(id='selected-bank-store'),
            dcc.Store(id='selected-metric-store', data=self.available_metrics[0] if self.available_metrics else None),
        ])

    def _create_sidebar(self) -> html.Div:
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Grasshopper Bank", className="display-6 grasshopper-title", style={"color": COLOR_SCHEME['primary']}),
                    html.H5("Bank Metrics Dashboard", className="subtitle", style={"color": COLOR_SCHEME['accent']}),
                ], className="sidebar-title"),
                html.Hr(style={"borderColor": COLOR_SCHEME['primary']}),
            ], className="sidebar-header"),

            html.Div([
                html.P("Select a metric to display", className="lead", style={"color": COLOR_SCHEME['text']}),
                dcc.Dropdown(
                    id='metric-selector',
                    options=[{'label': col, 'value': col, 'title': self.metric_definitions.get(col, '')}
                             for col in self.available_metrics],
                    value=self.available_metrics[0] if self.available_metrics else None,
                    clearable=False,
                    style={'width': '100%', 'color': COLOR_SCHEME['text']},
                    optionHeight=55
                ),
                html.Div(id='metric-definition', className="metric-definition mt-3"),
            ], className="sidebar-section"),

            html.Div([
                html.Hr(style={"borderColor": COLOR_SCHEME['primary']}),
                html.P("Select peer banks to compare", className="lead", style={"color": COLOR_SCHEME['text']}),
                html.Div(id='peer-selector', className="mt-3"),
                html.Button("Add All Peers", id="add-all-peers-btn", className="add-all-btn mt-2"),
            ], className="sidebar-section"),

            html.Div([
                html.Hr(style={"borderColor": COLOR_SCHEME['primary']}),
                html.P("Select trend timeline", className="lead", style={"color": COLOR_SCHEME['text']}),
                dcc.Dropdown(
                    id='trend-timeline-selector',
                    options=[
                        {'label': '1 Year', 'value': 1},
                        {'label': '2 Years', 'value': 2},
                        {'label': '3 Years', 'value': 3},
                        {'label': '4 Years', 'value': 4},
                        {'label': '5 Years', 'value': 5},
                        {'label': '6 Years', 'value': 6},
                    ],
                    value=4,
                    clearable=False,
                    style={'width': '100%', 'color': COLOR_SCHEME['text']},
                ),
                html.Div(id='selected-peers-info', className="mt-3", style={"color": COLOR_SCHEME['text']})
            ], className="sidebar-section"),

            html.Div([
                html.Hr(style={"borderColor": COLOR_SCHEME['primary']}),
                html.P("© 2025 Grasshopper Bank", className="text-center", style={"color": COLOR_SCHEME['light_text']}),
            ], className="sidebar-footer"),
        ], className="sidebar")

    def _create_content(self) -> html.Div:
        return html.Div([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Span("Date: ", className="date-label"),
                        dcc.Dropdown(
                            id='date-selector',
                            options=[{'label': date.strftime('%m/%d/%y'), 'value': date.strftime('%Y-%m-%d')}
                                     for date in self.unique_dates],
                            value=max(self.unique_dates).strftime('%Y-%m-%d') if len(self.unique_dates) > 0 else None,
                            clearable=False,
                            style={'width': '120px', 'display': 'inline-block'},
                        )
                    ], className="date-selector-container"),
                    width=12,
                    className="mb-3"
                ),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Grasshopper Bank vs Peer Banks", className="card-title")),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-bar-chart",
                                type="circle",
                                children=dcc.Graph(id='bar-chart', config={'displayModeBar': True}, style={'height': '350px'})
                            )
                        ])
                    ], className="h-100")
                ], md=6, className="mb-4"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col(html.H5("Historical Performance", className="card-title"), width=8),
                                dbc.Col(
                                    html.P(id="historical-date-range", className="text-right", style={'fontSize': '0.8rem'}),
                                    width=4,
                                    style={'display': 'flex', 'justifyContent': 'flex-end', 'alignItems': 'center'}
                                ),
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-historical-chart",
                                type="circle",
                                children=dcc.Graph(id='historical-chart', config={'displayModeBar': True}, style={'height': '350px'})
                            )
                        ])
                    ], className="h-100")
                ], md=6, className="mb-4"),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Metric Overview", className="card-title")),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-metric-overview",
                                type="circle",
                                children=html.Div(id='metric-overview', className="p-0")
                            )
                        ], className="p-0")
                    ], className="h-100")
                ], md=6, className="mb-4"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5("Trend Analysis", className="card-title")),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-trend-analysis",
                                type="circle",
                                children=html.Div(id='trend-analysis', className="p-0")
                            )
                        ], className="p-0")
                    ], className="h-100")
                ], md=6, className="mb-4"),
            ]),

            dbc.Card([
                dbc.CardHeader(html.H5("Bank Details", className="card-title")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-bank-details",
                        type="circle",
                        children=html.Div(id='bank-details')
                    )
                ])
            ], className="mb-4"),

            html.Div("All data sourced through FDIC API", className="source-info")
        ], className="content")

    def _register_callbacks(self, app: dash.Dash) -> None:
        @app.callback(
            dash.Output('peer-selector', 'children'),
            dash.Input('metric-selector', 'value')
        )
        def update_peer_selector(dummy):
            all_available_banks = sorted(list(set(self.df['Bank'].unique()) - {PRIMARY_BANK_DISPLAY_NAME}))
            return dcc.Dropdown(
                id='individual-peer-selector',
                options=[{'label': peer, 'value': peer} for peer in all_available_banks],
                value=all_available_banks,
                multi=True,
                style={'width': '100%', 'color': COLOR_SCHEME['text']},
            )

        @app.callback(
            dash.Output('individual-peer-selector', 'value'),
            dash.Input('add-all-peers-btn', 'n_clicks'),
            dash.State('individual-peer-selector', 'options')
        )
        def add_all_peers(n_clicks, all_options):
            if n_clicks is None:
                raise PreventUpdate
            return [option['value'] for option in all_options]

        @app.callback(
            dash.Output('metric-definition', 'children'),
            dash.Input('metric-selector', 'value')
        )
        def update_metric_definition(selected_metric):
            return html.P(self.metric_definitions.get(selected_metric, ''), className="metric-definition")

        @app.callback(
            dash.Output('selected-peers-info', 'children'),
            dash.Input('individual-peer-selector', 'value')
        )
        def update_selected_peers_info(selected_peers):
            return html.Div([
                html.P(f"Selected Peers: {len(selected_peers)} banks", style={"fontWeight": "bold"}),
                html.Div(
                    [html.Span(peer, className="selected-peer-tag") for peer in selected_peers],
                    className="selected-peers-container"
                )
            ], style={"margin-top": "10px", "color": COLOR_SCHEME['text']})

        @app.callback(
            [
                dash.Output('bar-chart', 'figure'),
                dash.Output('metric-overview', 'children'),
                dash.Output('selected-metric-store', 'data')
            ],
            [
                dash.Input('metric-selector', 'value'),
                dash.Input('date-selector', 'value'),
                dash.Input('individual-peer-selector', 'value'),
            ]
        )
        def update_bar_chart(selected_metric, selected_date, selected_peers):
            if not selected_metric or not selected_date:
                return self._create_empty_figure("No data available"), html.Div("No data available"), selected_metric

            selected_date_dt = pd.to_datetime(selected_date).to_pydatetime()
            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + (selected_peers or [])
            filtered_df = self.df[(self.df['Date'] == selected_date_dt) & (self.df['Bank'].isin(selected_banks))]

            if filtered_df.empty:
                return self._create_empty_figure(f"No data available for {selected_date_dt.strftime('%m/%d/%y')}"), \
                       html.Div("No data available for the selected date"), selected_metric

            sorted_df = filtered_df.sort_values(by=selected_metric, ascending=False)
            fig = self._create_bar_chart(sorted_df, selected_metric, selected_date_dt)
            overview = self._create_metric_overview(filtered_df, selected_metric)
            return fig, overview, selected_metric

        @app.callback(
            [
                dash.Output('historical-chart', 'figure'),
                dash.Output('historical-date-range', 'children')
            ],
            [
                dash.Input('selected-metric-store', 'data'),
                dash.Input('individual-peer-selector', 'value'),
                dash.Input('trend-timeline-selector', 'value')
            ]
        )
        def update_historical_chart(selected_metric, selected_peers, trend_timeline):
            if not selected_metric:
                return self._create_empty_figure("No metric selected"), ""

            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + (selected_peers or [])

            end_date = self.df['Date'].max()
            start_date = end_date - pd.DateOffset(years=trend_timeline or 4)
            date_range_text = f"From {start_date.strftime('%m/%d/%y')} to {end_date.strftime('%m/%d/%y')}"
            return self._create_historical_chart(selected_banks, selected_metric, trend_timeline or 4), date_range_text

        @app.callback(
            dash.Output('trend-analysis', 'children'),
            [
                dash.Input('selected-metric-store', 'data'),
                dash.Input('individual-peer-selector', 'value'),
                dash.Input('trend-timeline-selector', 'value')
            ]
        )
        def update_trend_analysis(selected_metric, selected_peers, trend_timeline):
            if not selected_metric:
                return html.P("Select a metric to view trend analysis", style={"color": COLOR_SCHEME['text']})
            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + (selected_peers or [])
            return self._create_trend_analysis(selected_banks, selected_metric, trend_timeline or 4)

        @app.callback(
            [
                dash.Output('bank-details', 'children'),
                dash.Output('selected-bank-store', 'data')
            ],
            [
                dash.Input('bar-chart', 'clickData'),
                dash.Input('date-selector', 'value')
            ],
            [
                dash.State('metric-selector', 'value'),
                dash.State('individual-peer-selector', 'value'),
                dash.State('selected-bank-store', 'data')
            ]
        )
        def update_bank_details(clickData, selected_date, selected_metric, selected_peers, stored_bank):
            selected_date_dt = pd.to_datetime(selected_date).to_pydatetime() if selected_date else None

            if clickData:
                bank = clickData['points'][0]['x']
                stored_bank = bank
            elif stored_bank and selected_date_dt:
                bank = stored_bank
            else:
                return html.P("Click on a bar to see details", style={"color": COLOR_SCHEME['text']}), None

            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + (selected_peers or [])
            if bank not in selected_banks:
                return html.P("Selected bank is not in the current comparison. Please select a displayed bank.",
                              style={"color": COLOR_SCHEME['text']}), stored_bank

            bank_df = self.df[(self.df['Bank'] == bank) & (self.df['Date'] == selected_date_dt)]
            if bank_df.empty:
                return html.P(f"No data available for {bank} on {selected_date_dt.strftime('%m/%d/%y')}",
                              style={"color": COLOR_SCHEME['text']}), stored_bank

            bank_data = bank_df.iloc[0]
            details = self._create_bank_details(bank, bank_data, selected_date_dt)
            return details, stored_bank

    def _create_empty_figure(self, message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=message,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text=message,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=20, color=COLOR_SCHEME['text'])
            )],
            plot_bgcolor=COLOR_SCHEME['card_bg'],
            paper_bgcolor=COLOR_SCHEME['card_bg'],
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig

    def _create_bar_chart(self, df: pd.DataFrame, metric: str, date: datetime) -> go.Figure:
        colors = [COLOR_SCHEME['grasshopper'] if bank == PRIMARY_BANK_DISPLAY_NAME else COLOR_SCHEME['peer'] for bank in df['Bank']]
        opacities = [1.0 if bank == PRIMARY_BANK_DISPLAY_NAME else 0.6 for bank in df['Bank']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['Bank'],
            y=df[metric],
            marker_color=colors,
            marker_opacity=opacities,
            hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:,.2f}<extra></extra>',
            name=''
        ))

        y_min = df[metric].min()
        y_max = df[metric].max()
        y_range = y_max - y_min
        y_padding = y_range * 0.1 if y_range != 0 else (abs(y_max) * 0.1 + 1)

        formatted_date = date.strftime('%m/%d/%y')
        fig.update_layout(
            title=f"{metric} as of {formatted_date}",
            title_x=0.01,
            margin=dict(l=50, r=20, t=50, b=100),
            plot_bgcolor=COLOR_SCHEME['card_bg'],
            paper_bgcolor=COLOR_SCHEME['card_bg'],
            font=dict(color=COLOR_SCHEME['text']),
            hoverlabel=dict(bgcolor=COLOR_SCHEME['card_bg'], font_size=12, font_color=COLOR_SCHEME['text']),
            xaxis=dict(
                title=None,
                tickangle=-45,
                tickfont=dict(size=10),
                showgrid=True,
                gridcolor=COLOR_SCHEME['grid'],
                gridwidth=1
            ),
            yaxis=dict(
                title=None,
                tickformat=',.0f' if metric in self.dollar_format_metrics else '.2f',
                range=[y_min - y_padding, y_max + y_padding],
                showgrid=True,
                gridcolor=COLOR_SCHEME['grid'],
                gridwidth=1
            )
        )
        return fig

    def _create_historical_chart(self, selected_banks: List[str], metric: str, trend_timeline: int) -> go.Figure:
        fig = go.Figure()
        filtered_df = self.df[self.df['Bank'].isin(selected_banks)]
        if filtered_df.empty:
            return self._create_empty_figure("No historical data available")

        end_date = filtered_df['Date'].max()
        start_date = end_date - pd.DateOffset(years=trend_timeline)
        filtered_df = filtered_df[filtered_df['Date'] >= start_date]

        pivot_df = filtered_df.pivot(index='Date', columns='Bank', values=metric)
        for bank in pivot_df.columns:
            color = COLOR_SCHEME['grasshopper'] if bank == PRIMARY_BANK_DISPLAY_NAME else COLOR_SCHEME['peer']
            line_width = 3 if bank == PRIMARY_BANK_DISPLAY_NAME else 1.5
            opacity = 1 if bank == PRIMARY_BANK_DISPLAY_NAME else COLOR_SCHEME['peer_opacity']
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df[bank],
                mode='lines',
                name=bank,
                line=dict(color=color, width=line_width),
                opacity=opacity,
                hovertemplate='%{x|%m/%d/%y}<br>' + bank + ': %{y:,.2f}<extra></extra>'
            ))

        if trend_timeline <= 2:
            dtick = 'M3'
            tickformat = '%b\n%Y'
        elif trend_timeline <= 4:
            dtick = 'M6'
            tickformat = '%b\n%Y'
        else:
            dtick = 'M12'
            tickformat = '%Y'

        fig.update_layout(
            title=f"{metric} - {trend_timeline} Year Trend",
            margin=dict(l=50, r=20, t=50, b=50),
            plot_bgcolor=COLOR_SCHEME['card_bg'],
            paper_bgcolor=COLOR_SCHEME['card_bg'],
            font=dict(color=COLOR_SCHEME['text']),
            hoverlabel=dict(bgcolor=COLOR_SCHEME['card_bg'], font_size=12, font_color=COLOR_SCHEME['text']),
            showlegend=False,
            xaxis=dict(
                title=None,
                showgrid=True,
                gridcolor=COLOR_SCHEME['grid'],
                tickformat=tickformat,
                dtick=dtick,
                tickangle=-45,
                tickfont=dict(size=9)
            ),
            yaxis=dict(
                title=None,
                showgrid=True,
                gridcolor=COLOR_SCHEME['grid'],
                tickformat=',.0f' if metric in self.dollar_format_metrics else '.2f',
                tickfont=dict(size=9)
            )
        )
        return fig

    def _create_metric_overview(self, df: pd.DataFrame, metric: str) -> html.Div:
        def format_value(value):
            if pd.isna(value):
                return "N/A"
            if metric in self.dollar_format_metrics:
                return f"${value:,.0f}"
            return f"{value:.2f}"

        gh_df = df[df['Bank'] == PRIMARY_BANK_DISPLAY_NAME]
        gh_value = gh_df[metric].values[0] if not gh_df.empty else None

        if gh_value is not None:
            gh_percentile = stats.percentileofscore(df[metric], gh_value)
            gh_rank = df[metric].rank(ascending=False, method='min')[df['Bank'] == PRIMARY_BANK_DISPLAY_NAME].values[0]
        else:
            gh_percentile = None
            gh_rank = None

        q1, q3 = np.percentile(df[metric], [25, 75])

        if gh_value is not None:
            if gh_value > q3:
                performance_group = "Top 25%"
                performance_color = COLOR_SCHEME['good']
            elif gh_value <= q1:
                performance_group = "Bottom 25%"
                performance_color = COLOR_SCHEME['danger']
            else:
                performance_group = "Middle 50%"
                performance_color = COLOR_SCHEME['warning']
        else:
            performance_group = "N/A"
            performance_color = COLOR_SCHEME['text']

        return html.Div([
            html.Div([
                html.Div("Current Snapshot", className="stat-section-title"),
                html.Div([html.Div("Average:", className="stat-label"), html.Div(format_value(df[metric].mean()))], className="stat-row"),
                html.Div([html.Div("Median:", className="stat-label"), html.Div(format_value(df[metric].median()))], className="stat-row"),
                html.Div([html.Div(f"{PRIMARY_BANK_DISPLAY_NAME} Value:", className="stat-label gh-highlight"),
                          html.Div(format_value(gh_value) if gh_value is not None else "N/A", className="gh-highlight")], className="stat-row"),
                html.Div([html.Div("Highest:", className="stat-label"),
                          html.Div(f"{format_value(df[metric].max())} ({df.loc[df[metric].idxmax(), 'Bank']})" if not df.empty else "N/A")],
                         className="stat-row"),
                html.Div([html.Div("Lowest:", className="stat-label"),
                          html.Div(f"{format_value(df[metric].min())} ({df.loc[df[metric].idxmin(), 'Bank']})" if not df.empty else "N/A")],
                         className="stat-row"),
            ], className="stat-section"),

            html.Div([
                html.Div(f"{PRIMARY_BANK_DISPLAY_NAME}'s Position", className="stat-section-title"),
                html.Div([html.Div("Percentile:", className="stat-label"),
                          html.Div(f"{gh_percentile:.1f}%" if gh_percentile is not None else "N/A")], className="stat-row"),
                html.Div([html.Div("Ranking:", className="stat-label"),
                          html.Div(f"#{gh_rank:.0f} out of {len(df)} banks" if gh_rank is not None else "N/A")], className="stat-row"),
                html.Div([html.Div("Group:", className="stat-label"),
                          html.Div(performance_group, style={"color": performance_color, "fontWeight": "bold"})], className="stat-row"),
                html.Div([html.Div("Standout Score:", className="stat-label"),
                          html.Div(self._calculate_zscore_display(df, metric) if gh_value is not None else "N/A")], className="stat-row"),
            ], className="stat-section"),
        ], style={"background-color": COLOR_SCHEME['card_bg'], "border-radius": "8px", "color": COLOR_SCHEME['text'], "padding": "10px"})

    def _calculate_zscore_display(self, df, metric):
        try:
            z_values = stats.zscore(df[metric].values)
            gh_index = df.index[df['Bank'] == PRIMARY_BANK_DISPLAY_NAME].tolist()
            if gh_index:
                return f"{z_values[gh_index[0]]:.2f} (std devs from avg)"
            return "N/A"
        except Exception:
            return "N/A (Cannot calculate)"

    def _create_trend_analysis(self, selected_banks: List[str], metric: str, trend_timeline: int) -> html.Div:
        filtered_df = self.df[self.df['Bank'].isin(selected_banks)]
        if filtered_df.empty:
            return html.Div("No trend data available")

        end_date = filtered_df['Date'].max()
        start_date = end_date - pd.DateOffset(years=trend_timeline)
        filtered_df = filtered_df[filtered_df['Date'] >= start_date]

        pivot_df = filtered_df.pivot(index='Date', columns='Bank', values=metric)
        if PRIMARY_BANK_DISPLAY_NAME not in pivot_df.columns or pivot_df[PRIMARY_BANK_DISPLAY_NAME].count() < 2:
            return html.Div(f"Insufficient data for {PRIMARY_BANK_DISPLAY_NAME} to perform trend analysis")

        gh_data = pivot_df[PRIMARY_BANK_DISPLAY_NAME].dropna()
        stats_data = {}

        first_values = pivot_df.iloc[0]
        last_values = pivot_df.iloc[-1]
        valid_banks = pivot_df.columns[~pivot_df.iloc[-1].isna() & ~pivot_df.iloc[0].isna()]

        for bank in valid_banks:
            bank_data = pivot_df[bank].dropna()
            if len(bank_data) < 2:
                continue
            growth_rate = ((last_values[bank] / first_values[bank]) - 1) * 100
            volatility = bank_data.std()

            if bank != PRIMARY_BANK_DISPLAY_NAME and len(gh_data) >= 2:
                overlap_df = pd.concat([gh_data, bank_data], axis=1).dropna()
                correlation = overlap_df.iloc[:, 0].corr(overlap_df.iloc[:, 1]) if len(overlap_df) >= 2 else np.nan
            else:
                correlation = np.nan

            x = np.arange(len(bank_data))
            slope, _ = np.polyfit(x, bank_data.values, 1)
            trend_direction = "Increasing" if slope > 0 else "Decreasing"

            stats_data[bank] = {
                'growth_rate': growth_rate,
                'volatility': volatility,
                'correlation': correlation if bank != PRIMARY_BANK_DISPLAY_NAME else np.nan,
                'trend_direction': trend_direction
            }

        correlations = {bank: data['correlation'] for bank, data in stats_data.items()
                        if bank != PRIMARY_BANK_DISPLAY_NAME and not np.isnan(data['correlation'])}

        most_similar = max(correlations.items(), key=lambda x: x[1]) if correlations else (None, np.nan)
        least_similar = min(correlations.items(), key=lambda x: x[1]) if correlations else (None, np.nan)

        avg_growth = np.mean([data['growth_rate'] for data in stats_data.values() if not np.isnan(data['growth_rate'])]) if stats_data else np.nan
        avg_volatility = np.mean([data['volatility'] for data in stats_data.values() if not np.isnan(data['volatility'])]) if stats_data else np.nan

        return html.Div([
            html.Div([
                html.Div(f"{trend_timeline}-Year Trend Analysis", className="stat-section-title"),
                html.Div([html.Div(f"{PRIMARY_BANK_DISPLAY_NAME} Growth:", className="stat-label"),
                          html.Div(f"{stats_data.get(PRIMARY_BANK_DISPLAY_NAME, {}).get('growth_rate', np.nan):.2f}%")],
                         className="stat-row"),
                html.Div([html.Div("Average Growth:", className="stat-label"),
                          html.Div(f"{avg_growth:.2f}%") if not np.isnan(avg_growth) else html.Div("N/A")],
                         className="stat-row"),
                html.Div([html.Div("Average Volatility:", className="stat-label"),
                          html.Div(f"{avg_volatility:.4f}") if not np.isnan(avg_volatility) else html.Div("N/A")],
                         className="stat-row"),
                html.Div([html.Div(f"Most Like {PRIMARY_BANK_DISPLAY_NAME}:", className="stat-label"),
                          html.Div(f"{most_similar[0]} (corr: {most_similar[1]:.2f})") if most_similar[0] else html.Div("N/A")],
                         className="stat-row"),
                html.Div([html.Div(f"Least Like {PRIMARY_BANK_DISPLAY_NAME}:", className="stat-label"),
                          html.Div(f"{least_similar[0]} (corr: {least_similar[1]:.2f})") if least_similar[0] else html.Div("N/A")],
                         className="stat-row"),
            ], className="stat-section"),
        ], style={"background-color": COLOR_SCHEME['card_bg'], "border-radius": "8px", "color": COLOR_SCHEME['text'], "padding": "10px"})

    def _create_bank_details(self, bank: str, bank_data: pd.Series, date: datetime) -> html.Div:
        formatted_date = date.strftime('%m/%d/%y')
        is_gh = bank == PRIMARY_BANK_DISPLAY_NAME
        card_bg_color = COLOR_SCHEME['grasshopper'] if is_gh else COLOR_SCHEME['secondary']

        metric_cols = []
        ordered_metrics = [m for m in self.available_metrics if m not in ['Bank', 'Date']]

        for col_idx in range(4):
            col_metrics = ordered_metrics[col_idx::4]
            metrics_html = []
            for metric in col_metrics:
                value = bank_data[metric]
                if pd.isna(value):
                    formatted_value = "N/A"
                elif metric in self.dollar_format_metrics:
                    formatted_value = f"${value:,.0f}"
                else:
                    formatted_value = f"{value:.2f}"
                metrics_html.append(
                    html.Div([
                        html.Div(metric, className="bank-detail-label"),
                        html.Div(formatted_value, className="bank-detail-value")
                    ], className="bank-detail-row")
                )
            metric_cols.append(dbc.Col(html.Div(metrics_html, className="bank-detail-col"), xs=12, sm=6, md=3))

        return html.Div([
            html.Div([
                html.H5(f"{bank} Metrics as of {formatted_date}", style={"color": "white", "margin": "0"}),
            ], style={"backgroundColor": card_bg_color, "padding": "10px 15px", "borderRadius": "8px 8px 0 0"}),
            html.Div(
                dbc.Row(metric_cols),
                style={"padding": "15px", "backgroundColor": COLOR_SCHEME['card_bg'], "borderRadius": "0 0 8px 8px",
                       "color": COLOR_SCHEME['text']}
            )
        ])

    def _get_custom_css(self) -> str:
        # Kept functional but trimmed slightly to reduce noise—safe to expand again if you want.
        return """
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f0f0; margin:0; }
            #app-container { display:flex; min-height:100vh; padding:0 10px; }
            .sidebar { width:450px; background:#fff; padding:1.5rem 1rem; margin-right:20px; overflow-y:auto;
                       border-right:1px solid #e0e0e0; box-shadow:0 2px 10px rgba(0,0,0,0.05); border-radius:10px;
                       display:flex; flex-direction:column; position:sticky; top:10px; height:calc(100vh - 20px); }
            .content { flex-grow:1; padding:1.5rem; overflow-y:auto; background:#f0f0f0; }
            .card { background:#fff; border:none; border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.05); overflow:hidden; }
            .card-header { background:#fff; border-bottom:1px solid #e0e0e0; padding:0.75rem 1rem; }
            .card-body { padding:1rem; }
            .date-selector-container { display:flex; align-items:center; background:#fff; padding:10px 15px; border-radius:10px;
                                       box-shadow:0 2px 10px rgba(0,0,0,0.05); margin-bottom:10px; }
            .date-label { font-weight:bold; margin-right:10px; color:#0E3E1B; }
            .metric-definition { font-size:0.9rem; color:#666; margin-top:10px; }
            .stat-section { margin-bottom:15px; padding:10px; background:#f5f5f5; border-radius:5px; }
            .stat-section-title { font-weight:bold; margin-bottom:5px; color:#0E3E1B; }
            .stat-row { display:flex; justify-content:space-between; margin-bottom:5px; }
            .stat-label { font-weight:bold; }
            .gh-highlight { color:#0E3E1B; font-weight:bold; }
            .add-all-btn { background:#0E3E1B; color:#fff; border:none; padding:5px 10px; border-radius:4px; cursor:pointer; }
            .add-all-btn:hover { background:#0A2E15; }
            .bank-detail-row { display:flex; justify-content:space-between; margin-bottom:8px; border-bottom:1px solid #e0e0e0; padding-bottom:5px; }
            .bank-detail-label, .bank-detail-value { font-size:0.8rem; flex:1; }
            .bank-detail-value { text-align:right; }
            .selected-peers-container { display:flex; flex-wrap:wrap; gap:5px; margin-top:5px; }
            .selected-peer-tag { background:#f0f0f0; padding:2px 8px; border-radius:10px; font-size:0.8rem; white-space:nowrap; }
            .source-info { font-size:0.8rem; color:#666; text-align:center; padding:10px 0; }
            @media (max-width: 992px) {
                #app-container { flex-direction:column; }
                .sidebar { width:100%; height:auto; position:static; margin-right:0; margin-bottom:20px; }
                .content { padding:1rem; }
            }
        """


def load_sample_data():
    logger.info("Loading sample data for development/testing")

    all_banks = list(BANK_NAME_MAPPING.values())
    all_month_ends = pd.date_range(start='2020-06-30', end='2024-12-31', freq='ME')
    dates = all_month_ends[all_month_ends.month.isin([3, 6, 9, 12])]

    data = []
    for bank in all_banks:
        base_assets = np.random.uniform(2000000000, 3000000000) if bank == PRIMARY_BANK_DISPLAY_NAME else np.random.uniform(1500000000, 5000000000)
        growth_rate = np.random.uniform(0.015, 0.08)

        for i, date in enumerate(dates):
            year_factor = 1 + (growth_rate * i) + np.random.uniform(-0.02, 0.02)
            total_assets = base_assets * year_factor
            total_deposits = total_assets * np.random.uniform(0.68, 0.78)
            total_loans = total_assets * np.random.uniform(0.68, 0.75)
            net_loans = total_loans * np.random.uniform(0.95, 0.99)
            tier1_capital = total_assets * np.random.uniform(0.11, 0.15)
            allowance_for_credit_loss = total_loans * np.random.uniform(0.01, 0.02)

            efficiency_ratio = np.random.uniform(50.0, 60.0) if bank == PRIMARY_BANK_DISPLAY_NAME else np.random.uniform(52.0, 68.0)
            roa = np.random.uniform(0.8, 1.6) if bank == PRIMARY_BANK_DISPLAY_NAME else np.random.uniform(0.6, 1.4)
            roe = np.random.uniform(9.0, 15.0) if bank == PRIMARY_BANK_DISPLAY_NAME else np.random.uniform(7.0, 14.0)

            real_estate_loans = total_loans * np.random.uniform(0.35, 0.55)
            construction_loans = real_estate_loans * np.random.uniform(0.08, 0.18)
            multifamily = real_estate_loans * np.random.uniform(0.1, 0.2)
            nonres_properties = real_estate_loans * np.random.uniform(0.3, 0.5)
            nim = np.random.uniform(2.8, 4.2)

            commercial_loans = total_loans * np.random.uniform(0.3, 0.45)
            auto_loans = total_loans * np.random.uniform(0.05, 0.12)
            credit_cards = total_loans * np.random.uniform(0.01, 0.04)
            consumer_loans = total_loans * np.random.uniform(0.08, 0.18)

            re_loans_to_tier1 = (real_estate_loans / (tier1_capital + allowance_for_credit_loss)) * 100
            construction_to_tier1 = (construction_loans / (tier1_capital + allowance_for_credit_loss)) * 100
            commercial_re_to_tier1 = ((construction_loans + multifamily + nonres_properties) / (tier1_capital + allowance_for_credit_loss)) * 100
            ci_loans_to_tier1 = (commercial_loans / (tier1_capital + allowance_for_credit_loss)) * 100
            auto_loans_to_tier1 = (auto_loans / (tier1_capital + allowance_for_credit_loss)) * 100

            data.append({
                'Bank': bank,
                'Date': date,
                'Total Assets': total_assets,
                'Total Deposits': total_deposits,
                'Total Loans and Leases': total_loans,
                'Net Loans and Leases': net_loans,
                'Total Securities': total_assets * np.random.uniform(0.12, 0.22),
                'Real Estate Loans': real_estate_loans,
                'RE Construction and Land Development': construction_loans,
                'Multifamily': multifamily,
                'Loans to Nonresidential Properties': nonres_properties,
                'Commercial and Industrial Loans': commercial_loans,
                'Auto Loans': auto_loans,
                'Credit Cards': credit_cards,
                'Consumer Loans': consumer_loans,
                'Tier 1 (Core) Capital': tier1_capital,
                'Allowance for Credit Loss': allowance_for_credit_loss,
                'Net Income': total_assets * np.random.uniform(0.009, 0.016),
                'Return on Assets': roa,
                'Return on Equity': roe,
                'Net Interest Margin': nim,
                'Efficiency Ratio': efficiency_ratio,
                'Leverage (Core Capital) Ratio': np.random.uniform(10.0, 14.0),
                'Total Risk-Based Capital Ratio': np.random.uniform(13.0, 18.0),
                'Net Loans and Leases to Deposits': (net_loans / total_deposits) * 100,
                'Net Loans and Leases to Assets': (net_loans / total_assets) * 100,
                'Real Estate Loans to Tier 1 + ACL': re_loans_to_tier1,
                'RE Construction and Land Development to Tier 1 + ACL': construction_to_tier1,
                'Commercial RE to Tier 1 + ACL': commercial_re_to_tier1,
                'C&I Loans to Tier 1 + ACL': ci_loans_to_tier1,
                'Auto Loans to Tier 1 + ACL': auto_loans_to_tier1,
                'Nonperforming Assets / Total Assets': np.random.uniform(0.1, 1.0),
                'Assets Past Due 30-89 Days / Total Assets': np.random.uniform(0.05, 0.5),
                'Assets Past Due 90+ Days / Total Assets': np.random.uniform(0.01, 0.3),
                'Noncurrent Loans / Total Loans': np.random.uniform(0.3, 2.0),
                'Net Charge-Offs / Total Loans & Leases': np.random.uniform(0.1, 0.8),
                'Net Charge-Offs / Allowance for Credit Loss': np.random.uniform(1.0, 15.0),
            })

    return pd.DataFrame(data)


def generate_temp_data(unique_banks: List[str], unique_dates: List[datetime]) -> pd.DataFrame:
    data = []
    for bank in unique_banks:
        for date in unique_dates:
            total_assets = np.random.uniform(1000000000, 5000000000)
            tier1_capital = total_assets * 0.12
            data.append({
                'Bank': bank,
                'Date': date,
                'Total Assets': total_assets,
                'Tier 1 (Core) Capital': tier1_capital,
                'Return on Assets': np.random.uniform(0.6, 1.5),
                'Return on Equity': np.random.uniform(7.0, 15.0),
            })
    return pd.DataFrame(data)


def main() -> Tuple[dash.Dash, Any]:
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

    try:
        data_service = BankDataService()
        metrics_df, dollar_format_metrics, metric_definitions = data_service.get_metrics_data(
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE
        )

        if metrics_df.empty:
            logger.warning("No data from API. Using sample data instead.")
            metrics_df = load_sample_data()
            calculator = BankMetricsCalculator([
                'Total Assets', 'Total Deposits', 'Total Loans and Leases',
                'Net Loans and Leases', 'Total Securities', 'Real Estate Loans',
                'Tier 1 (Core) Capital', 'Net Income', 'Auto Loans'
            ])
            dollar_format_metrics = calculator.dollar_format_metrics
            metric_definitions = calculator.metric_definitions

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        logger.warning("Using sample data instead.")
        metrics_df = load_sample_data()
        calculator = BankMetricsCalculator([
            'Total Assets', 'Total Deposits', 'Total Loans and Leases',
            'Net Loans and Leases', 'Total Securities', 'Real Estate Loans',
            'Tier 1 (Core) Capital', 'Net Income', 'Auto Loans'
        ])
        dollar_format_metrics = calculator.dollar_format_metrics
        metric_definitions = calculator.metric_definitions

    expected_banks = list(BANK_NAME_MAPPING.values())
    actual_banks = metrics_df['Bank'].unique()
    missing_banks = [bank for bank in expected_banks if bank not in actual_banks]

    if missing_banks:
        logger.warning(f"Missing data for banks: {missing_banks}. Generating fallback data.")
        fallback_df = generate_temp_data(missing_banks, metrics_df['Date'].unique())
        metrics_df = pd.concat([metrics_df, fallback_df])

    dashboard_builder = DashboardBuilder(metrics_df, dollar_format_metrics, metric_definitions)
    app = dashboard_builder.create_dashboard()
    server = app.server
    return app, server


app, server = main()

if __name__ == "__main__":
    app.run_server(debug=False)
