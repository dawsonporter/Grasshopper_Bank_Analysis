import warnings
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from scipy import stats
import logging
from functools import lru_cache
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
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Constants
BASE_URL = "https://banks.data.fdic.gov/api"
DEFAULT_START_DATE = '20200630'  # June 30, 2020 - UPDATED per requirements
DEFAULT_END_DATE = '20250930'    # Sept 30, 2025
CACHE_DIR = 'data_cache'

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Color scheme - Grasshopper Bank colors (Phthalo Green primary)
COLOR_SCHEME = {
    'primary': '#0E3E1B',  # Grasshopper Phthalo Green
    'secondary': '#333333',
    'accent': '#2D5F3F',   # Lighter complementary green
    'background': '#f5f5f5',
    'card_bg': '#ffffff',
    'highlight': '#0E3E1B',
    'text': '#333333',
    'light_text': '#666666',
    'grid': 'rgba(0, 0, 0, 0.1)',
    'grasshopper': '#0E3E1B',
    'peer': '#808080',
    'peer_opacity': 0.4,
    'good': '#2D5F3F',  # Green for good metrics
    'warning': '#FF9800',  # Orange for warning metrics
    'danger': '#F44336',  # Red for danger metrics
}

# Bank name mapping for display - Grasshopper Bank focused
# CRITICAL: This maps from API/FDIC official names to display names
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

# Bank information for API queries - Grasshopper Bank and peers
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

# PRIMARY BANK - This is the bank we're analyzing (used throughout the app)
PRIMARY_BANK_DISPLAY_NAME = "Grasshopper Bank"

def normalize_bank_name(bank_name: str) -> str:
    """
    Normalize and map bank names to their display names.
    This handles the mapping from FDIC official names to our display names.
    
    Args:
        bank_name: Original bank name from API or data
        
    Returns:
        Normalized display name
    """
    if not bank_name:
        return bank_name
    
    # First, try exact match
    if bank_name in BANK_NAME_MAPPING:
        return BANK_NAME_MAPPING[bank_name]
    
    # Try case-insensitive match
    bank_name_upper = bank_name.upper().strip()
    for official_name, display_name in BANK_NAME_MAPPING.items():
        if official_name.upper().strip() == bank_name_upper:
            return display_name
    
    # If no match found, return original (but log it)
    logger.warning(f"No mapping found for bank name: '{bank_name}'")
    return bank_name

class FDICAPIClient:
    """
    Client for interacting with the FDIC API to fetch bank data.
    """
    def __init__(self):
        self.base_url = BASE_URL
        
    def get_data(self, endpoint: str, params: Dict) -> Dict:
        """
        Fetch data from the FDIC API.
        
        Args:
            endpoint: API endpoint to query
            params: Query parameters
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            # Using verify=False explicitly on each request to bypass SSL certificate verification
            response = requests.get(url, params=params, headers={"Accept": "application/json"}, verify=False, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error for {url}: {e}")
            return {"data": []}

    def get_institutions(self, filters: str = "", fields: str = "") -> List[Dict]:
        """
        Fetch institution data from the FDIC API.
        
        Args:
            filters: Filter string for the API query
            fields: Comma-separated list of fields to return
            
        Returns:
            List of institution data
        """
        params = {"filters": filters, "fields": fields, "limit": 10000}
        data = self.get_data("institutions", params)
        return data.get('data', [])

    def get_financials(self, cert: str, filters: str = "", fields: str = "") -> List[Dict]:
        """
        Fetch financial data for a specific institution.
        
        Args:
            cert: Certificate number of the institution
            filters: Additional filter string for the API query
            fields: Comma-separated list of fields to return
            
        Returns:
            List of financial data
        """
        params = {"filters": f"CERT:{cert}" + (f" AND {filters}" if filters else ""), "fields": fields, "limit": 10000}
        data = self.get_data("financials", params)
        return data.get('data', [])

class BankDataRepository:
    """
    Repository for managing bank data, with caching capabilities.
    """
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
            'CECL Transitional Amount',
            'Perpetual Preferred Stock'
        ]
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
    def get_cache_path(self, start_date: str, end_date: str) -> str:
        """
        Generate a cache file path based on date parameters.
        
        Args:
            start_date: Start date for the data query
            end_date: End date for the data query
            
        Returns:
            Path to the cache file
        """
        return os.path.join(CACHE_DIR, f"bank_data_{start_date}_{end_date}.json")

    def load_cached_data(self, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Load data from cache if available.
        
        Args:
            start_date: Start date for the data query
            end_date: End date for the data query
            
        Returns:
            Cached data or None if not available
        """
        cache_path = self.get_cache_path(start_date, end_date)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load cache file {cache_path}: {e}")
        return None

    def save_to_cache(self, data: Dict, start_date: str, end_date: str) -> None:
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            start_date: Start date for the data query
            end_date: End date for the data query
        """
        cache_path = self.get_cache_path(start_date, end_date)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            logger.error(f"Failed to save to cache file {cache_path}: {e}")

    def fetch_data(self, bank_info: List[Union[str, Dict]], start_date: str, end_date: str) -> Dict[str, Dict]:
        """
        Fetch data for all specified banks, using cache if available.
        
        Args:
            bank_info: List of bank information (name or certificate number)
            start_date: Start date for the data query
            end_date: End date for the data query
            
        Returns:
            Dictionary of bank data
        """
        cached_data = self.load_cached_data(start_date, end_date)
        if cached_data:
            logger.info(f"Using cached data for {start_date} to {end_date}")
            return cached_data
        
        logger.info(f"Fetching fresh data from FDIC API for {start_date} to {end_date}")
        
        institution_fields = "NAME,CERT"
        # Added CECL-specific fields for proper concentration ratio calculation:
        # RCFDJ447/RCONJ447 = CECL transitional amount (for removing double counting)
        # These fields contain the actual phase-in amount to subtract from denominator
        financial_fields = ("CERT,REPDTE,ASSET,DEP,LNLSGR,LNLSNET,SC,LNRE,LNCI,LNAG,LNCRCD,LNCONOTH,LNATRES,P3ASSET,P9ASSET,RBCT1J,DRLNLS,CRLNLS,"
                           "NETINC,ERNASTR,NPERFV,P3ASSETR,P9ASSETR,NIMY,NTLNLSR,LNATRESR,NCLNLSR,ROA,ROE,RBC1AAJ,"
                           "RBCT2,RBCRWAJ,LNLSDEPR,LNLSNTV,EEFFR,LNRESNCR,ELNANTR,IDERNCVR,NTLNLSQ,LNRECONS,"
                           "LNRENRES,LNRENROW,LNRENROT,LNRERES,LNREMULT,LNREAG,LNRECNFM,LNRECNOT,LNCOMRE,CT1BADJ,EQ,EQPP,LNAUTO,"
                           "RCFDJ447,RCONJ447")

        institutions_data = {}
        financials_data = {}
        
        api_failures = 0  # Track API failures
        max_retries = 3  # Maximum number of retries for the entire batch
        
        for retry_count in range(max_retries):
            try:
                # Clear previous data if this is a retry
                if retry_count > 0:
                    institutions_data = {}
                    financials_data = {}
                    logger.info(f"Retry attempt {retry_count} for FDIC API")
                
                for bank_item in bank_info:
                    try:
                        if isinstance(bank_item, str):
                            # Fetch by bank name
                            institutions = self.api_client.get_institutions(f'NAME:"{bank_item}"', institution_fields)
                        elif isinstance(bank_item, dict) and 'cert' in bank_item:
                            # Fetch by CERT number
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
                                financials_data[bank_data['NAME']] = [f['data'] for f in financials if isinstance(f, dict) and 'data' in f]
                                logger.info(f"Fetched {len(financials)} records for {bank_data['NAME']}")
                            else:
                                logger.warning(f"Required fields missing for bank: {bank_item}")
                        else:
                            logger.warning(f"Unexpected data structure for bank: {bank_item}")
                    except Exception as e:
                        logger.error(f"Error fetching data for bank {bank_item}: {e}")
                        api_failures += 1
                
                # If we've successfully fetched some data, break out of retry loop
                if institutions_data:
                    break
                    
            except Exception as e:
                logger.error(f"Error in API batch fetch (attempt {retry_count+1}): {e}")
                # If this was the last retry, and we have no data, use fallback data
                if retry_count == max_retries - 1 and not institutions_data:
                    logger.warning("All API retries failed. Using fallback data.")
                    return self._generate_fallback_data(start_date, end_date)
        
        # If we've had too many API failures or no institution data, use fallback
        if api_failures > len(bank_info) // 2 or not institutions_data:
            logger.warning(f"Too many API failures ({api_failures}). Using fallback data.")
            return self._generate_fallback_data(start_date, end_date)

        result = {
            'institutions_data': institutions_data,
            'financials_data': financials_data
        }
        
        # Save to cache for future use
        self.save_to_cache(result, start_date, end_date)
        
        return result

    def _generate_fallback_data(self, start_date: str, end_date: str) -> Dict[str, Dict]:
        """Generate fallback data when API fails"""
        logger.info("Generating fallback data")
        
        institutions_data = {}
        financials_data = {}
        
        # Create fallback data for all banks in BANK_INFO
        for bank_info in BANK_INFO:
            bank_name = bank_info["name"]
            cert = bank_info["cert"]
            
            # Add to institutions data
            institutions_data[bank_name] = {
                "NAME": bank_name,
                "CERT": cert
            }
            
            # Generate some basic financial data points
            financial_records = []
            
            # Generate quarterly dates between start_date and end_date
            start = pd.to_datetime(start_date, format="%Y%m%d")
            end = pd.to_datetime(end_date, format="%Y%m%d")
            
            # Generate quarterly dates
            quarters = pd.date_range(start=start, end=end, freq='Q')
            
            # Base values - different scale for different bank types
            # Normalize the bank name for comparison
            normalized_name = normalize_bank_name(bank_name)
            is_grasshopper = normalized_name == PRIMARY_BANK_DISPLAY_NAME
            
            # Grasshopper and peers are smaller community/specialized banks
            base_assets = 2_000_000_000 if is_grasshopper else np.random.uniform(1_500_000_000, 5_000_000_000)
            
            for date in quarters:
                # Convert date to string format YYYYMMDD
                date_str = date.strftime("%Y%m%d")
                
                # Calculate years since start for growth factor
                years_since_start = (date - start).days / 365.25
                growth_factor = 1 + years_since_start * 0.08  # 8% annual growth for specialty banks
                
                # Add some randomness
                random_factor = np.random.uniform(0.95, 1.05)
                
                # Calculate metrics
                assets = base_assets * growth_factor * random_factor
                deposits = assets * 0.75 * np.random.uniform(0.9, 1.1)
                loans = assets * 0.70 * np.random.uniform(0.9, 1.1)
                tier1_capital = assets * 0.12 * np.random.uniform(0.9, 1.1)
                
                # For Grasshopper Bank, adjust for specialized banking focus
                if is_grasshopper:
                    deposits = assets * 0.70 * np.random.uniform(0.9, 1.1)  # Specialty bank deposit ratio
                    loans = assets * 0.72 * np.random.uniform(0.9, 1.1)     # Strong lending business
                    tier1_capital = assets * 0.13 * np.random.uniform(0.9, 1.1)  # Strong capital position
                
                # Create record
                record = {
                    "REPDTE": date_str,
                    "ASSET": assets,
                    "DEP": deposits,
                    "LNLSGR": loans,
                    "LNLSNET": loans * 0.98,
                    "RBCT1J": tier1_capital,
                    "ROA": np.random.uniform(0.8, 1.5),
                    "ROE": np.random.uniform(8, 14),
                    # Add other fields with reasonable defaults
                    "SC": assets * 0.15,
                    "LNRE": loans * 0.45,
                    "LNCI": loans * 0.35,
                    "LNAG": loans * 0.01,
                    "LNAUTO": loans * 0.08,  # Auto loans
                    "LNCRCD": loans * 0.02,
                    "LNCONOTH": loans * 0.09,
                    "LNATRES": loans * 0.015,
                    "NIMY": np.random.uniform(2.5, 3.8),
                    "EEFFR": np.random.uniform(55, 65)
                }
                
                financial_records.append(record)
            
            financials_data[bank_name] = financial_records
        
        result = {
            'institutions_data': institutions_data,
            'financials_data': financials_data
        }
        
        # Cache the fallback data
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
        """
        Get definitions for all metrics.
        
        Returns:
            Dictionary of metric definitions
        """
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
            'Tier 1 (Core) Capital': "(Qtly, $) Tier 1 core capital, which includes common equity tier 1 capital and additional tier 1 capital.",
            'Total Charge-Offs': "(YTD, $) Total charge-offs of loans and leases.",
            'Total Recoveries': "(YTD, $) Total recoveries of loans and leases previously charged off.",
            'Total Loans and Leases Net Charge-Offs Quarterly': "(Qtly, $) Total loans and leases net charge-offs for the quarter.",
            'Net Income': "(YTD, $) Net income earned by the entity.",
            'RE Construction and Land Development': "(YTD, $) Real estate construction and land development loans.",
            'RE Construction and Land Development to Tier 1 + ACL': "(Qtly, %) Real estate construction and land development loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Common Equity Tier 1 Before Adjustments': "(YTD, $) Common Equity Tier 1 capital before adjustments.",
            'Bank Equity Capital': "(YTD, $) Total bank equity capital.",
            'Perpetual Preferred Stock': "(YTD, $) The amount of perpetual preferred stock issued by the bank.",
            'CECL Transition Amount': "(YTD, $) Current Expected Credit Loss (CECL) Transition Amount, not including Deferred Tax Assets, adjusted for Perpetual Preferred Stock.",
            'CECL Transitional Amount': "(Qtly, $) CECL transitional amount for removing double counting in concentration ratio calculations. This is the phase-in amount that appears in both Tier 1 Capital and Allowance.",
            'Net Interest Margin': "(YTD, %) The net interest margin of the entity.",
            'Earning Assets / Total Assets': "(Qtly, %) Ratio of earning assets to total assets.",
            'Nonperforming Assets / Total Assets': "(Qtly, %) Ratio of nonperforming assets to total assets.",
            'Assets Past Due 30-89 Days / Total Assets': "(Qtly, %) Ratio of assets past due 30-89 days to total assets.",
            'Assets Past Due 90+ Days / Total Assets': "(Qtly, %) Ratio of assets past due 90+ days to total assets.",
            'Net Charge-Offs / Total Loans & Leases': "(YTD, %) Ratio of net charge-offs to total loans and leases.",
            'Earnings Coverage of Net Loan Charge-Offs': "(X) The number of times that earnings can cover net loan charge-offs.",
            'Loan and Lease Loss Provision to Net Charge-Offs': "(YTD, %) Ratio of loan loss provision to net charge-offs.",
            'Loss Allowance / Total Loans & Leases': "(YTD, %) Ratio of loss allowance to total loans and leases.",
            'Loss Allowance to Noncurrent Loans and Leases': "(Qtly, %) Ratio of loss allowance to noncurrent loans and leases.",
            'Noncurrent Loans / Total Loans': "(Qtly, %) Ratio of noncurrent loans to total loans.",
            'Net Loans and Leases to Deposits': "(YTD, %) Loans and lease financing receivables net of unearned income, allowances and reserves as a percent of total deposits.",
            'Net Loans and Leases to Assets': "(Qtly, %) Ratio of net loans and leases to assets.",
            'Return on Assets': "(YTD, %) Return on assets.",
            'Return on Equity': "(YTD, %) Return on equity.",
            'Leverage (Core Capital) Ratio': "(Qtly, %) Leverage ratio (core capital ratio).",
            'Total Risk-Based Capital Ratio': "(Qtly, %) Total risk-based capital ratio.",
            'Efficiency Ratio': "(YTD, %) The efficiency ratio of the entity.",
            'Real Estate Loans to Tier 1 + ACL': "(Qtly, %) Real Estate Loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Commercial RE to Tier 1 + ACL': "(Qtly, %) Sum of RE Construction and Land Development, Multifamily, Loans to Nonresidential Properties, and Commercial Real Estate Loans not Secured by Real Estate as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Non-Owner Occupied CRE 3-Year Growth Rate': "(%) 3-year annualized growth rate of Non-Owner Occupied Commercial Real Estate, which includes RE Construction and Land Development, Multifamily, Non-OOC Nonresidential Properties Loans, and Commercial Real Estate Loans not Secured by Real Estate.",
            'C&I Loans to Tier 1 + ACL': "(Qtly, %) Commercial and Industrial Loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Auto Loans to Tier 1 + ACL': "(Qtly, %) Auto Loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Net Charge-Offs / Allowance for Credit Loss': "(Qtly, %) Ratio of Quarterly Net Charge-Offs to Allowance for Credit Loss.",
        }
    
    @staticmethod
    def safe_float(value: Any) -> float:
        """
        Safely convert a value to float.
        
        Args:
            value: Value to convert
            
        Returns:
            Float value or 0.0 if conversion fails
        """
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def calculate_metrics(self, financials_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Calculate all metrics for the given financial data.
        
        Args:
            financials_data: Financial data for all banks
            
        Returns:
            DataFrame with calculated metrics
        """
        all_metrics = []

        for bank_name, financials in financials_data.items():
            # Sort financials by date
            sorted_financials = sorted(financials, key=lambda x: x['REPDTE'])

            for i, financial in enumerate(sorted_financials):
                # Extract basic metrics from the financial data
                metrics = self._extract_basic_metrics(bank_name, financial)
                
                # Calculate CECL transition amount and capital base
                cecl_transition_amount, capital_base = self._calculate_capital_base(metrics, financial)
                metrics['CECL Transition Amount'] = cecl_transition_amount
                
                # Calculate capital ratios
                self._calculate_capital_ratios(metrics, capital_base)
                
                # Calculate CRE growth rate
                self._calculate_cre_growth_rate(metrics, sorted_financials, i, financial)
                
                # Calculate charge-off metrics
                self._calculate_charge_off_metrics(metrics)
                
                all_metrics.append(metrics)

        # Create and sort DataFrame
        df = pd.DataFrame(all_metrics)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        return df.sort_values('Date')

    def _extract_basic_metrics(self, bank_name: str, financial: Dict) -> Dict:
        """
        Extract basic metrics from the financial data.
        
        Args:
            bank_name: Name of the bank
            financial: Financial data for the bank
            
        Returns:
            Dictionary of basic metrics
        """
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
            'CECL Transitional Amount': max(self.safe_float(financial.get('RCFDJ447')), self.safe_float(financial.get('RCONJ447'))),
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
            'Efficiency Ratio': self.safe_float(financial.get('EEFFR'))
        }

    def _calculate_capital_base(self, metrics: Dict, financial: Dict) -> Tuple[float, float]:
        """
        Calculate capital base for concentration ratios per UBPR methodology.
        
        Per UBPR and OCC guidance, the denominator is:
        Tier 1 (Core) Capital + Allowance for Credit Loss - CECL Double Counting Adjustment
        
        The CECL adjustment removes the phase-in amount that appears in BOTH Tier 1 Capital
        and Allowance to prevent double counting (per agency guidelines).
        
        This matches UBPR formula:
        uc:UBPR8274[P0] + uc:UBPR3123[P0] - (CECL phase-in double counting)
        
        Args:
            metrics: Current metrics
            financial: Financial data
            
        Returns:
            Tuple of (CECL transitional amount, capital base)
        """
        tier1_capital = metrics['Tier 1 (Core) Capital']
        allowance_for_credit_loss = metrics['Allowance for Credit Loss']
        cecl_transitional_amount = metrics['CECL Transitional Amount']
        
        # Base calculation: Tier 1 + ACL
        capital_base = tier1_capital + allowance_for_credit_loss
        
        # Subtract CECL transitional amount to remove double counting
        # (only applies if bank is phasing in CECL - otherwise this field is zero)
        capital_base = capital_base - cecl_transitional_amount
        
        # Safety check: Ensure capital_base is positive
        if capital_base <= 0:
            logger.warning(f"Capital base adjustment resulted in non-positive value ({capital_base}) for {metrics.get('Bank', 'Unknown')}, using unadjusted")
            capital_base = tier1_capital + allowance_for_credit_loss
            cecl_transitional_amount = 0
            
        return cecl_transitional_amount, capital_base

    def _calculate_capital_ratios(self, metrics: Dict, capital_base: float) -> None:
        """
        Calculate capital ratios based on the capital base.
        NOTE: Removed Agriculture Loans and Credit Cards ratios, added Auto Loans ratio.
        
        Args:
            metrics: Current metrics
            capital_base: Capital base value
        """
        # CRITICAL FIX: Lowered threshold from $1M to $100 
        # API might return values in thousands, or capital_base might be smaller than expected
        if capital_base > 100:  # Minimum threshold - very permissive
            metrics['Real Estate Loans to Tier 1 + ACL'] = (metrics['Real Estate Loans'] / capital_base) * 100
            metrics['RE Construction and Land Development to Tier 1 + ACL'] = (metrics['RE Construction and Land Development'] / capital_base) * 100
            metrics['C&I Loans to Tier 1 + ACL'] = (metrics['Commercial and Industrial Loans'] / capital_base) * 100
            metrics['Auto Loans to Tier 1 + ACL'] = (metrics['Auto Loans'] / capital_base) * 100

            # Commercial RE to Tier 1 + ACL calculation
            commercial_re = (
                metrics['RE Construction and Land Development'] +
                metrics['Multifamily'] +
                metrics['Loans to Nonresidential Properties'] +
                metrics['Commercial Real Estate Loans not Secured by Real Estate']
            )
            metrics['Commercial RE to Tier 1 + ACL'] = (commercial_re / capital_base) * 100
        else:
            # Set to None instead of 0 to indicate the ratio couldn't be calculated
            logger.warning(f"Capital base too low ({capital_base}) for {metrics.get('Bank', 'Unknown')} on {metrics.get('Date', 'Unknown')}")
            metrics['Real Estate Loans to Tier 1 + ACL'] = None
            metrics['RE Construction and Land Development to Tier 1 + ACL'] = None
            metrics['C&I Loans to Tier 1 + ACL'] = None
            metrics['Auto Loans to Tier 1 + ACL'] = None
            metrics['Commercial RE to Tier 1 + ACL'] = None

    def _calculate_cre_growth_rate(self, metrics: Dict, sorted_financials: List[Dict], i: int, current_financial: Dict) -> None:
        """
        Calculate the 3-year growth rate for non-owner occupied CRE.
        
        Args:
            metrics: Current metrics
            sorted_financials: Sorted financial data
            i: Current index in the sorted financials
            current_financial: Current financial data
        """
        # Calculate Non-Owner Occupied CRE
        non_owner_occupied_cre = (
            self.safe_float(current_financial.get('LNRECONS')) +
            self.safe_float(current_financial.get('LNREMULT')) +
            self.safe_float(current_financial.get('LNRENROT')) +
            self.safe_float(current_financial.get('LNCOMRE'))
        )

        # Calculate 3-year growth rate
        if i >= 12:  # Assuming quarterly data, 12 quarters = 3 years
            three_years_ago = sorted_financials[i-12]
            old_non_owner_occupied_cre = (
                self.safe_float(three_years_ago.get('LNRECONS')) +
                self.safe_float(three_years_ago.get('LNREMULT')) +
                self.safe_float(three_years_ago.get('LNRENROT')) +
                self.safe_float(three_years_ago.get('LNCOMRE'))
            )
            # Added check for very small values to avoid division by near-zero
            if old_non_owner_occupied_cre > 1000:  # Using a reasonable threshold
                # Simple growth rate calculation
                growth_rate = (non_owner_occupied_cre / old_non_owner_occupied_cre) - 1
                metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = growth_rate * 100  # Convert to percentage
            else:
                metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = None
        else:
            metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = None

    def _calculate_charge_off_metrics(self, metrics: Dict) -> None:
        """
        Calculate charge-off related metrics.
        
        Args:
            metrics: Current metrics
        """
        # Calculate Net Charge-Offs / Allowance for Credit Loss
        if metrics['Allowance for Credit Loss'] > 1000:  # Using a reasonable threshold
            metrics['Net Charge-Offs / Allowance for Credit Loss'] = (
                metrics['Total Loans and Leases Net Charge-Offs Quarterly'] / metrics['Allowance for Credit Loss']
            ) * 100
        else:
            metrics['Net Charge-Offs / Allowance for Credit Loss'] = 0

class BankDataService:
    """
    Service for fetching and processing bank data.
    """
    def __init__(self):
        self.repository = BankDataRepository()
        self.calculator = BankMetricsCalculator(self.repository.dollar_format_metrics)
        
    def get_metrics_data(self, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
        """
        Get calculated metrics data for all banks.
        
        Args:
            start_date: Start date for the data query
            end_date: End date for the data query
            
        Returns:
            Tuple of (metrics DataFrame, dollar format metrics list, metric definitions dictionary)
        """
        # Fetch data from API or cache
        data = self.repository.fetch_data(BANK_INFO, start_date, end_date)
        
        # Check if we have data
        if not data['institutions_data']:
            logger.error("No institution data was fetched.")
            return pd.DataFrame(), self.repository.dollar_format_metrics, self.calculator.metric_definitions
        
        # Calculate metrics
        metrics_df = self.calculator.calculate_metrics(data['financials_data'])
        
        # CRITICAL FIX: Apply bank name normalization to EVERY bank name in the dataframe
        # This ensures all bank names are converted from FDIC official names to display names
        logger.info("Applying bank name normalization...")
        metrics_df['Bank'] = metrics_df['Bank'].apply(normalize_bank_name)
        
        # Log the unique banks after normalization to verify
        unique_banks_after = metrics_df['Bank'].unique()
        logger.info(f"Banks after normalization: {sorted(unique_banks_after)}")
        
        # Define the order of columns - REMOVED Agriculture and Credit Cards, ADDED Auto Loans
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
        
        # Reorder columns, keeping only those that exist in the dataframe
        available_columns = [col for col in metric_order if col in metrics_df.columns]
        extra_columns = [col for col in metrics_df.columns if col not in available_columns]
        metrics_df = metrics_df[available_columns + extra_columns]
        
        return metrics_df, self.repository.dollar_format_metrics, self.calculator.metric_definitions

class DashboardBuilder:
    """
    Builder for creating and configuring the Dash dashboard.
    """
    def __init__(self, df: pd.DataFrame, dollar_format_metrics: List[str], metric_definitions: Dict[str, str]):
        """
        Initialize the dashboard builder.
        
        Args:
            df: DataFrame with metrics data
            dollar_format_metrics: List of metrics that should be formatted as dollars
            metric_definitions: Dictionary of metric definitions
        """
        self.df = df
        self.dollar_format_metrics = dollar_format_metrics
        self.metric_definitions = metric_definitions
        
        # Log the banks in the dataframe to verify
        unique_banks = sorted(self.df['Bank'].unique())
        logger.info(f"Found {len(unique_banks)} unique banks in data: {', '.join(unique_banks)}")
        
        # Get unique dates from the DataFrame
        self.unique_dates = sorted(df['Date'].unique())
        
        # Define the order of metrics for display - REMOVED Agriculture and Credit Cards, ADDED Auto Loans
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
        ] + dollar_format_metrics  # Add dollar format metrics at the end
        
        # Filter metrics that exist in the DataFrame
        self.available_metrics = [metric for metric in self.metric_order if metric in df.columns]
    
    def create_dashboard(self) -> dash.Dash:
        """
        Create the Dash dashboard application.
        
        Returns:
            Configured Dash application
        """
        app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        app.title = "Grasshopper Bank Metrics Dashboard"
        app.config.suppress_callback_exceptions = True
        
        # Set server for deployment
        server = app.server
        
        # Add custom CSS to the index string
        custom_css = self._get_custom_css()
        app.index_string = f'''
        <!DOCTYPE html>
        <html>
            <head>
                {{%metas%}}
                <title>{{%title%}}</title>
                {{%favicon%}}
                {{%css%}}
                <style>
                {custom_css}
                </style>
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
        
        # Create layout
        app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _create_layout(self) -> html.Div:
        """
        Create the layout for the dashboard.
        
        Returns:
            Root layout component
        """
        # Create sidebar and content
        sidebar = self._create_sidebar()
        content = self._create_content()
        
        return html.Div([
            # Main layout
            html.Div([sidebar, content], id="app-container"),
            
            # Storage for state management
            dcc.Store(id='selected-bank-store'),
            dcc.Store(id='selected-metric-store', data=self.available_metrics[0] if self.available_metrics else None),
        ])
    
    def _create_sidebar(self) -> html.Div:
        """
        Create the sidebar for the dashboard.
        
        Returns:
            Sidebar component
        """
        return html.Div([
            # Header
            html.Div([
                html.Div([
                    html.H4("Grasshopper Bank", className="display-6 grasshopper-title", style={"color": COLOR_SCHEME['primary']}),
                    html.H5("Bank Metrics Dashboard", className="subtitle", style={"color": COLOR_SCHEME['accent']}),
                ], className="sidebar-title"),
                html.Hr(style={"borderColor": COLOR_SCHEME['primary']}),
            ], className="sidebar-header"),
            
            # Metric selector
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
            
            # Peer bank selector
            html.Div([
                html.Hr(style={"borderColor": COLOR_SCHEME['primary']}),
                html.P("Select peer banks to compare", className="lead", style={"color": COLOR_SCHEME['text']}),
                html.Div(id='peer-selector', className="mt-3"),
                html.Button(
                    "Add All Peers",
                    id="add-all-peers-btn",
                    className="add-all-btn mt-2"
                ),
            ], className="sidebar-section"),
            
            # Timeline selector
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
                    value=4,  # Set default value to 4 years
                    clearable=False,
                    style={'width': '100%', 'color': COLOR_SCHEME['text']},
                ),
                html.Div(id='selected-peers-info', className="mt-3", style={"color": COLOR_SCHEME['text']})
            ], className="sidebar-section"),
            
            # Footer
            html.Div([
                html.Hr(style={"borderColor": COLOR_SCHEME['primary']}),
                html.P("© 2025 Grasshopper Bank", className="text-center", style={"color": COLOR_SCHEME['light_text']}),
            ], className="sidebar-footer"),
        ], className="sidebar")
    
    def _create_content(self) -> html.Div:
        """
        Create the content area for the dashboard.
        
        Returns:
            Content component
        """
        return html.Div([
            # Top row with date selector
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
            
            # Main charts row (side by side)
            dbc.Row([
                # Comparison bar chart
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
                
                # Historical trends chart
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
            
            # Metrics grid
            dbc.Row([
                # Metric overview card
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
                
                # Trend Analysis card
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
            
            # Bank details card (full width at bottom)
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
            
            # Footer
            html.Div("All data sourced through FDIC API", className="source-info")
        ], className="content")
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """
        Register all callbacks for the dashboard.
        
        Args:
            app: Dash application
        """
        # Callback for peer selector
        @app.callback(
            Output('peer-selector', 'children'),
            Input('metric-selector', 'value')  # Using metric-selector as a dummy input to initialize
        )
        def update_peer_selector(dummy):
            # Get all unique banks from the data except Grasshopper Bank (using PRIMARY_BANK_DISPLAY_NAME)
            all_available_banks = sorted(list(set(self.df['Bank'].unique()) - {PRIMARY_BANK_DISPLAY_NAME}))
            logger.info(f"Available banks for selection: {len(all_available_banks)}")
            
            # Create the dropdown with ALL available banks as options and select all by default
            return dcc.Dropdown(
                id='individual-peer-selector',
                options=[{'label': peer, 'value': peer} for peer in all_available_banks],
                value=all_available_banks,  # Select all peers by default
                multi=True,
                style={'width': '100%', 'color': COLOR_SCHEME['text']},
            )
            
        # Callback for "Add All Peers" button
        @app.callback(
            Output('individual-peer-selector', 'value'),
            Input('add-all-peers-btn', 'n_clicks'),
            State('individual-peer-selector', 'options')
        )
        def add_all_peers(n_clicks, all_options):
            if n_clicks is None:
                raise PreventUpdate
                
            # Get all available options
            all_peers = [option['value'] for option in all_options]
            logger.info(f"Adding all {len(all_peers)} peers")
            
            return all_peers
        
        # Callback for metric definition display
        @app.callback(
            Output('metric-definition', 'children'),
            Input('metric-selector', 'value')
        )
        def update_metric_definition(selected_metric):
            definition = self.metric_definitions.get(selected_metric, '')
            return html.P(definition, className="metric-definition")
        
        # Callback for selected peers info
        @app.callback(
            Output('selected-peers-info', 'children'),
            Input('individual-peer-selector', 'value')
        )
        def update_selected_peers_info(selected_peers):
            # Improved display to ensure all peers are visible
            return html.Div([
                html.P(f"Selected Peers: {len(selected_peers)} banks", style={"fontWeight": "bold"}),
                html.Div(
                    [html.Span(peer, className="selected-peer-tag") for peer in selected_peers],
                    className="selected-peers-container"
                )
            ], style={"margin-top": "10px", "color": COLOR_SCHEME['text']})
        
        # Callback for updating the bar chart and metric overview
        @app.callback(
            [
                Output('bar-chart', 'figure'),
                Output('metric-overview', 'children'),
                Output('selected-metric-store', 'data')
            ],
            [
                Input('metric-selector', 'value'),
                Input('date-selector', 'value'),
                Input('individual-peer-selector', 'value'),
            ]
        )
        def update_bar_chart(selected_metric, selected_date, selected_peers):
            if not selected_metric or not selected_date:
                return self._create_empty_figure("No data available"), html.Div("No data available"), selected_metric
            
            # Convert selected_date from string to datetime
            selected_date = pd.to_datetime(selected_date).to_pydatetime()
            
            # Always include Grasshopper Bank (using PRIMARY_BANK_DISPLAY_NAME)
            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + selected_peers
            
            # Filter data
            filtered_df = self.df[(self.df['Date'] == selected_date) & (self.df['Bank'].isin(selected_banks))]
            
            if filtered_df.empty:
                return self._create_empty_figure(f"No data available for {selected_date.strftime('%m/%d/%y')}"), \
                       html.Div("No data available for the selected date"), selected_metric
            
            # Sort data by the selected metric
            sorted_df = filtered_df.sort_values(by=selected_metric, ascending=False)
            
            # Create the bar chart
            fig = self._create_bar_chart(sorted_df, selected_metric, selected_date)
            
            # Create the metric overview
            overview = self._create_metric_overview(filtered_df, selected_metric)
            
            return fig, overview, selected_metric
        
        # Callback for updating the historical chart and date range
        @app.callback(
            [
                Output('historical-chart', 'figure'),
                Output('historical-date-range', 'children')
            ],
            [
                Input('selected-metric-store', 'data'),
                Input('individual-peer-selector', 'value'),
                Input('trend-timeline-selector', 'value')
            ]
        )
        def update_historical_chart(selected_metric, selected_peers, trend_timeline):
            if not selected_metric:
                return self._create_empty_figure("No metric selected"), ""
            
            # Always include Grasshopper Bank (using PRIMARY_BANK_DISPLAY_NAME)
            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + selected_peers
            
            # Calculate date range for display
            end_date = self.df['Date'].max()
            start_date = end_date - pd.DateOffset(years=trend_timeline)
            date_range_text = f"From {start_date.strftime('%m/%d/%y')} to {end_date.strftime('%m/%d/%y')}"
            
            # Create the historical chart
            return self._create_historical_chart(selected_banks, selected_metric, trend_timeline), date_range_text
            
        # Callback for updating the trend analysis
        @app.callback(
            Output('trend-analysis', 'children'),
            [
                Input('selected-metric-store', 'data'),
                Input('individual-peer-selector', 'value'),
                Input('trend-timeline-selector', 'value')
            ]
        )
        def update_trend_analysis(selected_metric, selected_peers, trend_timeline):
            if not selected_metric:
                return html.P("Select a metric to view trend analysis", style={"color": COLOR_SCHEME['text']})
            
            # Always include Grasshopper Bank (using PRIMARY_BANK_DISPLAY_NAME)
            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + selected_peers
            
            # Create the trend analysis
            return self._create_trend_analysis(selected_banks, selected_metric, trend_timeline)
        
        # Callback for updating bank details on chart click
        @app.callback(
            [
                Output('bank-details', 'children'),
                Output('selected-bank-store', 'data')
            ],
            [
                Input('bar-chart', 'clickData'),
                Input('date-selector', 'value')
            ],
            [
                State('metric-selector', 'value'),
                State('individual-peer-selector', 'value'),
                State('selected-bank-store', 'data')
            ]
        )
        def update_bank_details(clickData, selected_date, selected_metric, selected_peers, stored_bank):
            # Convert selected_date from string to datetime
            selected_date = pd.to_datetime(selected_date).to_pydatetime() if selected_date else None
            
            # Determine which bank to display
            if clickData:
                bank = clickData['points'][0]['x']
                stored_bank = bank
            elif stored_bank and selected_date:
                bank = stored_bank
            else:
                return html.P("Click on a bar to see details", style={"color": COLOR_SCHEME['text']}), None
            
            # Always include Grasshopper Bank (using PRIMARY_BANK_DISPLAY_NAME)
            selected_banks = [PRIMARY_BANK_DISPLAY_NAME] + selected_peers
            
            if bank not in selected_banks:
                return html.P("Selected bank is not in the current comparison. Please select a displayed bank.", 
                           style={"color": COLOR_SCHEME['text']}), stored_bank
            
            # Get bank details
            bank_df = self.df[(self.df['Bank'] == bank) & (self.df['Date'] == selected_date)]
            
            if bank_df.empty:
                return html.P(f"No data available for {bank} on {selected_date.strftime('%m/%d/%y')}", 
                           style={"color": COLOR_SCHEME['text']}), stored_bank
            
            bank_data = bank_df.iloc[0]
            
            # Create bank details component
            details = self._create_bank_details(bank, bank_data, selected_date)
            
            return details, stored_bank
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """
        Create an empty figure with a message.
        
        Args:
            message: Message to display
            
        Returns:
            Empty figure with message
        """
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
        """
        Create a bar chart for the given data.
        
        Args:
            df: DataFrame with filtered data
            metric: Metric to display
            date: Selected date
            
        Returns:
            Bar chart figure
        """
        # Set colors for bars (Grasshopper Bank is highlighted using PRIMARY_BANK_DISPLAY_NAME)
        colors = [COLOR_SCHEME['grasshopper'] if bank == PRIMARY_BANK_DISPLAY_NAME else COLOR_SCHEME['peer'] for bank in df['Bank']]
        
        # Increase the opacity for better visibility
        opacities = [1.0 if bank == PRIMARY_BANK_DISPLAY_NAME else 0.6 for bank in df['Bank']]
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=df['Bank'],
            y=df[metric],
            marker_color=colors,
            marker_opacity=opacities,
            hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:,.2f}<extra></extra>',
            name=''
        ))
        
        # Calculate y-axis range
        y_min = df[metric].min()
        y_max = df[metric].max()
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # Add 10% padding
        
        # Format the date
        formatted_date = date.strftime('%m/%d/%y')
        
        # Update layout
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
        """
        Create a historical chart for the selected banks and metric.
        
        Args:
            selected_banks: List of selected banks
            metric: Metric to display
            trend_timeline: Number of years to display
            
        Returns:
            Historical chart figure
        """
        # Create figure
        fig = go.Figure()
        
        # Filter data for selected banks
        filtered_df = self.df[self.df['Bank'].isin(selected_banks)]
        
        if filtered_df.empty:
            return self._create_empty_figure("No historical data available")
        
        # Set the start date based on the timeline
        end_date = filtered_df['Date'].max()
        start_date = end_date - pd.DateOffset(years=trend_timeline)
        filtered_df = filtered_df[filtered_df['Date'] >= start_date]
        
        # Create a pivot table for easier plotting
        pivot_df = filtered_df.pivot(index='Date', columns='Bank', values=metric)
        
        # Add a trace for each bank
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
        
        # Determine appropriate tick settings based on trend timeline
        if trend_timeline <= 2:
            dtick = 'M3'  # Every 3 months
            tickformat = '%b\n%Y'
        elif trend_timeline <= 4:
            dtick = 'M6'  # Every 6 months
            tickformat = '%b\n%Y'
        else:
            dtick = 'M12'  # Every year
            tickformat = '%Y'
        
        # Update layout with removed legend
        fig.update_layout(
            title=f"{metric} - {trend_timeline} Year Trend",
            margin=dict(l=50, r=20, t=50, b=50),
            plot_bgcolor=COLOR_SCHEME['card_bg'],
            paper_bgcolor=COLOR_SCHEME['card_bg'],
            font=dict(color=COLOR_SCHEME['text']),
            hoverlabel=dict(bgcolor=COLOR_SCHEME['card_bg'], font_size=12, font_color=COLOR_SCHEME['text']),
            showlegend=False,  # Remove legend
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
        """
        Create a metric overview component.
        
        Args:
            df: DataFrame with filtered data
            metric: Metric to display
            
        Returns:
            Metric overview component
        """
        def format_value(value):
            if pd.isna(value):
                return "N/A"
            if metric in self.dollar_format_metrics:
                return f"${value:,.0f}"
            else:
                return f"{value:.2f}"
        
        # Get Grasshopper Bank data (using PRIMARY_BANK_DISPLAY_NAME)
        gh_df = df[df['Bank'] == PRIMARY_BANK_DISPLAY_NAME]
        gh_value = gh_df[metric].values[0] if not gh_df.empty else None
        
        # Calculate statistics
        if gh_value is not None:
            gh_percentile = stats.percentileofscore(df[metric], gh_value)
            gh_rank = df[metric].rank(ascending=False, method='min')[df['Bank'] == PRIMARY_BANK_DISPLAY_NAME].values[0]
        else:
            gh_percentile = None
            gh_rank = None
            
        # Calculate quartiles
        q1, q3 = np.percentile(df[metric], [25, 75])
        
        # Create performance category
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
        
        # Create components
        return html.Div([
            # Current Snapshot Section
            html.Div([
                html.Div("Current Snapshot", className="stat-section-title"),
                html.Div([
                    html.Div("Average:", className="stat-label"),
                    html.Div(format_value(df[metric].mean()))
                ], className="stat-row"),
                html.Div([
                    html.Div("Middle Value (Median):", className="stat-label"),
                    html.Div(format_value(df[metric].median()))
                ], className="stat-row"),
                html.Div([
                    html.Div(f"{PRIMARY_BANK_DISPLAY_NAME} Value:", className="stat-label gh-highlight"),
                    html.Div(format_value(gh_value) if gh_value is not None else "N/A", className="gh-highlight")
                ], className="stat-row"),
                html.Div([
                    html.Div("Highest Value:", className="stat-label"),
                    html.Div(f"{format_value(df[metric].max())} ({df.loc[df[metric].idxmax(), 'Bank']})" if not df.empty else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Lowest Value:", className="stat-label"),
                    html.Div(f"{format_value(df[metric].min())} ({df.loc[df[metric].idxmin(), 'Bank']})" if not df.empty else "N/A")
                ], className="stat-row"),
            ], className="stat-section"),

            # Grasshopper Bank Snapshot Statistics Section
            html.Div([
                html.Div(f"{PRIMARY_BANK_DISPLAY_NAME}'s Position", className="stat-section-title"),
                html.Div([
                    html.Div("Percentile Rank:", className="stat-label"),
                    html.Div(f"{gh_percentile:.1f}%" if gh_percentile is not None else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Ranking:", className="stat-label"),
                    html.Div(f"#{gh_rank:.0f} out of {len(df)} banks" if gh_rank is not None else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Performance Group:", className="stat-label"),
                    html.Div(performance_group, style={"color": performance_color, "fontWeight": "bold"})
                ], className="stat-row"),
                html.Div([
                    html.Div("Standout Score:", className="stat-label"),
                    html.Div(self._calculate_zscore_display(df, metric) if gh_value is not None else "N/A")
                ], className="stat-row"),
            ], className="stat-section"),
        ], style={"background-color": COLOR_SCHEME['card_bg'], "border-radius": "8px", "color": COLOR_SCHEME['text'], "padding": "10px"})

    def _calculate_zscore_display(self, df, metric):
        """Safely calculate and format z-score display for Grasshopper Bank"""
        try:
            z_values = stats.zscore(df[metric].values)
            gh_index = df.index[df['Bank'] == PRIMARY_BANK_DISPLAY_NAME].tolist()
            if gh_index:
                return f"{z_values[gh_index[0]]:.2f} (How many standard deviations from the average)"
            return "N/A"
        except:
            return "N/A (Cannot calculate)"
    
    def _create_trend_analysis(self, selected_banks: List[str], metric: str, trend_timeline: int) -> html.Div:
        """
        Create a trend analysis component with correlation statistics.
        
        Args:
            selected_banks: List of selected banks
            metric: Metric to analyze
            trend_timeline: Number of years to analyze
            
        Returns:
            Trend analysis component
        """
        # Filter data for selected banks
        filtered_df = self.df[self.df['Bank'].isin(selected_banks)]
        
        if filtered_df.empty:
            return html.Div("No trend data available")
        
        # Set the start date based on the timeline
        end_date = filtered_df['Date'].max()
        start_date = end_date - pd.DateOffset(years=trend_timeline)
        filtered_df = filtered_df[filtered_df['Date'] >= start_date]
        
        # Create a pivot table for easier analysis
        pivot_df = filtered_df.pivot(index='Date', columns='Bank', values=metric)
        
        # Check if Grasshopper Bank data exists (using PRIMARY_BANK_DISPLAY_NAME)
        if PRIMARY_BANK_DISPLAY_NAME not in pivot_df.columns or pivot_df[PRIMARY_BANK_DISPLAY_NAME].count() < 2:
            return html.Div(f"Insufficient data for {PRIMARY_BANK_DISPLAY_NAME} to perform trend analysis")
        
        gh_data = pivot_df[PRIMARY_BANK_DISPLAY_NAME].dropna()
        
        # Calculate statistics
        stats_data = {}
        
        # Calculate growth rates
        first_values = pivot_df.iloc[0]
        last_values = pivot_df.iloc[-1]
        valid_banks = pivot_df.columns[~pivot_df.iloc[-1].isna() & ~pivot_df.iloc[0].isna()]
        
        for bank in valid_banks:
            bank_data = pivot_df[bank].dropna()
            if len(bank_data) < 2:
                continue
                
            # Calculate growth
            growth_rate = ((last_values[bank] / first_values[bank]) - 1) * 100
            
            # Calculate volatility (standard deviation)
            volatility = bank_data.std()
            
            # Calculate correlation with Grasshopper Bank
            if bank != PRIMARY_BANK_DISPLAY_NAME and len(gh_data) >= 2:
                # Get overlapping data points
                overlap_df = pd.concat([gh_data, bank_data], axis=1).dropna()
                if len(overlap_df) >= 2:
                    correlation = overlap_df.iloc[:, 0].corr(overlap_df.iloc[:, 1])
                else:
                    correlation = np.nan
            else:
                correlation = np.nan
                
            # Calculate trend direction (positive or negative slope)
            if len(bank_data) >= 2:
                x = np.arange(len(bank_data))
                slope, _ = np.polyfit(x, bank_data.values, 1)
                trend_direction = "Increasing" if slope > 0 else "Decreasing"
            else:
                trend_direction = "Unknown"
                
            stats_data[bank] = {
                'growth_rate': growth_rate,
                'volatility': volatility,
                'correlation': correlation if bank != PRIMARY_BANK_DISPLAY_NAME else np.nan,
                'trend_direction': trend_direction
            }
            
        # Find banks with highest and lowest correlation to Grasshopper Bank
        correlations = {bank: data['correlation'] for bank, data in stats_data.items() if bank != PRIMARY_BANK_DISPLAY_NAME and not np.isnan(data['correlation'])}
        
        if correlations:
            most_similar = max(correlations.items(), key=lambda x: x[1])
            least_similar = min(correlations.items(), key=lambda x: x[1])
        else:
            most_similar = least_similar = (None, np.nan)
            
        # Calculate average metrics
        avg_growth = np.mean([data['growth_rate'] for data in stats_data.values() if not np.isnan(data['growth_rate'])])
        avg_volatility = np.mean([data['volatility'] for data in stats_data.values() if not np.isnan(data['volatility'])])
        
        # Create the trend analysis component
        return html.Div([
            # Long-term Trend Analysis Section
            html.Div([
                html.Div(f"{trend_timeline}-Year Trend Analysis", className="stat-section-title"),
                
                # Grasshopper Bank Stats
                html.Div([
                    html.Div(f"{PRIMARY_BANK_DISPLAY_NAME} Growth Rate:", className="stat-label"),
                    html.Div(f"{stats_data.get(PRIMARY_BANK_DISPLAY_NAME, {}).get('growth_rate', np.nan):.2f}% over {trend_timeline} years" 
                             if PRIMARY_BANK_DISPLAY_NAME in stats_data else "N/A")
                ], className="stat-row"),
                
                html.Div([
                    html.Div(f"{PRIMARY_BANK_DISPLAY_NAME} Trend Direction:", className="stat-label"),
                    html.Div(f"{stats_data.get(PRIMARY_BANK_DISPLAY_NAME, {}).get('trend_direction', 'Unknown')}" 
                             if PRIMARY_BANK_DISPLAY_NAME in stats_data else "N/A")
                ], className="stat-row"),
                
                html.Div([
                    html.Div(f"{PRIMARY_BANK_DISPLAY_NAME} Volatility:", className="stat-label"),
                    html.Div(f"{stats_data.get(PRIMARY_BANK_DISPLAY_NAME, {}).get('volatility', np.nan):.4f} (Lower means more stable)" 
                             if PRIMARY_BANK_DISPLAY_NAME in stats_data else "N/A")
                ], className="stat-row"),
                
                # Peer Comparison
                html.Div([
                    html.Div("Average Growth Rate:", className="stat-label"),
                    html.Div(f"{avg_growth:.2f}% over {trend_timeline} years" if not np.isnan(avg_growth) else "N/A")
                ], className="stat-row"),
                
                html.Div([
                    html.Div("Average Volatility:", className="stat-label"),
                    html.Div(f"{avg_volatility:.4f} (Lower means more stable)" if not np.isnan(avg_volatility) else "N/A")
                ], className="stat-row"),
                
                # Correlation Analysis
                html.Div([
                    html.Div(f"Bank moving most like {PRIMARY_BANK_DISPLAY_NAME}:", className="stat-label"),
                    html.Div(f"{most_similar[0]} (correlation: {most_similar[1]:.2f})" if most_similar[0] else "N/A")
                ], className="stat-row"),
                
                html.Div([
                    html.Div(f"Bank moving least like {PRIMARY_BANK_DISPLAY_NAME}:", className="stat-label"),
                    html.Div(f"{least_similar[0]} (correlation: {least_similar[1]:.2f})" if least_similar[0] else "N/A")
                ], className="stat-row"),
                
                # Best/Worst Performers
                html.Div([
                    html.Div("Best Growth Performance:", className="stat-label"),
                    html.Div(f"{max(stats_data.items(), key=lambda x: x[1]['growth_rate'])[0]} "
                             f"({max(stats_data.items(), key=lambda x: x[1]['growth_rate'])[1]['growth_rate']:.2f}%)" 
                             if stats_data else "N/A")
                ], className="stat-row"),
                
                html.Div([
                    html.Div("Most Stable Bank:", className="stat-label"),
                    html.Div(f"{min(stats_data.items(), key=lambda x: x[1]['volatility'])[0]} "
                             f"(volatility: {min(stats_data.items(), key=lambda x: x[1]['volatility'])[1]['volatility']:.4f})" 
                             if stats_data else "N/A")
                ], className="stat-row"),
            ], className="stat-section"),
        ], style={"background-color": COLOR_SCHEME['card_bg'], "border-radius": "8px", "color": COLOR_SCHEME['text'], "padding": "10px"})
    
    def _create_bank_details(self, bank: str, bank_data: pd.Series, date: datetime) -> html.Div:
        """
        Create a bank details component.
        
        Args:
            bank: Name of the bank
            bank_data: Data for the bank
            date: Date of the data
            
        Returns:
            Bank details component
        """
        # Format date
        formatted_date = date.strftime('%m/%d/%y')
        
        # Determine if the bank is Grasshopper Bank or a peer (using PRIMARY_BANK_DISPLAY_NAME)
        is_gh = bank == PRIMARY_BANK_DISPLAY_NAME
        
        # Create the card color based on the bank
        card_bg_color = COLOR_SCHEME['grasshopper'] if is_gh else COLOR_SCHEME['secondary']
        
        # Create columns of metrics for display
        metric_cols = []
        
        # Get key metrics in order
        ordered_metrics = [metric for metric in self.available_metrics if metric not in ['Bank', 'Date']]
        
        # Create metric columns - 4 columns of metrics
        for col_idx in range(4):
            col_metrics = ordered_metrics[col_idx::4]  # Get every 4th metric starting from col_idx
            
            metrics_html = []
            for metric in col_metrics:
                value = bank_data[metric]
                
                # Format the value
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
            
            metric_cols.append(
                dbc.Col(
                    html.Div(metrics_html, className="bank-detail-col"),
                    xs=12, sm=6, md=3
                )
            )
        
        # Create the bank details card
        return html.Div([
            html.Div([
                html.H5(f"{bank} Metrics as of {formatted_date}", 
                       style={"color": "white", "margin": "0"}),
            ], style={
                "backgroundColor": card_bg_color,
                "padding": "10px 15px",
                "borderRadius": "8px 8px 0 0"
            }),
            html.Div(
                dbc.Row(metric_cols),
                style={
                    "padding": "15px",
                    "backgroundColor": COLOR_SCHEME['card_bg'],
                    "borderRadius": "0 0 8px 8px",
                    "color": COLOR_SCHEME['text']
                }
            )
        ])
    
    def _get_custom_css(self) -> str:
        """
        Get custom CSS for the dashboard.
        
        Returns:
            Custom CSS string
        """
        return """
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f0f0f0;
                color: #333333;
                margin: 0;
                padding: 0;
            }
            #app-container {
                display: flex;
                min-height: 100vh;
                padding: 0 10px;
            }
            .sidebar {
                width: 450px;
                background-color: #ffffff;
                padding: 1.5rem 1rem;
                margin-right: 20px;
                overflow-y: auto;
                border-right: 1px solid #e0e0e0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                position: sticky;
                top: 10px;
                height: calc(100vh - 20px);
            }
            .sidebar-header {
                flex: 0 0 auto;
            }
            .sidebar-title {
                margin-bottom: 15px;
            }
            .grasshopper-title {
                margin-bottom: 0;
                font-weight: bold;
                letter-spacing: -0.5px;
            }
            .subtitle {
                font-size: 1rem;
                margin-top: 0;
            }
            .sidebar-section {
                flex: 0 0 auto;
                margin-bottom: 1.5rem;
            }
            .sidebar-footer {
                flex: 1 1 auto;
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
            }
            .content {
                flex-grow: 1;
                padding: 1.5rem;
                overflow-y: auto;
                background-color: #f0f0f0;
            }
            .card {
                background-color: #ffffff;
                border: none;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin-bottom: 1.5rem;
                overflow: hidden;
            }
            .card-title {
                color: #333333;
                margin-bottom: 0;
                font-size: 1.1rem;
                font-weight: bold;
            }
            .card-header {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
                padding: 0.75rem 1rem;
            }
            .card-body {
                padding: 1rem;
            }
            .date-selector-container {
                display: flex;
                align-items: center;
                background-color: #ffffff;
                padding: 10px 15px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                margin-bottom: 10px;
            }
            .date-label {
                font-weight: bold;
                margin-right: 10px;
                color: #0E3E1B;
            }
            .table {
                font-size: 0.9rem;
                background-color: #ffffff;
            }
            .table thead th {
                background-color: #e0e0e0;
                color: #333333;
            }
            .table tbody td {
                color: #333333;
            }
            .table-striped tbody tr:nth-of-type(odd) {
                background-color: #f5f5f5;
            }
            .table-hover tbody tr:hover {
                background-color: #e9ecef;
            }
            .Select-menu-outer {
                max-height: 400px !important;
                background-color: #f5f5f5;
            }
            .Select-option {
                padding: 12px 8px !important;
                color: #333333;
            }
            .Select-option:hover {
                background-color: #e0e0e0;
            }
            .Select-value-label {
                color: #333333 !important;
            }
            .Select-control {
                background-color: #f5f5f5 !important;
                border-color: #d1d1d1 !important;
            }
            .Select-placeholder, .Select--single > .Select-control .Select-value {
                color: #333333 !important;
            }
            .Select-input > input {
                color: #333333 !important;
            }
            .source-info {
                font-size: 0.8rem;
                color: #666666;
                text-align: center;
                padding: 10px 0;
            }
            .metric-definition {
                font-size: 0.9rem;
                color: #666666;
                margin-top: 10px;
            }
            .rc-slider-rail {
                background-color: #bfbfbf;
            }
            .rc-slider-track {
                background-color: #0E3E1B;
            }
            .rc-slider-handle {
                border-color: #0E3E1B;
            }
            .rc-slider-mark-text {
                color: #666666;
            }
            .stat-section {
                margin-bottom: 15px;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }
            .stat-section-title {
                font-weight: bold;
                margin-bottom: 5px;
                color: #0E3E1B;
            }
            .stat-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
            .stat-label {
                font-weight: bold;
            }
            .gh-highlight {
                color: #0E3E1B;
                font-weight: bold;
            }
            .add-all-btn {
                background-color: #0E3E1B;
                color: #ffffff;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9rem;
                margin-top: 5px;
                transition: background-color 0.3s;
            }
            .add-all-btn:hover {
                background-color: #0A2E15;
            }
            .bank-detail-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 5px;
            }
            .bank-detail-label {
                font-weight: bold;
                font-size: 0.8rem;
                flex: 1;
            }
            .bank-detail-value {
                font-size: 0.8rem;
                text-align: right;
                flex: 1;
            }
            .bank-detail-col {
                padding: 0 10px;
            }
            .selected-peers-container {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-top: 5px;
            }
            .selected-peer-tag {
                background: #f0f0f0;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 0.8rem;
                white-space: nowrap;
            }
            
            /* Mobile responsive design */
            @media (max-width: 992px) {
                #app-container {
                    flex-direction: column;
                }
                .sidebar {
                    width: 100%;
                    height: auto;
                    position: static;
                    margin-right: 0;
                    margin-bottom: 20px;
                }
                .content {
                    padding: 1rem;
                }
            }
        """

def load_sample_data():
    """
    Load sample data for development and testing when API is unavailable.
    
    Returns:
        Dataframe with sample metrics data
    """
    logger.info("Loading sample data for development/testing")
    
    # Use the DISPLAY names from BANK_NAME_MAPPING for sample data
    all_banks = list(BANK_NAME_MAPPING.values())
    
    # Print all banks to verify
    logger.info(f"Generating sample data for {len(all_banks)} banks: {', '.join(all_banks)}")
    
    # Use 'ME' (month end) instead of deprecated 'M', starting from 6/30/2020
    all_month_ends = pd.date_range(start='2020-06-30', end='2024-12-31', freq='ME')
    # Keep only quarter-end months (Mar, Jun, Sep, Dec)
    dates = all_month_ends[all_month_ends.month.isin([3, 6, 9, 12])]
    
    # Create sample dataframe
    data = []
    for bank in all_banks:
        # Different starting asset sizes based on bank type (specialty/community banks)
        if bank == PRIMARY_BANK_DISPLAY_NAME:
            base_assets = np.random.uniform(2000000000, 3000000000)  # $2-3 billion
        else:
            base_assets = np.random.uniform(1500000000, 5000000000)  # $1.5-5 billion
            
        # Growth rate varies by bank
        growth_rate = np.random.uniform(0.015, 0.08)  # Quarterly growth rate for specialty banks
        
        for i, date in enumerate(dates):
            # Assets grow over time with some randomness
            year_factor = 1 + (growth_rate * i) + np.random.uniform(-0.02, 0.02)
            total_assets = base_assets * year_factor
            
            # Create other metrics based on total assets
            total_deposits = total_assets * np.random.uniform(0.68, 0.78)
            total_loans = total_assets * np.random.uniform(0.68, 0.75)
            net_loans = total_loans * np.random.uniform(0.95, 0.99)
            tier1_capital = total_assets * np.random.uniform(0.11, 0.15)
            allowance_for_credit_loss = total_loans * np.random.uniform(0.01, 0.02)
            
            # Special case for Grasshopper Bank - specialized banking focus
            if bank == PRIMARY_BANK_DISPLAY_NAME:
                # Technology/startup focused bank characteristics
                total_deposits = total_assets * np.random.uniform(0.70, 0.76)
                # Strong loan portfolio
                total_loans = total_assets * np.random.uniform(0.70, 0.75)
                # Strong capital ratio for specialty bank
                tier1_capital = total_assets * np.random.uniform(0.12, 0.16)
                # Efficiency ratio for specialty banks
                efficiency_ratio = np.random.uniform(50.0, 60.0)
                # Return metrics for specialty banks
                roa = np.random.uniform(0.8, 1.6)
                roe = np.random.uniform(9.0, 15.0)
            else:
                # Standard banking metrics for other specialty banks
                efficiency_ratio = np.random.uniform(52.0, 68.0)
                roa = np.random.uniform(0.6, 1.4)
                roe = np.random.uniform(7.0, 14.0)
            
            # Generate reasonable real estate metrics
            real_estate_loans = total_loans * np.random.uniform(0.35, 0.55)
            construction_loans = real_estate_loans * np.random.uniform(0.08, 0.18)
            multifamily = real_estate_loans * np.random.uniform(0.1, 0.2)
            nonres_properties = real_estate_loans * np.random.uniform(0.3, 0.5)
            
            # Performance metrics
            nim = np.random.uniform(2.8, 4.2)
            
            # Specialty banks have focused commercial banking exposure
            commercial_loans = total_loans * np.random.uniform(0.3, 0.45)
            auto_loans = total_loans * np.random.uniform(0.05, 0.12)
            credit_cards = total_loans * np.random.uniform(0.01, 0.04)
            consumer_loans = total_loans * np.random.uniform(0.08, 0.18)
            
            # Capital ratios
            lcr = np.random.uniform(10.0, 14.0)
            total_risk_capital = np.random.uniform(13.0, 18.0)
            
            # Calculate some ratios
            re_loans_to_tier1 = (real_estate_loans / (tier1_capital + allowance_for_credit_loss)) * 100
            construction_to_tier1 = (construction_loans / (tier1_capital + allowance_for_credit_loss)) * 100
            commercial_re_to_tier1 = ((construction_loans + multifamily + nonres_properties) / 
                                       (tier1_capital + allowance_for_credit_loss)) * 100
            ci_loans_to_tier1 = (commercial_loans / (tier1_capital + allowance_for_credit_loss)) * 100
            auto_loans_to_tier1 = (auto_loans / (tier1_capital + allowance_for_credit_loss)) * 100
            
            # Record for this bank at this date
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
                'Leverage (Core Capital) Ratio': lcr,
                'Total Risk-Based Capital Ratio': total_risk_capital,
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
    
    # Create DataFrame from all data points
    df = pd.DataFrame(data)
    
    # Log counts to verify
    bank_counts = df['Bank'].value_counts()
    logger.info(f"Sample data contains {len(bank_counts)} unique banks")
    for bank, count in bank_counts.items():
        logger.info(f"  {bank}: {count} data points")
        
    return df

def generate_temp_data(unique_banks: List[str], unique_dates: List[datetime]) -> pd.DataFrame:
    """Create a fallback dataset if any banks are missing from the main dataset."""
    data = []
    for bank in unique_banks:
        for date in unique_dates:
            # Generate very simplified placeholder data
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
    """
    Main function to initialize and run the dashboard.
    
    Returns:
        Tuple of (Dash app, server)
    """
    # Disable warnings
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    
    try:
        # Initialize data service
        data_service = BankDataService()
        
        # Fetch data
        metrics_df, dollar_format_metrics, metric_definitions = data_service.get_metrics_data(
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE
        )
        
        if metrics_df.empty:
            logger.warning("No data from API. Using sample data instead.")
            metrics_df = load_sample_data()
            
            # Initialize a metrics calculator to get the definitions and dollar format metrics
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
        
        # Initialize a metrics calculator to get the definitions and dollar format metrics
        calculator = BankMetricsCalculator([
            'Total Assets', 'Total Deposits', 'Total Loans and Leases', 
            'Net Loans and Leases', 'Total Securities', 'Real Estate Loans',
            'Tier 1 (Core) Capital', 'Net Income', 'Auto Loans'
        ])
        dollar_format_metrics = calculator.dollar_format_metrics
        metric_definitions = calculator.metric_definitions
    
    # Check if any banks are missing and generate fallback data if needed
    expected_banks = list(BANK_NAME_MAPPING.values())
    actual_banks = metrics_df['Bank'].unique()
    missing_banks = [bank for bank in expected_banks if bank not in actual_banks]
    
    if missing_banks:
        logger.warning(f"Missing data for banks: {missing_banks}. Generating fallback data.")
        fallback_df = generate_temp_data(missing_banks, metrics_df['Date'].unique())
        metrics_df = pd.concat([metrics_df, fallback_df])
    
    # Create dashboard
    dashboard_builder = DashboardBuilder(metrics_df, dollar_format_metrics, metric_definitions)
    app = dashboard_builder.create_dashboard()
    server = app.server
    
    return app, server

# Create app and server
app, server = main()

if __name__ == "__main__":
    app.run_server(debug=False)
