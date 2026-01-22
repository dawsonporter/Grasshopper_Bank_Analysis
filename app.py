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
DEFAULT_START_DATE = '20200630'  # June 30, 2020
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
