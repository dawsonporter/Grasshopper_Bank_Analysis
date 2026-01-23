# Grasshopper Bank Metrics Dashboard
[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-purple)](https://grasshopper-bank-dashboard-7b37a160d5b8.herokuapp.com/)

An interactive analytics dashboard for benchmarking **Grasshopper Bank** against a curated set of **digital / specialty bank peers** using **FDIC quarterly financial data**. Built with **Python + Dash + Plotly**, the app helps analysts quickly compare capital, credit, balance-sheet mix, and performance trends over time.

**Live Dashboard**: https://grasshopper-bank-dashboard-7b37a160d5b8.herokuapp.com/

---

## Overview

This dashboard is designed to answer a simple question fast:

> “How does Grasshopper Bank compare to similar banks on the FDIC call report?”

It pulls quarterly bank financial data from the FDIC Banks Data API, computes a standardized set of metrics (including capital-base adjusted exposure ratios and credit performance measures), and renders:

- a **cross-sectional view** (Grasshopper vs peers on a selected quarter), and  
- a **historical view** (trend lines over a selected lookback window).

Grasshopper Bank is always highlighted to keep the comparison centered.

---

## Why These Peer Banks?

The peer set is intentionally **not** “largest banks” or “generic regional banks.” It focuses on **banks with similar operating DNA**—digital-forward, specialty lending and/or fintech partner models—where balance-sheet mix and growth dynamics tend to be more comparable.

Peers included:
- **Live Oak Bank** – specialty SBA / business banking focus  
- **Celtic Bank** – sponsor finance / fintech enablement profile  
- **Coastal Community Bank** – fintech & partner banking footprint  
- **Choice Financial Group** – sponsor / fintech partnerships and niche verticals  
- **Metropolitan Commercial Bank** – commercial/fintech-adjacent mix  
- **Cross River Bank** – payments/embedded finance ecosystem  
- **Axos Bank** – digital-first banking model with scalable deposit gathering

This is meant to produce **more meaningful “apples-to-apples” ranges** on things like capital-adjusted exposure, credit cost behavior, and operating performance, versus comparing to banks with fundamentally different product stacks.

---

## Key Features

- **Grasshopper vs. Peer Snapshot (Quarter-End)**
  - Bar chart for quick ranking on a chosen metric and quarter
  - Grasshopper visually emphasized

- **Historical Trend Analysis (1–6 years)**
  - Multi-bank trend lines with flexible lookback
  - Timeline selector (1, 2, 3, 4, 5, 6 years)

- **Automated Positioning & Context**
  - Percentile ranking and rank # out of selected banks
  - Quartile “performance group” (Top 25%, Middle 50%, Bottom 25%)
  - Z-score (“how many standard deviations from average”)

- **Correlation & Similarity**
  - Identifies which bank is moving most/least like Grasshopper for the chosen metric over the selected horizon

- **Deep-Dive Bank Details**
  - Click a bank bar to view a dense metric readout for that quarter

- **Resilience / Caching**
  - Local JSON caching to reduce repeated API calls
  - Fallback sample data generation if the FDIC API is unreachable

---

## Metrics Included

The dashboard combines **balance sheet**, **credit**, **capital**, and **performance** measures, including (but not limited to):

### Capital / Exposure (Tier 1 + ACL)
- Real Estate Loans to Tier 1 + ACL
- RE Construction & Land Development to Tier 1 + ACL
- Commercial RE to Tier 1 + ACL
- C&I Loans to Tier 1 + ACL
- Auto Loans to Tier 1 + ACL

### Credit Quality / Loss
- Nonperforming Assets / Total Assets
- Assets Past Due 30–89 / Total Assets
- Assets Past Due 90+ / Total Assets
- Net Charge-Offs / Total Loans & Leases
- Net Charge-Offs / Allowance for Credit Loss
- Provision to Net Charge-Offs
- Earnings Coverage of Net Loan Charge-Offs

### Profitability / Efficiency
- Return on Assets (ROA)
- Return on Equity (ROE)
- Net Interest Margin (NIM)
- Efficiency Ratio

### Balance Sheet Size & Mix (Dollar Metrics)
- Total Assets, Deposits, Loans & Leases, Net Loans & Leases
- Securities, Real Estate Loans, C&I Loans
- Consumer products (e.g., Auto Loans, Credit Cards, Consumer Loans)
- Tier 1 (Core) Capital, Net Income
- Allowance for Credit Loss, Charge-offs / Recoveries

> Metric tooltips in the app include short definitions to keep interpretation consistent.

---

## Data Source

All bank financials are sourced from the **FDIC Banks Data API** (quarterly call report data):
- Base URL used by the app: `https://banks.data.fdic.gov/api`

The app queries:
- `institutions` endpoint (bank identifiers)
- `financials` endpoint (quarterly fields used to compute metrics)

---

## Technology Stack

- **Python**
- **Dash** (web app framework)
- **Plotly** (interactive charts)
- **Pandas / NumPy** (data handling and computation)
- **SciPy** (percentiles, z-scores, correlations)
- **Requests** (FDIC API calls)
- **Heroku** (deployment/hosting)

---

## Local Development

### 1) Clone & install
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
