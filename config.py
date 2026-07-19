# config.py
# Central configuration for LLM Numeric Reasoning Degradation Benchmark
# All parameters are defined here; no hardcoding in other scripts.

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# ETF UNIVERSE
# =============================================================================

# 20 diversified ETFs spanning major asset classes and factors.
# yfinance tickers — all freely available market data, no index IP concerns.
TICKERS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "EFA",   # MSCI EAFE (Developed International)
    "EEM",   # MSCI Emerging Markets
    "AGG",   # US Aggregate Bond
    "LQD",   # Investment Grade Corporate Bond
    "HYG",   # High Yield Corporate Bond
    "TLT",   # 20+ Year Treasury
    "GLD",   # Gold
    "VNQ",   # US Real Estate (REIT)
    "XLE",   # Energy Sector
    "XLF",   # Financial Sector
    "XLV",   # Health Care Sector
    "XLK",   # Technology Sector
    "XLU",   # Utilities Sector
    "XLI",   # Industrials Sector
    "XLP",   # Consumer Staples Sector
    "VTV",   # Vanguard Value
    "VUG",   # Vanguard Growth
]

# =============================================================================
# DATE RANGE
# =============================================================================

START_DATE = "2019-01-01"
END_DATE   = "2023-12-31"   # 5 full calendar years; keeps file size manageable

# =============================================================================
# RETURN FREQUENCY
# =============================================================================

# "monthly" or "daily"
# Monthly is recommended for v1 (cleaner series, fewer rows, easier for agent to process)
RETURN_FREQUENCY = "monthly"

# Annualization factor applied to volatility denominator in Sharpe and Sortino
# 12 for monthly, 252 for daily
ANNUALIZATION_FACTOR = 12

# =============================================================================
# API KEYS
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FRED_SERIES_ID  = "TB3MS"          # 3-Month T-Bill Secondary Market Rate (monthly)
FRED_API_KEY    = os.getenv("FRED_API_KEY")   # Replace with your FRED API key

# FRED publishes rates as annualized percentages (e.g., 5.25 means 5.25%).
# Per-period conversion: divide by (100 * ANNUALIZATION_FACTOR)
# This is handled in utils.py; documented here for methodology transparency.

# =============================================================================
# SORTINO: MINIMUM ACCEPTABLE RETURN (MAR)
# =============================================================================

# MAR is set to the risk-free rate for each period (consistent with Sharpe convention).
# Downside deviation uses returns below the MAR, not below zero.
# This is enforced in utils.py.
MAR_SOURCE = "risk_free_rate"   # Options: "risk_free_rate" or "zero"

# =============================================================================
# MAX DRAWDOWN
# =============================================================================

# Computed on the price-level series (NAV), not the return series.
# Expressed as a negative decimal (e.g., -0.3412 = -34.12% drawdown).
DRAWDOWN_ON = "price"    # "price" or "returns" — price is standard convention

# =============================================================================
# OUTPUT PRECISION
# =============================================================================

SHARPE_DECIMALS  = 4
SORTINO_DECIMALS = 4
DRAWDOWN_DECIMALS = 4   # expressed as decimal, e.g. -0.3412

# =============================================================================
# FILE PATHS
# =============================================================================

WORKBOOK_PATH    = "data/etf_returns.xlsx"       # Fixed input workbook (produced by data_pull.py)
GROUND_TRUTH_PATH = "data/ground_truth.csv"      # Reference output (produced by ground_truth.py)
TRIAL_OUTPUT_DIR  = "results/"                   # API trial CSVs written here
DRAFT_OUTPUT_PATH = "results/draft_output.csv"   # Test-run output (--tabs N flag)

# =============================================================================
# EXPERIMENTAL CONDITIONS
# =============================================================================

# Six conditions as defined in the experimental design.
# "tool_access": whether the API call includes a code execution tool
# "system_prompt": one of "none", "neutral", "must_use_python"

CONDITIONS = {
    "A": {
        "platform": "claude.ai",
        "model": "claude-sonnet-4-6",
        "tool_access": False,
        "system_prompt": "none",
        "notes": "Manual trial — attach workbook, no system prompt"
    },
    "B": {
        "platform": "claude.ai",
        "model": "claude-opus-4-6",
        "tool_access": False,
        "system_prompt": "none",
        "notes": "Manual trial — attach workbook, no system prompt"
    },
    "C": {
        "platform": "api",
        "model": "claude-sonnet-4-6",
        "tool_access": False,
        "system_prompt": "neutral",
        "notes": "API trial — neutral system prompt, no tool access"
    },
    "D": {
        "platform": "api",
        "model": "claude-sonnet-4-6",
        "tool_access": False,
        "system_prompt": "must_use_python",
        "notes": "API trial — instructed to use Python, but no execution capability"
    },
    "E": {
        "platform": "api",
        "model": "claude-sonnet-4-6",
        "tool_access": True,
        "system_prompt": "neutral",
        "notes": "API trial — tool access enabled, neutral system prompt"
    },
    "F": {
        "platform": "api",
        "model": "claude-sonnet-4-6",
        "tool_access": True,
        "system_prompt": "must_use_python",
        "notes": "API trial — tool access enabled, instructed to use Python"
    },
}

# =============================================================================
# SYSTEM PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPTS = {
    "none": None,

    "neutral": (
        "You are a financial analyst assistant. "
        "The user will provide ETF return data and ask you to compute performance metrics. "
        "Respond clearly and accurately."
    ),

    "must_use_python": (
        "You are a financial analyst assistant. "
        "The user will provide ETF return data and ask you to compute performance metrics. "
        "ALL calculations MUST be performed by writing and executing a Python script. "
        "Do not perform any arithmetic manually or through prose reasoning. "
        "Respond clearly and accurately."
    ),
}

# =============================================================================
# REPLICATION
# =============================================================================

N_REPLICATIONS = 3      # Number of runs per API condition (Conditions C–F)
API_TEMPERATURE = 0     # Fixed for reproducibility across replications

# =============================================================================
# VERSIONS
# =============================================================================

# v1: Single full-period metrics (Sharpe, Sortino, Max Drawdown) across all tabs
# v2: Rolling 12-month trailing metrics, recalculated monthly (extension)
VERSION = "v1"    # Switch to "v2" when running the rolling extension
ROLLING_WINDOW_MONTHS = 12    # Used only when VERSION == "v2"
