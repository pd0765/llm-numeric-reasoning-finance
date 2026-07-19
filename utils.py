# utils.py
# Shared helper functions for LLM Numeric Reasoning Degradation Benchmark.
# Imported by ground_truth.py and api_agent.py to ensure identical calculation
# logic between the reference output and the tool-augmented agent conditions.

import numpy as np
import pandas as pd

from config import (
    ANNUALIZATION_FACTOR,
    MAR_SOURCE,
    SHARPE_DECIMALS,
    SORTINO_DECIMALS,
    DRAWDOWN_DECIMALS,
)


# =============================================================================
# RISK-FREE RATE CONVERSION
# =============================================================================

def convert_rf_to_periodic(rf_series: pd.Series) -> pd.Series:
    """
    Convert FRED annualized percentage rates to per-period decimal rates.

    FRED publishes TB3MS as an annualized percentage (e.g., 5.25 = 5.25% p.a.).
    This function divides by (100 * ANNUALIZATION_FACTOR) to produce a
    per-period decimal rate aligned with the return series frequency.

    Parameters
    ----------
    rf_series : pd.Series
        Raw FRED rate series (annualized percentages).

    Returns
    -------
    pd.Series
        Per-period decimal rates.
    """
    return rf_series / (100 * ANNUALIZATION_FACTOR)


# =============================================================================
# EXCESS RETURNS
# =============================================================================

def compute_excess_returns(returns: pd.Series, rf_periodic: pd.Series) -> pd.Series:
    """
    Subtract the per-period risk-free rate from raw periodic returns.

    Both series must share the same DatetimeIndex. Any periods present in
    returns but missing from rf_periodic are forward-filled before subtraction.

    Parameters
    ----------
    returns : pd.Series
        Per-period total returns as decimals (e.g., 0.0123 = 1.23%).
    rf_periodic : pd.Series
        Per-period risk-free rates as decimals, aligned to same frequency.

    Returns
    -------
    pd.Series
        Excess returns (returns minus risk-free rate), per period.
    """
    rf_aligned = rf_periodic.reindex(returns.index, method="ffill")
    return returns - rf_aligned


# =============================================================================
# SHARPE RATIO
# =============================================================================

def compute_sharpe(returns: pd.Series, rf_periodic: pd.Series) -> float:
    """
    Compute the annualized Sharpe ratio.

    Formula:
        Sharpe = mean(excess_returns) / std(excess_returns) * sqrt(ANNUALIZATION_FACTOR)

    Standard deviation uses ddof=1 (sample std).

    Parameters
    ----------
    returns : pd.Series
        Per-period total returns as decimals.
    rf_periodic : pd.Series
        Per-period risk-free rates as decimals.

    Returns
    -------
    float
        Annualized Sharpe ratio, rounded to SHARPE_DECIMALS.
    """
    excess = compute_excess_returns(returns, rf_periodic)
    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)

    if std_excess == 0 or np.isnan(std_excess):
        return np.nan

    sharpe = mean_excess / std_excess * np.sqrt(ANNUALIZATION_FACTOR)
    return round(sharpe, SHARPE_DECIMALS)


# =============================================================================
# SORTINO RATIO
# =============================================================================

def compute_sortino(returns: pd.Series, rf_periodic: pd.Series) -> float:
    """
    Compute the annualized Sortino ratio.

    MAR (Minimum Acceptable Return) is set to the per-period risk-free rate
    when MAR_SOURCE == "risk_free_rate" (per config.py), or zero otherwise.

    Formula:
        downside_returns = returns[returns < MAR] - MAR
        downside_deviation = sqrt(mean(downside_returns ** 2)) * sqrt(ANNUALIZATION_FACTOR)
        Sortino = (mean(returns) - mean(MAR)) / downside_deviation * ANNUALIZATION_FACTOR

    Note: annualized numerator uses mean(returns)*ANNUALIZATION_FACTOR minus
    mean(MAR)*ANNUALIZATION_FACTOR; denominator uses downside_deviation already
    annualized via sqrt(ANNUALIZATION_FACTOR).

    Parameters
    ----------
    returns : pd.Series
        Per-period total returns as decimals.
    rf_periodic : pd.Series
        Per-period risk-free rates as decimals.

    Returns
    -------
    float
        Annualized Sortino ratio, rounded to SORTINO_DECIMALS.
    """
    if MAR_SOURCE == "risk_free_rate":
        mar = rf_periodic.reindex(returns.index, method="ffill")
    else:
        mar = pd.Series(0.0, index=returns.index)

    downside_diff = returns - mar
    downside_returns = downside_diff[downside_diff < 0]

    if len(downside_returns) == 0:
        return np.nan

    downside_deviation = np.sqrt((downside_diff.clip(upper=0) ** 2).mean()) * np.sqrt(ANNUALIZATION_FACTOR)

    if downside_deviation == 0 or np.isnan(downside_deviation):
        return np.nan

    annualized_excess = (returns.mean() - mar.mean()) * ANNUALIZATION_FACTOR
    sortino = annualized_excess / downside_deviation
    return round(sortino, SORTINO_DECIMALS)


# =============================================================================
# MAXIMUM DRAWDOWN
# =============================================================================

def compute_max_drawdown(prices: pd.Series) -> float:
    """
    Compute maximum drawdown on the price-level series.

    Drawdown at each point is defined as:
        (price - running_peak) / running_peak

    Maximum drawdown is the minimum (most negative) value of this series.
    Expressed as a negative decimal (e.g., -0.3412 = -34.12% drawdown).

    Parameters
    ----------
    prices : pd.Series
        Price-level (NAV) series. Must contain no NaN values.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal, rounded to DRAWDOWN_DECIMALS.
    """
    prices = prices.dropna()

    if len(prices) < 2:
        return np.nan

    running_peak = prices.cummax()
    drawdown = (prices - running_peak) / running_peak
    max_dd = drawdown.min()
    return round(max_dd, DRAWDOWN_DECIMALS)


# =============================================================================
# RETURN SERIES BUILDER
# =============================================================================

def prices_to_returns(prices: pd.Series) -> pd.Series:
    """
    Convert a price-level series to simple periodic returns.

    Uses simple (arithmetic) returns: (P_t - P_{t-1}) / P_{t-1}.
    The first observation is dropped (NaN from pct_change).

    Parameters
    ----------
    prices : pd.Series
        Price-level (NAV) series with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Simple periodic returns as decimals, with first observation dropped.
    """
    return prices.pct_change().dropna()


# =============================================================================
# ALIGNMENT UTILITY
# =============================================================================

def align_series(returns: pd.Series, rf_periodic: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Align returns and risk-free rate series to their overlapping date range.

    Drops any periods where either series has a NaN after alignment.

    Parameters
    ----------
    returns : pd.Series
        Per-period return series with a DatetimeIndex.
    rf_periodic : pd.Series
        Per-period risk-free rate series with a DatetimeIndex.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Aligned (returns, rf_periodic) with identical index, no NaNs.
    """
    rf_aligned = rf_periodic.reindex(returns.index, method="ffill")
    combined = pd.concat([returns, rf_aligned], axis=1).dropna()
    combined.columns = ["returns", "rf"]
    return combined["returns"], combined["rf"]
