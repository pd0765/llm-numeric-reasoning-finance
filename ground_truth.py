# ground_truth.py
# Computes verified performance metrics for every ETF tab in the fixed input
# workbook. Produces a reference CSV that all trial outputs are scored against.
# Run once after data_pull.py; do not re-run between trials.
#
# Output: data/ground_truth.csv
#   - One row per ETF ticker
#   - Columns: Ticker, Sharpe, Sortino, Max_Drawdown
#
# Usage:
#   python ground_truth.py            # full run (all ETF tabs)
#   python ground_truth.py --tabs 3   # test run (first 3 ETF tabs only)

import argparse
import os

import pandas as pd

from config import (
    TICKERS,
    WORKBOOK_PATH,
    GROUND_TRUTH_PATH,
    SHARPE_DECIMALS,
    SORTINO_DECIMALS,
    DRAWDOWN_DECIMALS,
)
from utils import (
    convert_rf_to_periodic,
    prices_to_returns,
    align_series,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ground truth performance metrics from the input workbook."
    )
    parser.add_argument(
        "--tabs",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N ETF tabs for test runs. Omit for full run."
    )
    return parser.parse_args()


# =============================================================================
# WORKBOOK READERS
# =============================================================================

def read_rf_series(workbook_path: str) -> tuple[pd.Series, pd.Series]:
    """
    Read the RF tab from the input workbook.

    Returns both the annualized percentage series and the pre-computed
    per-period decimal series written by data_pull.py.

    Parameters
    ----------
    workbook_path : str
        Path to the fixed input workbook.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (rf_annual_pct, rf_periodic_decimal) with DatetimeIndex.
    """
    rf_df = pd.read_excel(workbook_path, sheet_name="RF", parse_dates=["Date"])
    rf_df = rf_df.set_index("Date").sort_index()
    rf_annual = rf_df["RF_Annual_Pct"]
    rf_periodic = rf_df["RF_Periodic_Decimal"]
    return rf_annual, rf_periodic


def read_etf_tab(workbook_path: str, ticker: str) -> tuple[pd.Series, pd.Series]:
    """
    Read one ETF tab from the input workbook.

    Returns the price series and the pre-computed return series.
    Max drawdown is computed from prices; Sharpe and Sortino from returns.

    Parameters
    ----------
    workbook_path : str
        Path to the fixed input workbook.
    ticker : str
        ETF ticker symbol matching the tab name.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (prices, returns) with DatetimeIndex, NaNs dropped.
    """
    df = pd.read_excel(workbook_path, sheet_name=ticker, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    prices = df["Price"].dropna()
    returns = df["Return"].dropna()
    return prices, returns


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics_for_ticker(
    ticker: str,
    workbook_path: str,
    rf_periodic: pd.Series,
) -> dict:
    """
    Compute Sharpe, Sortino, and Max Drawdown for a single ETF ticker.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol.
    workbook_path : str
        Path to the fixed input workbook.
    rf_periodic : pd.Series
        Per-period decimal risk-free rates with DatetimeIndex.

    Returns
    -------
    dict
        Keys: Ticker, Sharpe, Sortino, Max_Drawdown.
        Values: rounded floats per precision settings in config.py.
    """
    prices, returns = read_etf_tab(workbook_path, ticker)

    # Align return series and risk-free rate to overlapping dates before
    # computing Sharpe and Sortino — guards against any index mismatch.
    returns_aligned, rf_aligned = align_series(returns, rf_periodic)

    sharpe  = compute_sharpe(returns_aligned, rf_aligned)
    sortino = compute_sortino(returns_aligned, rf_aligned)
    max_dd  = compute_max_drawdown(prices)

    return {
        "Ticker":       ticker,
        "Sharpe":       sharpe,
        "Sortino":      sortino,
        "Max_Drawdown": max_dd,
    }


# =============================================================================
# GROUND TRUTH WRITER
# =============================================================================

def write_ground_truth(records: list[dict], output_path: str) -> None:
    """
    Write the ground truth records to a CSV file.

    Parameters
    ----------
    records : list[dict]
        List of metric dicts, one per ticker.
    output_path : str
        Destination path for the ground truth CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records, columns=["Ticker", "Sharpe", "Sortino", "Max_Drawdown"])
    df.to_csv(output_path, index=False)
    print(f"\nGround truth saved to: {output_path}")
    print(df.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    tickers = TICKERS[:args.tabs] if args.tabs is not None else TICKERS

    if args.tabs is not None:
        print(f"TEST RUN: limiting to first {args.tabs} ETF tabs.\n")
    else:
        print("FULL RUN: processing all ETF tabs.\n")

    if not os.path.exists(WORKBOOK_PATH):
        raise FileNotFoundError(
            f"Input workbook not found at '{WORKBOOK_PATH}'. "
            "Run data_pull.py first."
        )

    print(f"Reading workbook: {WORKBOOK_PATH}")
    _, rf_periodic = read_rf_series(WORKBOOK_PATH)

    records = []
    for ticker in tickers:
        print(f"  Computing metrics: {ticker}")
        try:
            result = compute_metrics_for_ticker(ticker, WORKBOOK_PATH, rf_periodic)
            records.append(result)
            print(
                f"    Sharpe={result['Sharpe']:.{SHARPE_DECIMALS}f}  "
                f"Sortino={result['Sortino']:.{SORTINO_DECIMALS}f}  "
                f"Max_Drawdown={result['Max_Drawdown']:.{DRAWDOWN_DECIMALS}f}"
            )
        except Exception as e:
            print(f"    ERROR for {ticker}: {e}")

    write_ground_truth(records, GROUND_TRUTH_PATH)

    print(f"\nground_truth.py complete.")
    print(f"  Tickers processed : {len(records)}")
    print(f"  Output path       : {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
