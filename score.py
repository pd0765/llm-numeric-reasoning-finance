# score.py
# Scores trial output CSVs against the ground truth reference.
# Produces per-ticker error metrics, failure mode classification,
# and a condition-level summary table.
#
# Handles both API trial CSVs (Conditions C-F) and manually entered
# Claude.ai trial CSVs (Conditions A-B), provided they share the same
# column structure: Ticker, Sharpe, Sortino, Max_Drawdown.
#
# Usage:
#   python score.py --trial results/condition_C_rep1_v1.csv
#   python score.py --all                                      # score all CSVs in results/
#   python score.py --summary                                  # print condition-level summary table

import argparse
import os

import numpy as np
import pandas as pd

from config import (
    GROUND_TRUTH_PATH,
    TRIAL_OUTPUT_DIR,
    SHARPE_DECIMALS,
    SORTINO_DECIMALS,
    DRAWDOWN_DECIMALS,
    VERSION,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score trial output CSVs against the ground truth reference."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--trial",
        type=str,
        metavar="PATH",
        help="Path to a single trial output CSV to score."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Score all trial CSVs found in the results directory."
    )
    group.add_argument(
        "--summary",
        action="store_true",
        help="Print condition-level summary table across all scored results."
    )
    return parser.parse_args()


# =============================================================================
# FAILURE MODE CLASSIFIER
# =============================================================================

# Tolerance thresholds for classifying errors by magnitude.
# Errors below TOLERANCE are considered correct (floating point noise).
# Errors above LARGE_ERROR_THRESHOLD are flagged as large errors.
TOLERANCE          = 0.001   # within 0.001 of ground truth = correct
LARGE_ERROR_THRESHOLD = 0.10  # absolute error >= 0.10 = large error

def classify_failure(trial_val, gt_val) -> str:
    """
    Classify a single metric value against its ground truth.

    Categories
    ----------
    "correct"       : within TOLERANCE of ground truth
    "small_error"   : absolute error in (TOLERANCE, LARGE_ERROR_THRESHOLD)
    "large_error"   : absolute error >= LARGE_ERROR_THRESHOLD
    "missing"       : trial value is NaN (tab dropped or parse failure)
    "sign_error"    : correct magnitude but wrong sign (e.g. drawdown sign flip)

    Parameters
    ----------
    trial_val : float or NaN
        Value produced by the model for this metric.
    gt_val : float
        Ground truth value for this metric.

    Returns
    -------
    str
        Failure mode category string.
    """
    if pd.isna(trial_val):
        return "missing"

    abs_error = abs(trial_val - gt_val)

    if abs_error <= TOLERANCE:
        return "correct"

    # Check for sign error: same magnitude, opposite sign
    if abs(abs(trial_val) - abs(gt_val)) <= TOLERANCE and (trial_val * gt_val < 0):
        return "sign_error"

    if abs_error >= LARGE_ERROR_THRESHOLD:
        return "large_error"

    return "small_error"


# =============================================================================
# SINGLE TRIAL SCORER
# =============================================================================

def score_trial(trial_path: str, gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score one trial CSV against the ground truth DataFrame.

    For each ticker and each metric, computes:
    - trial value
    - ground truth value
    - absolute error
    - failure mode classification

    Parameters
    ----------
    trial_path : str
        Path to the trial output CSV.
    gt_df : pd.DataFrame
        Ground truth DataFrame with columns: Ticker, Sharpe, Sortino, Max_Drawdown.

    Returns
    -------
    pd.DataFrame
        Scored results with one row per ticker per metric.
        Columns: Trial, Ticker, Metric, GT_Value, Trial_Value, Abs_Error, Failure_Mode.
    """
    trial_df = pd.read_csv(trial_path)
    trial_name = os.path.basename(trial_path).replace(".csv", "")

    metrics = ["Sharpe", "Sortino", "Max_Drawdown"]
    records = []

    # Merge on Ticker to align rows
    merged = pd.merge(gt_df, trial_df, on="Ticker", suffixes=("_gt", "_trial"), how="left")

    for _, row in merged.iterrows():
        for metric in metrics:
            gt_col    = f"{metric}_gt"
            trial_col = f"{metric}_trial"

            gt_val    = row[gt_col]    if gt_col    in row.index else np.nan
            trial_val = row[trial_col] if trial_col in row.index else np.nan

            abs_error = abs(trial_val - gt_val) if not pd.isna(trial_val) else np.nan
            failure   = classify_failure(trial_val, gt_val)

            records.append({
                "Trial":        trial_name,
                "Ticker":       row["Ticker"],
                "Metric":       metric,
                "GT_Value":     round(gt_val,    6) if not pd.isna(gt_val)    else np.nan,
                "Trial_Value":  round(trial_val, 6) if not pd.isna(trial_val) else np.nan,
                "Abs_Error":    round(abs_error, 6) if not pd.isna(abs_error) else np.nan,
                "Failure_Mode": failure,
            })

    return pd.DataFrame(records)


# =============================================================================
# CONDITION-LEVEL SUMMARY
# =============================================================================

def build_summary(all_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate scored results to a condition-level summary table.

    For each trial (condition + replication), reports:
    - Total ticker-metric pairs scored
    - Count and rate of each failure mode
    - Mean absolute error per metric (excluding NaN)

    Parameters
    ----------
    all_scored : pd.DataFrame
        Concatenated scored results from all trials.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per trial.
    """
    summary_records = []

    for trial_name, group in all_scored.groupby("Trial"):
        n_total = len(group)

        failure_counts = group["Failure_Mode"].value_counts().to_dict()
        n_correct     = failure_counts.get("correct",     0)
        n_small       = failure_counts.get("small_error", 0)
        n_large       = failure_counts.get("large_error", 0)
        n_missing     = failure_counts.get("missing",     0)
        n_sign        = failure_counts.get("sign_error",  0)

        # MAE per metric
        mae_by_metric = (
            group[group["Abs_Error"].notna()]
            .groupby("Metric")["Abs_Error"]
            .mean()
            .round(6)
            .to_dict()
        )

        summary_records.append({
            "Trial":              trial_name,
            "N_Total":            n_total,
            "N_Correct":          n_correct,
            "Pct_Correct":        round(n_correct / n_total * 100, 1),
            "N_Small_Error":      n_small,
            "N_Large_Error":      n_large,
            "N_Missing":          n_missing,
            "N_Sign_Error":       n_sign,
            "MAE_Sharpe":         mae_by_metric.get("Sharpe",       np.nan),
            "MAE_Sortino":        mae_by_metric.get("Sortino",      np.nan),
            "MAE_Max_Drawdown":   mae_by_metric.get("Max_Drawdown", np.nan),
        })

    summary_df = pd.DataFrame(summary_records).sort_values("Trial").reset_index(drop=True)
    return summary_df


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_scored_csv(scored_df: pd.DataFrame, trial_path: str) -> str:
    """
    Write the per-ticker scored results to a CSV alongside the trial file.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored results from score_trial().
    trial_path : str
        Original trial CSV path; scored CSV is written to same directory.

    Returns
    -------
    str
        Path where the scored CSV was written.
    """
    base     = os.path.basename(trial_path).replace(".csv", "_scored.csv")
    out_path = os.path.join(os.path.dirname(trial_path), base)
    scored_df.to_csv(out_path, index=False)
    return out_path


def print_scored_summary(scored_df: pd.DataFrame, trial_name: str) -> None:
    """Print a concise per-metric failure mode breakdown for one trial."""
    print(f"\n  {'Metric':<15} {'Correct':>8} {'Small':>8} {'Large':>8} {'Missing':>8} {'Sign':>8} {'MAE':>10}")
    print(f"  {'-'*67}")
    for metric, grp in scored_df.groupby("Metric"):
        counts = grp["Failure_Mode"].value_counts().to_dict()
        mae    = grp["Abs_Error"].mean()
        print(
            f"  {metric:<15}"
            f" {counts.get('correct',     0):>8}"
            f" {counts.get('small_error', 0):>8}"
            f" {counts.get('large_error', 0):>8}"
            f" {counts.get('missing',     0):>8}"
            f" {counts.get('sign_error',  0):>8}"
            f" {mae:>10.4f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if not os.path.exists(GROUND_TRUTH_PATH):
        raise FileNotFoundError(
            f"Ground truth not found at '{GROUND_TRUTH_PATH}'. "
            "Run ground_truth.py first."
        )

    gt_df = pd.read_csv(GROUND_TRUTH_PATH)
    print(f"Ground truth loaded: {len(gt_df)} tickers from {GROUND_TRUTH_PATH}\n")

    # -------------------------------------------------------------------------
    # --summary: aggregate across all scored CSVs already in results/
    # -------------------------------------------------------------------------
    if args.summary:
        scored_files = [
            os.path.join(TRIAL_OUTPUT_DIR, f)
            for f in os.listdir(TRIAL_OUTPUT_DIR)
            if f.endswith("_scored.csv")
        ]
        if not scored_files:
            print("No scored CSVs found in results/. Run --all or --trial first.")
            return

        all_scored = pd.concat([pd.read_csv(f) for f in scored_files], ignore_index=True)
        summary_df = build_summary(all_scored)

        print("=== CONDITION-LEVEL SUMMARY ===\n")
        print(summary_df.to_string(index=False))

        summary_path = os.path.join(TRIAL_OUTPUT_DIR, f"summary_{VERSION}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        return

    # -------------------------------------------------------------------------
    # Collect trial files to score
    # -------------------------------------------------------------------------
    if args.all:
        trial_files = [
            os.path.join(TRIAL_OUTPUT_DIR, f)
            for f in os.listdir(TRIAL_OUTPUT_DIR)
            if f.endswith(".csv") and "_scored" not in f and "summary" not in f
        ]
        if not trial_files:
            print(f"No trial CSVs found in {TRIAL_OUTPUT_DIR}.")
            return
    else:
        trial_files = [args.trial]

    # -------------------------------------------------------------------------
    # Score each trial
    # -------------------------------------------------------------------------
    all_scored_frames = []

    for trial_path in sorted(trial_files):
        if not os.path.exists(trial_path):
            print(f"  WARNING: File not found — {trial_path}")
            continue

        trial_name = os.path.basename(trial_path).replace(".csv", "")
        print(f"Scoring: {trial_name}")

        scored_df = score_trial(trial_path, gt_df)
        out_path  = write_scored_csv(scored_df, trial_path)
        print_scored_summary(scored_df, trial_name)
        print(f"  Scored CSV written to: {out_path}")

        all_scored_frames.append(scored_df)

    # -------------------------------------------------------------------------
    # Print aggregate summary if multiple trials were scored
    # -------------------------------------------------------------------------
    if len(all_scored_frames) > 1:
        all_scored = pd.concat(all_scored_frames, ignore_index=True)
        summary_df = build_summary(all_scored)

        print("\n=== CONDITION-LEVEL SUMMARY ===\n")
        print(summary_df.to_string(index=False))

        summary_path = os.path.join(TRIAL_OUTPUT_DIR, f"summary_{VERSION}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    print("\nscore.py complete.")


if __name__ == "__main__":
    main()
