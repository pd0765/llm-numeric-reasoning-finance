# main.py
# Entry point and orchestrator for the LLM Numeric Reasoning Degradation Benchmark.
# Imports from all other scripts and controls execution flow based on arguments.
#
# Run order for a clean experiment:
#   1. python main.py --setup              # build workbook and ground truth (run once)
#   2. python main.py --condition C --rep 1  # single API trial
#      python main.py --run-all             # all API conditions, all replications
#   3. python main.py --score-all           # score all trial CSVs
#   4. python main.py --summary             # print condition-level summary table
#
# Test run (first N tabs only, output to draft path):
#   python main.py --setup --tabs 3
#   python main.py --condition C --rep 1 --tabs 3
#   python main.py --score-all
#   python main.py --summary

import argparse
import os
import sys


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Numeric Reasoning Degradation Benchmark — main orchestrator."
    )

    # --- Mode flags (mutually exclusive) ---
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--setup",
        action="store_true",
        help="Run data_pull.py and ground_truth.py in sequence (run once before trials)."
    )
    mode.add_argument(
        "--condition",
        type=str,
        choices=["C", "D", "E", "F"],
        help="Run a single API trial condition."
    )
    mode.add_argument(
        "--run-all",
        action="store_true",
        help="Run all API conditions (C through F) for all replications."
    )
    mode.add_argument(
        "--score-all",
        action="store_true",
        help="Score all trial CSVs in the results directory."
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="Print condition-level summary table from already-scored CSVs."
    )

    # --- Shared optional arguments ---
    parser.add_argument(
        "--rep",
        type=int,
        default=None,
        metavar="N",
        help="Replication number (required with --condition)."
    )
    parser.add_argument(
        "--tabs",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N ETF tabs for test runs."
    )

    return parser.parse_args()


# =============================================================================
# SETUP: DATA PULL + GROUND TRUTH
# =============================================================================

def run_setup(tabs: int | None) -> None:
    """
    Run data_pull.py followed by ground_truth.py.
    Constitutes the one-time setup step before any trials are executed.

    Parameters
    ----------
    tabs : int or None
        If provided, limits both scripts to the first N tickers (test mode).
    """
    from data_pull    import main as data_pull_main
    from ground_truth import main as ground_truth_main

    print("=" * 60)
    print("STEP 1: data_pull.py")
    print("=" * 60)

    # Inject --tabs argument into sys.argv for downstream argparse calls
    _inject_tabs_arg(tabs)
    data_pull_main()

    print("\n" + "=" * 60)
    print("STEP 2: ground_truth.py")
    print("=" * 60)

    _inject_tabs_arg(tabs)
    ground_truth_main()

    print("\nSetup complete. Workbook and ground truth are ready.")


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_single_trial(condition: str, rep: int, tabs: int | None) -> None:
    """
    Run a single API trial for the specified condition and replication.

    Parameters
    ----------
    condition : str
        One of "C", "D", "E", "F".
    rep : int
        Replication number (1-indexed).
    tabs : int or None
        If provided, limits to the first N tickers (test mode).
    """
    from api_agent import run_trial, read_workbook_as_text
    from config    import TICKERS, WORKBOOK_PATH

    if not os.path.exists(WORKBOOK_PATH):
        _abort(f"Workbook not found at '{WORKBOOK_PATH}'. Run --setup first.")

    tickers  = TICKERS[:tabs] if tabs is not None else TICKERS
    is_draft = tabs is not None

    print("=" * 60)
    print(f"TRIAL: Condition {condition} | Rep {rep}")
    print("=" * 60)

    run_trial(condition, rep, tickers, is_draft)


# =============================================================================
# RUN ALL API CONDITIONS
# =============================================================================

def run_all_trials(tabs: int | None) -> None:
    """
    Run all API conditions (C through F) for all replications.

    Parameters
    ----------
    tabs : int or None
        If provided, limits to the first N tickers (test mode).
    """
    from api_agent import run_trial
    from config    import TICKERS, WORKBOOK_PATH, N_REPLICATIONS

    if not os.path.exists(WORKBOOK_PATH):
        _abort(f"Workbook not found at '{WORKBOOK_PATH}'. Run --setup first.")

    tickers  = TICKERS[:tabs] if tabs is not None else TICKERS
    is_draft = tabs is not None
    conditions = ["C", "D", "E", "F"]

    print("=" * 60)
    print("RUN ALL: Conditions C–F, all replications")
    print("=" * 60)

    for condition in conditions:
        for rep in range(1, N_REPLICATIONS + 1):
            run_trial(condition, rep, tickers, is_draft)

    print("\nAll API trials complete.")


# =============================================================================
# SCORE ALL
# =============================================================================

def run_score_all() -> None:
    """
    Score all trial CSVs found in the results directory.
    Includes manually entered Condition A and B CSVs if present.
    """
    from score import main as score_main

    print("=" * 60)
    print("SCORING: all trial CSVs in results/")
    print("=" * 60)

    # Inject --all flag for score.py's argparse
    sys.argv = [sys.argv[0], "--all"]
    score_main()


# =============================================================================
# SUMMARY
# =============================================================================

def run_summary() -> None:
    """
    Print the condition-level summary table from already-scored CSVs.
    """
    from score import main as score_main

    print("=" * 60)
    print("SUMMARY: condition-level results")
    print("=" * 60)

    sys.argv = [sys.argv[0], "--summary"]
    score_main()


# =============================================================================
# UTILITIES
# =============================================================================

def _inject_tabs_arg(tabs: int | None) -> None:
    """
    Replace sys.argv with a clean argument list for downstream argparse calls.
    Injects --tabs N if tabs is provided, otherwise passes no extra arguments.
    """
    if tabs is not None:
        sys.argv = [sys.argv[0], "--tabs", str(tabs)]
    else:
        sys.argv = [sys.argv[0]]


def _abort(message: str) -> None:
    """Print an error message and exit."""
    print(f"\nERROR: {message}")
    sys.exit(1)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    # Validate --rep requirement
    if args.condition and args.rep is None:
        _abort("--rep is required when using --condition.")

    if args.setup:
        run_setup(args.tabs)

    elif args.condition:
        run_single_trial(args.condition, args.rep, args.tabs)

    elif args.run_all:
        run_all_trials(args.tabs)

    elif args.score_all:
        run_score_all()

    elif args.summary:
        run_summary()

    print("\nmain.py complete.")


if __name__ == "__main__":
    main()
