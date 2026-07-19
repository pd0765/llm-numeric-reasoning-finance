# api_agent.py
# Executes API-based trial conditions (C through F) for the LLM Numeric
# Reasoning Degradation Benchmark. Reads the fixed input workbook, constructs
# the appropriate prompt and system prompt per condition, calls the Anthropic
# API, parses the response, and writes results to a structured CSV.
#
# Conditions A and B are manual Claude.ai trials and are not handled here.
#
# Output: results/<condition>_rep<N>_<version>.csv
#   - One row per ETF ticker
#   - Columns: Ticker, Sharpe, Sortino, Max_Drawdown
#
# Usage:
#   python api_agent.py --condition C --rep 1            # single condition, single rep
#   python api_agent.py --condition C --rep 1 --tabs 3   # test run (first 3 tabs)
#   python api_agent.py --all                            # all API conditions, all reps

import argparse
import json
import os
import re
import time

import anthropic
import pandas as pd

from config import (
    TICKERS,
    WORKBOOK_PATH,
    TRIAL_OUTPUT_DIR,
    DRAFT_OUTPUT_PATH,
    CONDITIONS,
    SYSTEM_PROMPTS,
    ANTHROPIC_API_KEY,
    API_TEMPERATURE,
    N_REPLICATIONS,
    VERSION,
    ANNUALIZATION_FACTOR,
    SHARPE_DECIMALS,
    SORTINO_DECIMALS,
    DRAWDOWN_DECIMALS,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run API-based trial conditions for the benchmark."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--condition",
        type=str,
        choices=["C", "D", "E", "F"],
        help="Single condition to run (C, D, E, or F)."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all API conditions (C through F) for all replications."
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=None,
        metavar="N",
        help="Replication number (1 to N_REPLICATIONS). Required with --condition."
    )
    parser.add_argument(
        "--tabs",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N ETF tabs for test runs. Output goes to draft path."
    )
    return parser.parse_args()


# =============================================================================
# WORKBOOK READER
# =============================================================================

def read_workbook_as_text(workbook_path: str, tickers: list[str]) -> str:
    """
    Read the input workbook and serialize ETF return data as structured text
    for inclusion in the API prompt.

    Each ETF is represented as a labeled block containing its monthly return
    series. The RF tab is included as a reference block. Price series are
    omitted from the prompt — the agent is asked to compute metrics from
    returns and risk-free rates only (max drawdown from returns via cumulative
    product is a valid alternative convention communicated in the prompt).

    Parameters
    ----------
    workbook_path : str
        Path to the fixed input workbook.
    tickers : list[str]
        ETF tickers to include in the prompt.

    Returns
    -------
    str
        Fully serialized text representation of the workbook data.
    """
    blocks = []

    # Read RF tab
    rf_df = pd.read_excel(workbook_path, sheet_name="RF", parse_dates=["Date"])
    rf_df = rf_df.set_index("Date").sort_index()
    rf_lines = ["=== RISK-FREE RATE (TB3MS, 3-Month T-Bill, per-period decimal) ==="]
    rf_lines.append("Date,RF_Periodic_Decimal")
    for date, row in rf_df.iterrows():
        rf_lines.append(f"{date.strftime('%Y-%m-%d')},{row['RF_Periodic_Decimal']:.8f}")
    blocks.append("\n".join(rf_lines))

    # Read each ETF tab
    for ticker in tickers:
        df = pd.read_excel(workbook_path, sheet_name=ticker, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        lines = [f"=== ETF: {ticker} ==="]
        lines.append("Date,Price,Return")
        for date, row in df.iterrows():
            ret_str = f"{row['Return']:.6f}" if pd.notna(row['Return']) else ""
            lines.append(f"{date.strftime('%Y-%m-%d')},{row['Price']:.6f},{ret_str}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def build_user_prompt(data_text: str, tickers: list[str], use_tools: bool = False) -> str:
    """
    Construct the user prompt containing the data and calculation instructions.

    Convention details are stated explicitly so that any deviation by the
    agent constitutes a scoreable failure mode rather than an ambiguous error.

    Parameters
    ----------
    data_text : str
        Serialized workbook data from read_workbook_as_text().
    tickers : list[str]
        ETF tickers included in this run.
    use_tools : bool
        True for tool-enabled conditions (E/F). Adjusts the output instruction
        so it does not conflict with multi-step code execution responses.
        The required JSON format is identical regardless.

    Returns
    -------
    str
        Full user prompt string.
    """
    ticker_list = ", ".join(tickers)

    if use_tools:
        output_instruction = (
            "CRITICAL INSTRUCTION: Use the code execution tool to perform ALL calculations. "
            "After your computations are complete, make sure the final output printed by your "
            "script is a single raw JSON object in the format below — nothing else after it."
        )
        reminder = (
            "REMINDER: The last thing printed by your code (or written as your final text) "
            "must be the JSON object and nothing else."
        )
    else:
        output_instruction = (
            "CRITICAL INSTRUCTION: Your response must consist of ONLY a single raw JSON object. "
            "Do not write any prose, reasoning, arithmetic, or explanation. "
            "Begin your response with { and end with }. "
            "Any response that is not pure JSON will be treated as a complete failure."
        )
        reminder = "REMINDER: Output ONLY the JSON object. No text before or after it."

    prompt = f"""{output_instruction}

REQUIRED OUTPUT FORMAT:
{{
  "results": [
    {{"Ticker": "SPY", "Sharpe": 0.1234, "Sortino": 0.5678, "Max_Drawdown": -0.3412}},
    {{"Ticker": "QQQ", "Sharpe": 0.1234, "Sortino": 0.5678, "Max_Drawdown": -0.3412}}
  ]
}}

TASK: Compute the following annualized performance metrics for these ETFs: {ticker_list}

1. SHARPE RATIO
   - Subtract the per-period risk-free rate from each monthly return to obtain excess returns.
   - Compute the mean and sample standard deviation (ddof=1) of the excess return series.
   - Annualize: (mean_excess / std_excess) * sqrt({ANNUALIZATION_FACTOR})
   - Round to {SHARPE_DECIMALS} decimal places.

2. SORTINO RATIO
   - Use the per-period risk-free rate as the Minimum Acceptable Return (MAR) each period.
   - Downside deviation: for each period, compute (return - MAR); retain only negative values,
     square them, take the mean, then take the square root. Annualize by multiplying by sqrt({ANNUALIZATION_FACTOR}).
   - Annualized numerator: (mean(return) - mean(MAR)) * {ANNUALIZATION_FACTOR}
   - Sortino = annualized numerator / annualized downside deviation.
   - Round to {SORTINO_DECIMALS} decimal places.

3. MAXIMUM DRAWDOWN
   - Compute from the PRICE series (not returns).
   - At each date, drawdown = (price - running_peak_price) / running_peak_price
   - Maximum drawdown = the minimum (most negative) value of this drawdown series.
   - Express as a negative decimal (e.g., -0.3412 means -34.12% drawdown).
   - Round to {DRAWDOWN_DECIMALS} decimal places.

{reminder}

DATA:
{data_text}
"""
    return prompt


# =============================================================================
# RESPONSE CONTENT SERIALIZER (for pause_turn continuation)
# =============================================================================

def _serialize_response_content(content_blocks) -> list:
    """
    Convert SDK response content blocks back to API-compatible dicts
    for use as an assistant turn in a continuation request after pause_turn.

    The Anthropic SDK returns typed objects (TextBlock, ToolUseBlock, etc.).
    The messages API requires plain dicts when constructing history.

    Parameters
    ----------
    content_blocks : list
        The response.content list from a prior API call.

    Returns
    -------
    list[dict]
        Serialized content blocks suitable for inclusion in messages history.
    """
    serialized = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)

        if block_type == "text":
            serialized.append({"type": "text", "text": block.text})

        elif block_type == "server_tool_use":
            serialized.append({
                "type": "server_tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

        elif block_type == "bash_code_execution_tool_result":
            # Serialize the nested content object as a dict
            inner = block.content
            inner_type = getattr(inner, "type", None)
            if inner_type == "bash_code_execution_result":
                inner_dict = {
                    "type": "bash_code_execution_result",
                    "stdout": getattr(inner, "stdout", ""),
                    "stderr": getattr(inner, "stderr", ""),
                    "return_code": getattr(inner, "return_code", 0),
                    "content": [],
                }
            else:
                # Fallback for error or unknown types — use to_dict if available
                inner_dict = inner.to_dict() if hasattr(inner, "to_dict") else {"type": str(inner_type)}

            serialized.append({
                "type": "bash_code_execution_tool_result",
                "tool_use_id": block.tool_use_id,
                "content": inner_dict,
            })

        elif block_type == "text_editor_code_execution_tool_result":
            inner = block.content
            inner_dict = inner.to_dict() if hasattr(inner, "to_dict") else {}
            serialized.append({
                "type": "text_editor_code_execution_tool_result",
                "tool_use_id": block.tool_use_id,
                "content": inner_dict,
            })

        else:
            # Fallback: use to_dict() if the SDK provides it
            if hasattr(block, "to_dict"):
                serialized.append(block.to_dict())
            else:
                # Skip unknown block types rather than crashing
                print(f"  WARNING: Unknown block type '{block_type}' — skipping in serialization.")

    return serialized


# =============================================================================
# API CALL
# =============================================================================

def call_api(
    condition_key: str,
    user_prompt: str,
) -> str:
    """
    Call the Anthropic API for the specified condition.

    Tool access (code execution) is enabled for Conditions E and F per the
    experimental design. Temperature is fixed at API_TEMPERATURE for all
    API conditions.

    Parameters
    ----------
    condition_key : str
        One of "C", "D", "E", "F".
    user_prompt : str
        Full user prompt string.

    Returns
    -------
    str
        Raw text response from the model.
    """
    condition = CONDITIONS[condition_key]
    model     = condition["model"]
    sp_key    = condition["system_prompt"]
    use_tools = condition["tool_access"]

    system_prompt = SYSTEM_PROMPTS.get(sp_key)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Tool-enabled conditions need a much higher token budget: the model must
    # write a Python script with all tickers' data embedded as literals, which
    # alone can exceed 8 K tokens for 20 tickers. 32 K gives ample headroom.
    max_tokens = 32768 if use_tools else 8096

    kwargs = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": API_TEMPERATURE,
        "messages":    [{"role": "user", "content": user_prompt}],
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    if use_tools:
        kwargs["tools"] = [
            {
                "type": "code_execution_20250825",
                "name": "code_execution",
            }
        ]

    # Accumulate text and stdout across ALL responses (initial + every
    # pause_turn continuation). The bash_code_execution_tool_result block
    # carrying the model's computed JSON may land in any intermediate response,
    # not necessarily the final one — so we must collect at every step.
    all_text_parts = []

    def _collect(content_blocks) -> None:
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if hasattr(block, "text"):
                all_text_parts.append(block.text)
            elif block_type == "bash_code_execution_tool_result":
                inner = getattr(block, "content", None)
                if inner is not None:
                    stderr = getattr(inner, "stderr", "") or ""
                    stdout = getattr(inner, "stdout", "") or ""
                    if stderr.strip():
                        print(f"  Code execution stderr:\n{stderr[:500]}")
                    if stdout.strip():
                        all_text_parts.append(stdout)
                    else:
                        print(f"  Code execution result: empty stdout/stderr (inner={type(inner).__name__})")
            elif block_type not in (None, "text", "server_tool_use",
                                     "text_editor_code_execution_tool_result"):
                print(f"  Unhandled block type: {block_type!r}")

    # Use streaming for all calls — required when max_tokens is large enough
    # that the SDK estimates the request could exceed 10 minutes (non-streaming
    # is rejected with a ValueError in that case). get_final_message() returns
    # the same Message object as messages.create(), so the rest of the code
    # is unchanged.
    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    print(f"  stop_reason: {response.stop_reason} | blocks: {len(response.content)}")
    _collect(response.content)

    # For tool-enabled conditions (E and F), the API may return stop_reason
    # "pause_turn" when a long-running code execution turn is paused mid-flight.
    # Feed the partial response back as an additional assistant turn so the
    # model can continue. Accumulate the full message history across iterations
    # so the server has complete context on each continuation call.
    MAX_CONTINUATIONS = 10
    continuations = 0
    cont_messages = list(kwargs["messages"])

    while response.stop_reason == "pause_turn" and continuations < MAX_CONTINUATIONS:
        continuations += 1
        print(f"  pause_turn received — continuing (attempt {continuations}/{MAX_CONTINUATIONS})...")

        prior_content = _serialize_response_content(response.content)
        cont_messages = cont_messages + [{"role": "assistant", "content": prior_content}]
        continuation_kwargs = {**kwargs, "messages": cont_messages}
        with client.messages.stream(**continuation_kwargs) as stream:
            response = stream.get_final_message()
        _collect(response.content)

    if response.stop_reason == "pause_turn":
        print(f"  WARNING: Still pause_turn after {MAX_CONTINUATIONS} continuations.")

    return "\n".join(all_text_parts)


# =============================================================================
# CLEANUP API CALL (no-tool conditions only)
# =============================================================================

def call_api_cleanup(
    condition_key: str,
    prior_response: str,
    tickers: list[str],
) -> str:
    """
    Second-pass API call to extract JSON from a prose response.

    Used for no-tool conditions (C and D) when the first call returns
    prose arithmetic instead of clean JSON. Sends the model's own prior
    response back and asks it to format the final answers only.

    Parameters
    ----------
    condition_key : str
        One of "C", "D".
    prior_response : str
        Raw text from the first API call.
    tickers : list[str]
        ETF tickers expected in the output.

    Returns
    -------
    str
        Raw text response from the cleanup call.
    """
    condition     = CONDITIONS[condition_key]
    model         = condition["model"]
    sp_key        = condition["system_prompt"]
    system_prompt = SYSTEM_PROMPTS.get(sp_key)

    ticker_list = ", ".join(tickers)
    cleanup_prompt = (
        f"Based on your calculations above, return your final answers for "
        f"{ticker_list} as a JSON object only. "
        f"Your entire response must be a single JSON object beginning with {{ and ending with }}. "
        f"No prose, no explanation, no markdown.\n\n"
        f"Required format:\n"
        f'{{"results": ['
        f'{{"Ticker": "SPY", "Sharpe": 0.1234, "Sortino": 0.5678, "Max_Drawdown": -0.3412}}, ...'
        f']}}'
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    messages = [
        {"role": "user",      "content": prior_response},
        {"role": "assistant", "content": prior_response},
        {"role": "user",      "content": cleanup_prompt},
    ]

    kwargs = {
        "model":       model,
        "max_tokens":  2048,
        "temperature": API_TEMPERATURE,
        "messages":    messages,
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    text_blocks = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(text_blocks)


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_response(response_text: str, tickers: list[str]) -> list[dict]:
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip()

    parsed = None

    # Attempt 1: direct JSON parse
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract ALL balanced {...} blocks, try each until one parses
    if parsed is None:
        candidates = []
        start = None
        depth = 0
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(cleaned[start:i+1])
                    start = None

        # Try candidates in reverse order — last complete block is most likely final answer
        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
                if "results" in parsed:
                    break
            except json.JSONDecodeError:
                continue

    if parsed is None:
        print("  WARNING: Could not parse JSON from response. Recording NaN for all tickers.")
        return [
            {"Ticker": t, "Sharpe": float("nan"), "Sortino": float("nan"), "Max_Drawdown": float("nan")}
            for t in tickers
        ]

    results = parsed.get("results", [])
    result_map = {r["Ticker"]: r for r in results if "Ticker" in r}

    records = []
    for ticker in tickers:
        if ticker in result_map:
            r = result_map[ticker]
            records.append({
                "Ticker":       ticker,
                "Sharpe":       r.get("Sharpe",       float("nan")),
                "Sortino":      r.get("Sortino",      float("nan")),
                "Max_Drawdown": r.get("Max_Drawdown", float("nan")),
            })
        else:
            print(f"  WARNING: {ticker} missing from model response. Recording NaN.")
            records.append({
                "Ticker":       ticker,
                "Sharpe":       float("nan"),
                "Sortino":      float("nan"),
                "Max_Drawdown": float("nan"),
            })

    return records


# =============================================================================
# OUTPUT WRITER
# =============================================================================

def write_trial_output(
    records: list[dict],
    condition_key: str,
    rep: int,
    is_draft: bool,
) -> str:
    """
    Write trial results to a CSV file.

    Draft runs (--tabs N) write to DRAFT_OUTPUT_PATH.
    Full runs write to TRIAL_OUTPUT_DIR/<condition>_rep<N>_<version>.csv.

    Parameters
    ----------
    records : list[dict]
        Parsed metric dicts from the model response.
    condition_key : str
        Condition label (C, D, E, or F).
    rep : int
        Replication number.
    is_draft : bool
        True if this is a test run.

    Returns
    -------
    str
        Path where the CSV was written.
    """
    df = pd.DataFrame(records, columns=["Ticker", "Sharpe", "Sortino", "Max_Drawdown"])

    if is_draft:
        output_path = DRAFT_OUTPUT_PATH
    else:
        os.makedirs(TRIAL_OUTPUT_DIR, exist_ok=True)
        filename = f"condition_{condition_key}_rep{rep}_{VERSION}.csv"
        output_path = os.path.join(TRIAL_OUTPUT_DIR, filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Output written to: {output_path}")
    return output_path


# =============================================================================
# SINGLE TRIAL RUNNER
# =============================================================================

def run_trial(condition_key: str, rep: int, tickers: list[str], is_draft: bool) -> None:
    """
    Execute a single trial for one condition and one replication.

    For no-tool conditions (C and D), a second cleanup API call is made
    automatically if the first response cannot be parsed as JSON. The cleanup
    call sends the model's own prose response back and asks it to reformat
    the final answers as JSON only.

    Parameters
    ----------
    condition_key : str
        One of "C", "D", "E", "F".
    rep : int
        Replication number (1-indexed).
    tickers : list[str]
        ETF tickers to include in this trial.
    is_draft : bool
        True if running in test mode (--tabs N).
    """
    print(f"\n--- Condition {condition_key} | Rep {rep} | {VERSION} ---")
    print(f"  Model      : {CONDITIONS[condition_key]['model']}")
    print(f"  Tool access: {CONDITIONS[condition_key]['tool_access']}")
    print(f"  System prompt: {CONDITIONS[condition_key]['system_prompt']}")
    print(f"  Tickers    : {tickers}")

    use_tools   = CONDITIONS[condition_key]["tool_access"]
    data_text   = read_workbook_as_text(WORKBOOK_PATH, tickers)
    user_prompt = build_user_prompt(data_text, tickers, use_tools=use_tools)

    print("  Calling API...")
    start = time.time()
    response_text = call_api(condition_key, user_prompt)
    elapsed = time.time() - start
    print(f"  API call complete ({elapsed:.1f}s)")

    # Save raw response for diagnostics
    raw_dir = os.path.join(TRIAL_OUTPUT_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filename = f"condition_{condition_key}_rep{rep}_{VERSION}_raw.txt"
    raw_path = os.path.join(raw_dir, raw_filename)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"  Raw response saved to: {raw_path}")

    # Attempt to parse first response
    records = parse_response(response_text, tickers)
    has_nan = any(
        pd.isna(r["Sharpe"]) or pd.isna(r["Sortino"]) or pd.isna(r["Max_Drawdown"])
        for r in records
    )

    # For no-tool conditions, run cleanup call if first parse failed or has NaNs
    if has_nan and not CONDITIONS[condition_key]["tool_access"]:
        print("  First response unparseable — running cleanup call...")
        start2 = time.time()
        cleanup_text = call_api_cleanup(condition_key, response_text, tickers)
        elapsed2 = time.time() - start2
        print(f"  Cleanup call complete ({elapsed2:.1f}s)")

        cleanup_filename = f"condition_{condition_key}_rep{rep}_{VERSION}_cleanup_raw.txt"
        cleanup_path = os.path.join(raw_dir, cleanup_filename)
        with open(cleanup_path, "w", encoding="utf-8") as f:
            f.write(cleanup_text)
        print(f"  Cleanup raw response saved to: {cleanup_path}")

        records = parse_response(cleanup_text, tickers)

    write_trial_output(records, condition_key, rep, is_draft)
    print(f"  Parsed {len(records)} ticker results.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    tickers  = TICKERS[:args.tabs] if args.tabs is not None else TICKERS
    is_draft = args.tabs is not None

    if not os.path.exists(WORKBOOK_PATH):
        raise FileNotFoundError(
            f"Input workbook not found at '{WORKBOOK_PATH}'. "
            "Run data_pull.py first."
        )

    if is_draft:
        print(f"TEST RUN: limiting to first {args.tabs} ETF tabs. Output -> draft path.\n")

    if args.all:
        api_conditions = ["C", "D", "E", "F"]
        for condition_key in api_conditions:
            for rep in range(1, N_REPLICATIONS + 1):
                run_trial(condition_key, rep, tickers, is_draft)
    else:
        condition_key = args.condition
        rep = args.rep
        if rep is None:
            raise ValueError("--rep is required when using --condition.")
        if rep < 1 or rep > N_REPLICATIONS:
            raise ValueError(f"--rep must be between 1 and {N_REPLICATIONS}.")
        run_trial(condition_key, rep, tickers, is_draft)

    print("\napi_agent.py complete.")


if __name__ == "__main__":
    main()
