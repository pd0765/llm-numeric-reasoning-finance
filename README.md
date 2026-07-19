# LLM Numeric Reasoning in Financial Calculations: A Controlled Experiment

A controlled experiment measuring whether Claude produces accurate multi-step quantitative financial calculations when asked to reason in prose versus when given a code execution tool. The core finding is unambiguous: **tool access is the only variable that matters. Prompt framing has no effect in either direction.**

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation — A Realistic Office Scenario](#motivation--a-realistic-office-scenario)
3. [Research Question](#research-question)
4. [Experimental Design](#experimental-design)
5. [Metrics](#metrics)
6. [Input Data](#input-data)
7. [Calculation Conventions](#calculation-conventions)
8. [User Prompts — Conditions A and B](#user-prompts--conditions-a-and-b)
9. [File Structure](#file-structure)
10. [Reproducing the Experiment](#reproducing-the-experiment)
11. [Results](#results)
12. [Key Findings](#key-findings)
13. [Limitations](#limitations)
14. [Versioning and Future Work](#versioning-and-future-work)

---

## Overview

This project benchmarks Claude Sonnet 4.6 (and Opus 4.6 in one condition) on a realistic financial calculation task: computing annualized Sharpe ratio, Sortino ratio, and Maximum Drawdown for 20 ETFs from monthly return data spanning 2019–2023.

Six conditions vary the platform, tool access, and system prompt. All other variables — model version, input data, metric definitions, calculation conventions, date range, and tickers — are held constant. This design isolates the effect of each dimension.

The experiment targets a question with direct relevance to financial services practitioners: when an LLM agent is asked to perform quantitative analysis, does giving it a code execution tool meaningfully change the reliability of its output, versus asking it to compute in prose?

---

## Motivation — A Realistic Office Scenario

An analyst at an asset management firm is asked to compute Sharpe ratios, Sortino ratios, and maximum drawdowns across a set of ETFs. Their first stop is claude.ai. They attach the workbook, submit a prompt with the required formulas, and receive a clean table — correct values, right signs, four decimal places. The workflow takes a few minutes and produces results the team is comfortable using.

A few weeks later, the team's usage approaches its subscription tier limit. To manage costs, the manager suggests switching to the API: *"Just take the same user prompt and drop it into the Python script. It's the same model."* The suggestion is intuitive. Same model, same prompt, same data — why would the output be any different?

What the transition silently discards is the execution environment. When the analyst attached the workbook in claude.ai, the platform was invoking internal analysis tools to process the file. The model was computing by executing code. A bare API call, without a code execution tool explicitly provisioned, removes that capability entirely. The model is now asked to produce 60 metrics across 59 monthly observations using nothing but prose arithmetic.

Two outcomes are possible from here, depending on how the API call is configured:

- **The obvious failure (Condition D in this experiment):** The system prompt instructs the model to write Python, but no execution environment is available. The model produces syntactically correct code that never runs. Every value in the output table is blank or missing. Someone will notice.

- **The dangerous failure (Condition C):** The system prompt makes no mention of Python. The model attempts the arithmetic in prose, accumulates error across 59 monthly observations, and returns a complete, well-formatted table — plausible magnitudes, correct signs, four decimal places. The numbers are wrong approximately 97% of the time. Nothing about the output signals a problem.

This experiment was designed to quantify exactly how wrong Condition C is, and to determine whether anything short of tool access — model selection, system prompt framing, explicit calculation instructions — can close the gap. The answer is no.

---

## Research Question

> Does providing a large language model with a code execution tool meaningfully improve the accuracy of quantitative financial calculations, compared to asking the model to reason through the arithmetic in natural language?

---

## Experimental Design

Six conditions, each run on the same 20-ticker dataset with the same metric definitions:

| Condition | Platform | Model | Tool Access | System Prompt |
|-----------|----------|-------|-------------|---------------|
| A | claude.ai (manual) | Claude Sonnet 4.6 | None\* | None |
| B | claude.ai (manual) | Claude Opus 4.6 | None\* | None |
| C | API | Claude Sonnet 4.6 | None | Neutral |
| D | API | Claude Sonnet 4.6 | None | Must use Python |
| E | API | Claude Sonnet 4.6 | Code execution | Neutral |
| F | API | Claude Sonnet 4.6 | Code execution | Must use Python |

\*Conditions A and B have no explicitly granted tools, but claude.ai silently uses internal analysis tools when a workbook is attached — an important confound discussed in [Key Findings](#key-findings).

**Replications:** Each API condition (C–F) is run 3 independent times at `temperature=0`. Conditions A and B are run 3 times each in the claude.ai interface.

### System prompt text

The two API system prompts, as defined in `config.py`:

**Neutral** (Conditions C and E):
```
You are a financial analyst assistant. The user will provide ETF return data and ask you
to compute performance metrics. Respond clearly and accurately.
```

**Must use Python** (Conditions D and F):
```
You are a financial analyst assistant. The user will provide ETF return data and ask you
to compute performance metrics. ALL calculations MUST be performed by writing and executing
a Python script. Do not perform any arithmetic manually or through prose reasoning. Respond
clearly and accurately.
```

**Design rationale:** Three dimensions are manipulated — platform/model, tool access, and system prompt — while input data is held constant across all cells. This gives clean attribution when results differ. The C vs. D contrast tests whether the system prompt alone ("use Python") has any effect without execution capability. The E vs. F contrast tests the same question when execution is available. Neither prompt dimension produces any effect in either direction.

---

## Metrics

Three metrics were selected to cover structurally distinct cognitive demands:

| Metric | Primary Challenge |
|--------|-------------------|
| Sharpe Ratio | Arithmetic drift, wrong annualization, risk-free rate handling |
| Sortino Ratio | Denominator confusion, MAR convention, partial computation |
| Maximum Drawdown | Path dependency, peak tracking, price vs. return level confusion |

Sharpe and Sortino share the same arithmetic framework (mean, standard deviation, per-period risk-free subtraction). Maximum Drawdown is structurally different — it requires the model to track the running peak of a price series across 59 monthly observations and identify the largest peak-to-trough decline. This tests algorithmic reasoning rather than formula application, and its errors tend to be different in character from the distribution-based metrics.

---

## Input Data

- **ETF universe:** 20 tickers — `SPY, QQQ, IWM, EFA, EEM, AGG, LQD, HYG, TLT, GLD, VNQ, XLE, XLF, XLV, XLK, XLU, XLI, XLP, VTV, VUG`
- **Price data source:** `yfinance` (adjusted closing prices, resampled to month-end)
- **Date range:** 2019-01-01 to 2023-12-31 (60 month-end observations, 59 return observations after differencing)
- **Risk-free rate:** FRED series `TB3MS` (3-Month Treasury Bill Secondary Market Rate, monthly), converted to per-period decimal by dividing by `(100 × 12)`
- **Workbook structure:** One tab per ETF with columns `Date`, `Price`, `Return`; one RF tab with columns `Date`, `RF_Annual_Pct`, `RF_Periodic_Decimal`
- **The workbook is built once by `data_pull.py` and held constant across all conditions**

The 3-Month T-Bill rate is the standard academic and practitioner reference for the risk-free rate when computing Sharpe ratios on equity or index return series. It is free, programmatically accessible via `fredapi`, and the most widely cited short-rate proxy in empirical finance.

---

## Calculation Conventions

These conventions are locked down in `config.py` and implemented identically in `utils.py`, which is imported by both `ground_truth.py` (the reference implementation) and `api_agent.py` (the tool-enabled conditions). They are reproduced here verbatim to make the ground truth reproducible without running the code.

| Convention | Value |
|------------|-------|
| Return type | Simple arithmetic (periodic), first observation dropped |
| Annualization factor | 12 (monthly data) |
| Sharpe denominator | Sample std of excess returns (`ddof=1`) |
| Sortino MAR | Per-period RF rate each period (not zero) |
| Sortino downside deviation | `clip(upper=0)` on `(return − MAR)`, then square, mean, sqrt — mean taken over all periods |
| Sortino numerator | `(mean(return) − mean(MAR)) × 12` |
| Max Drawdown | Price-level series, running peak via `cummax()`, expressed as negative decimal |
| Output precision | 4 decimal places for all three metrics |
| Scoring tolerance | ±0.001 to classify as "correct" |

**Why document conventions this explicitly?** Because the Sortino MAR convention is where the model silently went wrong in early trials — see [Finding 6: Formula Convention Drift](#finding-6-formula-convention-drift-silent-and-undetectable).

---

## User Prompts — Conditions A and B

Conditions A and B are manual claude.ai trials. The workbook `data/etf_returns.xlsx` is attached directly in the claude.ai interface. No system prompt is used.

### Original Prompt (used in early trials — produced Sortino formula drift)

```
You are given monthly ETF return data and a risk-free rate series. The attached workbook
contains one tab per ETF with columns Date, Price, and Return, plus an RF tab with columns
Date, RF_Annual_Pct, and RF_Periodic_Decimal. Using the data in the workbook, compute the
following annualized performance metrics for each ETF:

1. SHARPE RATIO — Subtract the per-period risk-free rate (RF_Periodic_Decimal) from each
monthly return to obtain excess returns. Compute the mean and sample standard deviation
(ddof=1) of the excess return series. Annualize: (mean_excess / std_excess) × sqrt(12).
Round to 4 decimal places.

2. SORTINO RATIO — Use the per-period risk-free rate as the Minimum Acceptable Return (MAR)
each period. Downside deviation: for each period, compute (return − MAR); retain only negative
values, square them, take the mean, then take the square root. Annualize by multiplying by
sqrt(12). Annualized numerator: (mean(return) − mean(MAR)) × 12. Sortino = annualized
numerator / annualized downside deviation. Round to 4 decimal places.

3. MAXIMUM DRAWDOWN — Compute from the PRICE series (not returns). At each date, drawdown =
(price − running_peak_price) / running_peak_price. Maximum drawdown = the minimum (most
negative) value of this drawdown series. Express as a negative decimal (e.g., −0.3412 means
−34.12% drawdown). Round to 4 decimal places.

Return all results in a summary table with one row per ETF and columns: Ticker, Sharpe,
Sortino, Max_Drawdown.
```

> **What went wrong:** Despite the explicit instruction to use the per-period RF rate as MAR, Claude Sonnet 4.6 defaulted to zero as the Minimum Acceptable Return — a common Sortino convention in textbooks and many online implementations. This produced Sortino values approximately 60% of the correct figures. The values were directionally correct, plausible in magnitude, and formatted to 4 decimal places — entirely indistinguishable from correct output without ground truth. This is the core danger documented in [Finding 6](#finding-6-formula-convention-drift-silent-and-undetectable).

### Corrected Prompt (used for all scored trials)

The Sharpe and Max Drawdown instructions are identical to the original. Only the Sortino instruction was changed:

```
2. SORTINO RATIO — Using the same excess returns computed for Sharpe (return −
RF_Periodic_Decimal each period): identify the periods where the excess return is negative.
Do NOT use zero as the threshold — the downside is defined relative to the risk-free rate,
not relative to zero. Downside deviation = sqrt(mean(negative_excess_returns²)) × sqrt(12),
where the mean is taken over all periods (not just the negative ones). Annualized numerator
= mean(excess_return) × 12. Sortino = annualized numerator / downside deviation. Round to
4 decimal places.
```

> **What changed and why it worked:** The revised instruction eliminates the phrase "Minimum Acceptable Return" and the word "MAR" entirely — both of which appear to trigger a learned association with a zero-benchmark convention. Instead, it anchors the downside deviation calculation to the excess return series already defined for Sharpe, making the shared-RF-rate framework explicit rather than relying on the model to interpret "MAR = RF" correctly. With this wording, Conditions A and B produced 100% correct output across all 3 replications per condition.

---

## File Structure

```
ai_agent_calculations/
├── config.py           — All experiment parameters: conditions, tickers, FRED series,
│                         date range, file paths, system prompt templates
├── utils.py            — Shared metric functions (Sharpe, Sortino, Max Drawdown)
│                         imported by ground_truth.py and api_agent.py
├── data_pull.py        — Pulls ETF prices via yfinance and RF rate via fredapi,
│                         builds data/etf_returns.xlsx (run once)
├── ground_truth.py     — Computes verified reference metrics, writes
│                         data/ground_truth.csv (run once)
├── api_agent.py        — Runs API conditions C–F; handles prompt construction,
│                         two-call cleanup for no-tool conditions, stdout capture
│                         for code execution conditions
├── score.py            — Scores trial CSVs against ground truth; classifies
│                         failure modes; writes per-trial scored CSVs and summary
├── main.py             — Orchestrator entry point
├── data/
│   ├── etf_returns.xlsx        — 21-sheet workbook (20 ETF tabs + RF tab)
│   └── ground_truth.csv        — Reference values for all 20 tickers
└── results/
    ├── condition_A_rep[1-3]_v1.csv        — Raw trial outputs
    ├── condition_B_rep[1-3]_v1.csv
    ├── condition_C_rep[1-3]_v1.csv
    ├── condition_D_rep[1-3]_v1.csv
    ├── condition_E_rep[1-3]_v1.csv
    ├── condition_F_rep[1-3]_v1.csv
    ├── *_scored.csv                        — Per-ticker error breakdown (60 rows each)
    ├── summary_v1.csv                      — Condition-level summary table
    └── raw/
        └── condition_[C-F]_rep[1-3]_v1_raw.txt   — Full model responses
```

**Note on `data/` and `results/`:** These directories are excluded from the repository. The workbook and ground truth are reproducible by running `python main.py --setup`. Trial outputs are reproducible by running the full experiment.

**Note on `.env`:** A `.env` file at the project root stores API keys and is excluded from the repository. Required keys:

```
FRED_API_KEY=your_fred_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

---

## Reproducing the Experiment

### Prerequisites

```bash
pip install -r requirements.txt
```

Create a `.env` file at the project root with your `FRED_API_KEY` and `ANTHROPIC_API_KEY`.

### One-time setup

```bash
python main.py --setup
```

This runs `data_pull.py` (builds the workbook) and `ground_truth.py` (computes reference values). Takes approximately 30 seconds.

### API conditions (C–F)

```bash
# Run all API conditions, all replications
python main.py --run-all

# Or run a single condition/rep
python main.py --condition C --rep 1
python main.py --condition E --rep 2
```

Each tool-enabled trial (E/F) takes approximately 2–5 minutes for 20 tickers. Each no-tool trial (C/D) is faster but includes a two-call cleanup step.

### Manual conditions (A and B)

Open claude.ai, attach `data/etf_returns.xlsx`, select the appropriate model (Sonnet 4.6 for A, Opus 4.6 for B), and submit the corrected prompt from [User Prompts — Conditions A and B](#user-prompts--conditions-a-and-b). Record results into a CSV with columns `Ticker, Sharpe, Sortino, Max_Drawdown` and save as `results/condition_A_rep1_v1.csv` (incrementing the rep number for each repetition).

### Scoring

```bash
python main.py --score-all   # scores all trial CSVs
python main.py --summary     # prints condition-level summary table
```

### Failure mode categories

| Category | Definition |
|----------|------------|
| `correct` | Within ±0.001 of ground truth |
| `small_error` | Absolute error in (0.001, 0.10) |
| `large_error` | Absolute error ≥ 0.10 |
| `missing` | NaN — tab dropped, parse failure, or no output produced |
| `sign_error` | Correct magnitude, wrong sign |

---

## Results

### Summary table

| Condition | Platform | Model | Tool Access | System Prompt | N Correct / Total | % Correct |
|-----------|----------|-------|-------------|---------------|-------------------|-----------|
| A | claude.ai | Sonnet 4.6 | None\* | None | 180 / 180 | 100% |
| B | claude.ai | Opus 4.6 | None\* | None | 180 / 180 | 100% |
| C | API | Sonnet 4.6 | None | Neutral | 5 / 180 | 2.8% |
| D | API | Sonnet 4.6 | None | Must use Python | 0 / 180 | 0% (all missing) |
| E | API | Sonnet 4.6 | Code execution | Neutral | 180 / 180 | 100% |
| F | API | Sonnet 4.6 | Code execution | Must use Python | 180 / 180 | 100% |

Each condition: 3 replications × 20 tickers × 3 metrics = 180 scored values.

### Condition C — error breakdown by metric (averaged across 3 reps)

| Metric | MAE | Small errors | Large errors | Correct |
|--------|-----|--------------|--------------|---------|
| Sharpe | 0.095 | ~11 | ~7 | ~1–2 |
| Sortino | 0.189 | ~8 | ~12 | 0 |
| Max Drawdown | 0.055 | ~16 | ~3 | 0–1 |

The model always produces a number — no missing values in Condition C — but the arithmetic is consistently wrong. Sortino carries the highest error, consistent with its greater computational complexity.

### Condition D — architecture mismatch

All 60 values (20 tickers × 3 metrics) are missing across all 3 replications. The model wrote syntactically valid Python scripts in response to the "must use Python" system prompt, but no execution environment was provisioned. The scripts were never run. Both the first and cleanup API calls returned only Python code.

---

## Key Findings

### Finding 1: Tool access is the only variable that matters

Conditions E and F both achieve 100% accuracy regardless of prompt framing. Conditions C and D both fail regardless of prompt framing. The system prompt has no measurable effect in either direction.

The result is cleanly separable: the C vs. D contrast (same tool absence, different prompt) produces no difference. The E vs. F contrast (same tool access, different prompt) produces no difference. The only split that matters is tool access on or off.

### Finding 2: The failure is arithmetic execution, not comprehension

The raw response files for Condition C show the model working through correct methodology in prose — right formulas, right alignment with the stated conventions, correct interpretation of the RF rate series. The errors are not conceptual. They accumulate across 59 monthly arithmetic operations per ticker, repeated for 20 tickers. Methodology is sound; execution is not.

### Finding 3: The dataset is not large

20 tickers × 60 rows × 3 columns = approximately 1,200 numbers in a single context window. This is not a long-context or big-data problem. Even at this modest scale, pure prose arithmetic fails approximately 97% of the time.

### Finding 4: Token exhaustion produces silent failure

In early trials, no-tool conditions (C and D) were given no explicit output format instruction. The model ignored the intended JSON output and instead defaulted to longhand prose arithmetic, showing every step. Across 20 tickers × 3 metrics × 59 observations, this consumed the available token budget before the model finished — producing no usable output. Nothing in the response signaled that it was incomplete. A human reviewer skimming the output would see confident-looking arithmetic and no indication that the answer was never delivered.

The fix for Conditions C and D was a two-call strategy: the first call lets the model work through its arithmetic, and a follow-up message requests only the final JSON. This makes the output parseable, but introduces its own complication — see Finding 5.

### Finding 5: The model confabulates under cleanup pressure

When the two-call cleanup strategy asks for final answers based on calculations the model never completed, the model generates plausible-looking but fabricated numbers. The confabulated output is structured, formatted to 4 decimal places, has correct signs, and carries magnitudes consistent with what a financial professional would expect for these ETFs during 2019–2023. It is entirely indistinguishable from correct output without an independent reference.

This is arguably the most consequential finding for financial services practitioners: a no-tool LLM agent, when pushed past its arithmetic capacity, does not refuse and does not flag uncertainty. It generates a confident-looking result. The failure is invisible at the surface.

### Finding 6: Formula convention drift — silent and undetectable

In the first version of the Conditions A and B prompt, the Sortino instruction explicitly stated "Use the per-period risk-free rate as the Minimum Acceptable Return (MAR) each period." Despite this instruction, Claude Sonnet 4.6 defaulted to zero as the MAR — a common convention in many textbook and online Sortino implementations. The resulting Sortino values were approximately 60% of the correct figures: wrong magnitude, correct sign, plausible appearance.

This error was undetectable without ground truth. The values looked like Sortino ratios. They were positive where they should be positive, higher for better-performing ETFs, formatted to 4 decimal places. The only way to catch the error was to verify against independently computed reference values.

The fix was to eliminate the words "Minimum Acceptable Return" and "MAR" from the prompt entirely and instead anchor the downside calculation to the excess return series already defined for Sharpe. This removed the ambiguous terminology that triggered the model's learned association with a zero-threshold convention. With the revised wording, all Sortino values were exact to ground truth.

**Implication:** In formula-intensive financial tasks, the risk is not that the model makes an obvious mistake. The risk is that it applies a plausible but wrong convention — one that matches a real-world alternative — in a way that produces numbers too reasonable to flag on inspection. Verification must happen at the formula specification level, not the output level.

### Finding 7: The D/F contrast isolates the role of execution capability

Condition D instructs the model to use Python and provides no execution environment. Condition F gives the same instruction with code execution enabled. D produces 0% accuracy, 100% missing. F produces 100% accuracy. The instruction adds no value without the capability to execute it.

### Finding 8: Conditions A and B — the silent tool confound

Conditions A and B are nominally "no tools" — the experimental design intended them as a claude.ai analog to Conditions C and D. In practice, claude.ai silently uses internal analysis tools when a workbook is attached. The model executes code internally without disclosing this in the response. This likely explains the 100% accuracy in these conditions and constitutes a meaningful confound: A and B cannot be treated as true no-tool conditions.

This confound is not a design flaw so much as an empirical finding in itself: the claude.ai interface provides execution capability by default when structured data is present, regardless of whether tools are explicitly granted by the user.

### Finding 9: Tool-enabled responses are structurally different

Raw response files for Conditions E and F open with one sentence — typically acknowledging the task — followed immediately by code execution. The model delegates arithmetic to the tool without producing any prose. Condition C raw files run to hundreds of lines of manual arithmetic. The behavioral difference is categorical, not a matter of degree.

---

## Limitations

**Sample size.** Six conditions, 3 replications each, 20 tickers, 59 return observations. The experiment demonstrates a consistent pattern but is not a large-sample study.

**Single model family.** All conditions use Claude (Sonnet 4.6 or Opus 4.6). The extent to which findings generalize to other model families is unknown.

**Single task type.** The metrics tested (Sharpe, Sortino, Max Drawdown) share a common structure: multi-step arithmetic over a fixed-length time series. Other financial calculation tasks — option pricing, regression, optimization — may exhibit different failure modes.

**Temperature and stochasticity.** API conditions are run at `temperature=0`. Claude.ai conditions cannot have temperature fixed; this introduces uncontrolled variance in A and B. The 3-rep structure partially mitigates this.

**Conditions A and B confound.** As noted in Finding 8, claude.ai's internal tool use when handling file attachments means A and B cannot be interpreted as pure no-tool conditions.

**v1 scope.** All metrics are full-period scalars (one number per ticker per metric). Rolling calculations, multi-period comparisons, or portfolio-level aggregation are not tested in this version.

---

## Versioning and Future Work

**v1 (this repository):** Single full-period metrics across all 20 ETFs, 2019–2023. One scalar per ticker per metric. Six conditions, 3 replications each.

**v2 (planned):** Rolling 12-month trailing metrics, recalculated monthly. This introduces a qualitatively different failure mode: does the model correctly advance the window, or does it anchor to a fixed start date? Rolling calculations substantially increase the task complexity. The versioning boundary is intentional — v1 isolates numeric reasoning errors cleanly, and v2 adds window management errors as a second dimension.

The `ROLLING_WINDOW_MONTHS` parameter is already stubbed in `config.py`.
