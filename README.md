# People Analytics Agent Crew (CrewAI)

A multi-agent crew (analyst, writer, fact-checker) that produces an
executive summary of workforce metrics, built to test how reliable
LLM agents are at counting and verifying numbers.

**Data:** the public, fictional IBM HR Analytics Employee Attrition
dataset from Kaggle (1,470 rows). Not real employee data. Only six
columns are used; age, gender, and marital status are deliberately
dropped as a data-minimization guardrail.

## What I found

| Stage | Result |
|---|---|
| Agents read raw CSV through a file tool | Covered 11%, 63%, and 39% of employees across 3 runs; different wrong numbers each run; the checker approved everything |
| Custom stats tool (counting done in Python) | Every number matched an independently computed answer key |
| Checker tested with planted errors (2 wrong numbers, 1 false ranking) | Caught both numbers in all runs; ranking claims and its own change log were inconsistent until rankings were also computed in code |
| Final YAML-configured crew | All numbers correct; one vague claim ("notably high" overtime) still approved |

Why the raw approach failed: the file tool returns 100 rows per call,
agents requested pages out of order and past the end of the file, and
hit the step limit, then answered from partial data. The fact-checker
used the same tool, so it confirmed the mistakes instead of catching them.

Takeaway: let code do the counting, let the LLM interpret, and test the
checker with deliberate errors. LLM review alone is not a safe gate for
numbers that reach leadership; a human should review before use.

## Setup

1. Python 3.12 (CrewAI does not support 3.14)
2. `py -3.12 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. Create `.env` with `OPENAI_API_KEY=your-key`
5. Download the dataset from Kaggle (pavansubhasht/ibm-hr-analytics-attrition-dataset)
   and place the CSV in this folder
6. `python prep_ibm_data.py` to trim the data and write answer_key.json

## Run

- `python hr_crew_raw.py` raw-CSV baseline (inaccurate by design)
- `python hr_crew_stats.py` crew with the department_stats tool
- `python hr_crew_yaml.py` same crew configured via config/*.yaml
- `python test_checker.py` fact-checker test with planted errors

On Windows, run `$env:PYTHONUTF8 = "1"` first to avoid emoji encoding errors.

## Known limitations and next steps

- The checker sometimes mis-describes its own changes or approves vague
  claims; the next step is a code-based claim verifier.
- Results vary between runs; these are small samples, not benchmarks.