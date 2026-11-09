# People Analytics Agent Crew (CrewAI)

A multi-agent project that produces an executive summary of workforce
metrics, built to test how reliable LLM agents are at counting and
verifying numbers.

**Data:** the public, fictional IBM HR Analytics Employee Attrition
dataset from Kaggle (1,470 rows). Not real employee data. Only six
columns are used; age, gender, and marital status are deliberately
dropped as a data-minimization guardrail.

## What I found

| Stage | Result |
|---|---|
| Agents read raw CSV through a file tool | Covered 11%, 63%, and 39% of employees across 3 runs; different wrong numbers each run; the checker approved everything |
| Custom stats tool (counting done in Python) | Every number matched an independently computed answer key |
| LLM checker tested with planted errors (2 wrong numbers, 1 false ranking) | Caught the wrong numbers, but ranking claims and its own change log were inconsistent |
| Final design: LLM drafts, code verifies (verifier.py) | 5 of 5 unit tests pass identically every run, including the 3 planted errors; live runs: 2 passed on attempt 1, 1 needed 3 attempts |

Why the raw approach failed: the file tool returns 100 rows per call,
agents requested pages out of order and past the end of the file, and
hit the step limit, then answered from partial data. The fact-checker
used the same tool, so it confirmed the mistakes instead of catching them.

## Final design: the LLM drafts, code verifies

The LLM fact-checker was replaced with verifier.py. The writer returns a
summary plus a structured list of every number and ranking it claims. Code
then checks each claim against statistics computed from all 1,470 rows,
checks that every number and every highest/lowest word in the text has a
matching claim, and flags unverifiable wording. Failed drafts go back to
the writer with the exact problems, up to 3 attempts; otherwise the report
is marked NEEDS HUMAN REVIEW.

In one live run the writer needed 3 attempts: attempt 1 omitted a ranking
claim, attempt 2 omitted two numeric claims, and attempt 3 passed. The
numbers in the text were correct throughout; the gate enforced that every
statement had a checked claim behind it.

Takeaway: let code do the counting and the checking, let the LLM write,
and keep a human reviewing anything that reaches leadership.

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
- `python hr_crew_stats.py` crew with the department_stats tool and an LLM checker
- `python hr_crew_yaml.py` same crew configured via config/*.yaml
- `python test_checker.py` LLM fact-checker test with planted errors
- `python test_verifier.py` unit tests for the code-based verifier (no LLM, free)
- `python hr_crew_verified.py` final design: writer plus code verification

On Windows, run `$env:PYTHONUTF8 = "1"` first to avoid emoji encoding errors.

## Known limitations

- The verifier only checks what can be expressed as a claim. Vague
  judgments such as "high" or "elevated", and the quality of the
  recommendation, still need a human reader.
- The writer can ignore instructions; the gate protects the output, not
  the writer's behavior.
- Results vary between runs and the samples are small. These are
  observations, not benchmarks.