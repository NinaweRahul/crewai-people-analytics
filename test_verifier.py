"""
test_verifier.py

Purpose:
    Tests verifier.py without any LLM. Confirms that (1) the computed
    statistics match the independently produced answer_key.json, (2) a
    correct draft passes, and (3) planted errors are always caught.

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: verifier.py, answer_key.json, hr_employees_trimmed.csv

Usage:
    python test_verifier.py
"""

import json
from pathlib import Path

from verifier import Claim, Draft, compute_stats, verify_draft

STATS = compute_stats()
results = []


def check(name: str, ok: bool) -> None:
    """Record and print one test result."""
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


def m(dept: str, metric: str, value: float) -> Claim:
    return Claim(kind="metric", department=dept, metric=metric, value=value)


def r(dept: str, metric: str, position: str) -> Claim:
    return Claim(kind="ranking", department=dept, metric=metric, position=position)


# ---- Test 1: computed stats match the independent answer key --------------
key = json.loads(Path("answer_key.json").read_text(encoding="utf-8"))
match = all(
    abs(STATS[d][k] - key[d][k]) < 0.0051 for d in key for k in key[d]
)
check("computed stats match answer_key.json", match)

# ---- Test 2: a correct draft passes ---------------------------------------
GOOD_TEXT = (
    "Sales has the highest attrition at 20.6% (92 leavers of 446). Human "
    "Resources has 19.0% attrition (12 leavers of 63) and the lowest "
    "overtime rate at 27.0%. Research & Development has 13.8% attrition "
    "(133 leavers of 961) with 28.2% overtime."
)
good_claims = [
    m("Sales", "attrition_rate_pct", 20.6), m("Sales", "leavers", 92),
    m("Sales", "headcount", 446),
    m("Human Resources", "attrition_rate_pct", 19.0),
    m("Human Resources", "leavers", 12), m("Human Resources", "headcount", 63),
    m("Human Resources", "overtime_pct", 27.0),
    m("Research & Development", "attrition_rate_pct", 13.8),
    m("Research & Development", "leavers", 133),
    m("Research & Development", "headcount", 961),
    m("Research & Development", "overtime_pct", 28.2),
    r("Sales", "attrition_rate_pct", "highest"),
    r("Human Resources", "overtime_pct", "lowest"),
]
issues = verify_draft(Draft(summary=GOOD_TEXT, claims=good_claims), STATS)
check("correct draft passes with no issues", issues == [])

# ---- Test 3: the three planted errors from the earlier checker test -------
BAD_TEXT = (
    "Sales has the highest attrition at 25.6% (92 leavers of 446). Human "
    "Resources has 19.0% attrition (12 leavers of 63) and the highest "
    "overtime rate at 40.0%. Research & Development has 13.8% attrition "
    "(133 leavers of 961) with 28.2% overtime."
)
bad_claims = [c for c in good_claims if not (
    (c.metric == "attrition_rate_pct" and c.department == "Sales" and c.kind == "metric")
    or (c.metric == "overtime_pct" and c.department == "Human Resources")
)]
bad_claims += [
    m("Sales", "attrition_rate_pct", 25.6),
    m("Human Resources", "overtime_pct", 40.0),
    r("Human Resources", "overtime_pct", "highest"),
]
bad_claims = [c for c in bad_claims if not (
    c.kind == "ranking" and c.department == "Human Resources" and c.position == "lowest"
)]
issues = verify_draft(Draft(summary=BAD_TEXT, claims=bad_claims), STATS)
for i in issues:
    print("      flagged:", i)
check("planted errors: exactly 3 issues flagged", len(issues) == 3)

# ---- Test 4: a number with no claim is caught -----------------------------
issues = verify_draft(
    Draft(summary=GOOD_TEXT + " Attrition rose 30.0% last year.", claims=good_claims),
    STATS,
)
check("unclaimed number is flagged", any("30.0" in i for i in issues))

# ---- Test 5: vague wording is caught --------------------------------------
issues = verify_draft(
    Draft(summary=GOOD_TEXT + " Overtime is notably high.", claims=good_claims),
    STATS,
)
check("vague wording is flagged", any("notably" in i for i in issues))

print(f"\n{sum(results)} of {len(results)} tests passed")