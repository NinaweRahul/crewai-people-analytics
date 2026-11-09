"""
verifier.py

Purpose:
    Deterministic fact-checking for the People Analytics crew. Computes
    per-department statistics in code, then verifies a structured draft
    (summary text plus a list of claims) against them. No LLM is involved
    in the checking, so the result is the same every time.

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: pydantic (installed with crewai)
"""

import csv
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

DATA_FILE = Path("hr_employees_trimmed.csv")

Metric = Literal[
    "headcount",
    "leavers",
    "attrition_rate_pct",
    "avg_years_at_company",
    "overtime_pct",
]

# Allowed difference between a claimed value and the true value.
TOLERANCE = {
    "headcount": 0,
    "leavers": 0,
    "attrition_rate_pct": 0.05,
    "avg_years_at_company": 0.005,
    "overtime_pct": 0.05,
}

# Wording that cannot be verified against numbers is flagged for review.
VAGUE_TERMS = [
    "notably", "significant", "substantial", "alarming",
    "concerning", "strongly", "clearly", "correlat",
]
RANKING_WORDS = r"\b(highest|lowest|largest|smallest|most|least)\b"


class Claim(BaseModel):
    """One checkable statement the writer makes in its summary."""
    kind: Literal["metric", "ranking"]
    department: str
    metric: Metric
    value: Optional[float] = None                        # metric claims
    position: Optional[Literal["highest", "lowest"]] = None  # ranking claims


class Draft(BaseModel):
    """What the writer agent must return."""
    summary: str
    claims: list[Claim]


def compute_stats(path: Path = DATA_FILE) -> dict:
    """Compute exact per-department statistics from every row of the file."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats = {}
    for dept in sorted({r["Department"] for r in rows}):
        group = [r for r in rows if r["Department"] == dept]
        n = len(group)
        leavers = sum(1 for r in group if r["Attrition"] == "Yes")
        overtime = sum(1 for r in group if r["OverTime"] == "Yes")
        years = sum(int(r["YearsAtCompany"]) for r in group)
        stats[dept] = {
            "headcount": n,
            "leavers": leavers,
            "attrition_rate_pct": round(leavers / n * 100, 1),
            "avg_years_at_company": round(years / n, 2),
            "overtime_pct": round(overtime / n * 100, 1),
        }
    return stats


def rankings(stats: dict) -> dict:
    """Name the highest and lowest department for key metrics."""
    out = {}
    for metric, label in [
        ("attrition_rate_pct", "attrition"),
        ("overtime_pct", "overtime"),
        ("headcount", "headcount"),
    ]:
        out[f"highest_{label}"] = max(stats, key=lambda d: stats[d][metric])
        out[f"lowest_{label}"] = min(stats, key=lambda d: stats[d][metric])
    return out


def verify_draft(draft: Draft, stats: dict) -> list[str]:
    """Return a list of problems found in the draft (empty list = passed)."""
    issues = []
    ranking_claims = 0

    # 1. Check every structured claim against the data.
    for c in draft.claims:
        if c.department not in stats:
            issues.append(f"Unknown department '{c.department}'.")
            continue

        if c.kind == "metric":
            if c.value is None:
                issues.append(f"Metric claim for {c.department} has no value.")
                continue
            actual = stats[c.department][c.metric]
            if abs(actual - c.value) > TOLERANCE[c.metric] + 1e-9:
                issues.append(
                    f"{c.department} {c.metric}: draft says {c.value}, "
                    f"data says {actual}."
                )
        else:
            ranking_claims += 1
            if c.position is None:
                issues.append(f"Ranking claim for {c.department} has no position.")
                continue
            pick = max if c.position == "highest" else min
            true_dept = pick(stats, key=lambda d: stats[d][c.metric])
            if true_dept != c.department:
                issues.append(
                    f"Ranking: draft says {c.department} has the {c.position} "
                    f"{c.metric}, data says {true_dept}."
                )

    # 2. Every number in the text must be covered by a metric claim.
    claimed_values = [
        c.value for c in draft.claims
        if c.kind == "metric" and c.value is not None
    ]
    for token in sorted(set(re.findall(r"\d+(?:\.\d+)?", draft.summary))):
        if not any(abs(float(token) - v) < 0.0051 for v in claimed_values):
            issues.append(f"Number {token} in the summary has no matching claim.")

    # 3. Every ranking word in the text needs a ranking claim.
    ranking_words = len(re.findall(RANKING_WORDS, draft.summary.lower()))
    if ranking_words > ranking_claims:
        issues.append(
            f"Summary has {ranking_words} ranking words but only "
            f"{ranking_claims} ranking claims."
        )

    # 4. Flag wording that cannot be checked against the numbers.
    lowered = draft.summary.lower()
    for term in VAGUE_TERMS:
        if term in lowered:
            issues.append(f"Unverifiable wording in the summary: '{term}'.")

    return issues