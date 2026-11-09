"""
prep_ibm_data.py

Purpose:
    Trims the public IBM HR Analytics Attrition dataset (a fictional
    dataset published on Kaggle) to the six columns needed for this
    project, and computes an answer key of true per-department statistics.
    Demographic columns (age, gender, marital status) are deliberately
    excluded as a data-minimization guardrail.

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: standard library only

Usage:
    python prep_ibm_data.py
"""

import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_FILE = Path("WA_Fn-UseC_-HR-Employee-Attrition.csv")
TRIMMED_FILE = Path("hr_employees_trimmed.csv")
KEY_FILE = Path("answer_key.json")

KEEP_COLUMNS = [
    "EmployeeNumber",
    "Department",
    "JobRole",
    "OverTime",
    "YearsAtCompany",
    "Attrition",
]


def load_rows(path: Path) -> list[dict]:
    """Read the source CSV and keep only the approved columns."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [{col: row[col] for col in KEEP_COLUMNS} for row in reader]


def compute_answer_key(rows: list[dict]) -> dict:
    """Compute the true per-department statistics.

    Definitions (the agents are given these exact definitions too):
        attrition_rate_pct = employees with Attrition == Yes / headcount * 100
        avg_years_at_company = mean YearsAtCompany over all employees
        overtime_pct = employees with OverTime == Yes / headcount * 100
    """
    key = {}
    for dept in sorted({r["Department"] for r in rows}):
        group = [r for r in rows if r["Department"] == dept]
        n = len(group)
        leavers = sum(1 for r in group if r["Attrition"] == "Yes")
        overtime = sum(1 for r in group if r["OverTime"] == "Yes")
        years = sum(int(r["YearsAtCompany"]) for r in group)
        key[dept] = {
            "headcount": n,
            "leavers": leavers,
            "attrition_rate_pct": round(leavers / n * 100, 1),
            "avg_years_at_company": round(years / n, 2),
            "overtime_pct": round(overtime / n * 100, 1),
        }
    return key


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"{SOURCE_FILE} not found in this folder.")

    rows = load_rows(SOURCE_FILE)

    with TRIMMED_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    key = compute_answer_key(rows)
    KEY_FILE.write_text(json.dumps(key, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {TRIMMED_FILE}")
    print(json.dumps(key, indent=2))


if __name__ == "__main__":
    main()