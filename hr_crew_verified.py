"""
hr_crew_verified.py

Purpose:
    People Analytics crew with a code-based verification gate. A writer
    agent drafts an executive summary plus a structured list of claims;
    verifier.py checks every claim against the data. Failed drafts are
    sent back to the writer with the exact problems (up to MAX_ATTEMPTS).
    If it still fails, the report is marked NEEDS HUMAN REVIEW.
    Data: public, fictional IBM HR Analytics dataset (Kaggle).

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: crewai, python-dotenv, verifier.py, hr_employees_trimmed.csv

Usage:
    python hr_crew_verified.py
"""

import json
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

from verifier import (
    DATA_FILE, Draft, compute_stats, rankings, verify_draft,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
MAX_ATTEMPTS = 3
REPORT_FILE = Path("hr_report_verified.md")

WRITER_TASK = (
    "Call department_stats, then write an executive summary of about 120 "
    "words covering: which department has the highest attrition, how "
    "overtime compares across departments, and one recommendation. Rules: "
    "(1) Quote numbers exactly as the tool returns them. (2) Do not use "
    "judgment words such as notably, significant, or concerning, and do "
    "not claim any correlation or cause. (3) Do not mention the word "
    "count or any number that is not a department statistic. (4) Return "
    "two fields. 'summary' is the text. 'claims' has one entry for EVERY "
    "number in the summary and EVERY highest or lowest statement. A metric "
    "claim has kind 'metric', department, metric, and value. A ranking "
    "claim has kind 'ranking', department, metric, and position (highest "
    "or lowest). Metric names must be exactly: headcount, leavers, "
    "attrition_rate_pct, avg_years_at_company, overtime_pct."
)


@tool("department_stats")
def department_stats() -> str:
    """Return exact per-department workforce statistics and rankings.

    Values are computed in code over every row of the HR dataset, so they
    are exact. Includes a rankings section naming the highest and lowest
    department for attrition, overtime, and headcount.
    """
    stats = compute_stats()
    return json.dumps(
        {"departments": stats, "rankings": rankings(stats)}, indent=2
    )


def draft_summary(llm: LLM, feedback: str) -> Draft | None:
    """Run the writer once and return its structured draft (or None)."""
    writer = Agent(
        role="Executive Summary Writer",
        goal="Write accurate, concise executive summaries with checkable claims",
        backstory=(
            "You write for senior leaders. You use only numbers from the "
            "tool and list every number and ranking as a structured claim."
        ),
        tools=[department_stats],
        llm=llm,
        verbose=True,
    )
    task = Task(
        description=WRITER_TASK + feedback,
        expected_output="A summary and a complete list of claims.",
        agent=writer,
        output_pydantic=Draft,
    )
    result = Crew(
        agents=[writer], tasks=[task], process=Process.sequential, verbose=True
    ).kickoff()

    try:
        return result.pydantic or Draft.model_validate_json(result.raw)
    except Exception:
        return None  # malformed output counts as a failed attempt


def build_report(draft: Draft | None, issues: list[str], attempts: int) -> str:
    """Assemble the final markdown report."""
    status = "VERIFIED" if not issues else "NEEDS HUMAN REVIEW"
    lines = [
        "# People Analytics Summary",
        f"**Status: {status}** (attempts used: {attempts})",
        "",
        draft.summary if draft else "(no valid draft was produced)",
        "",
    ]
    if draft:
        lines.append(f"Claims checked in code: {len(draft.claims)}")
    if issues:
        lines += ["", "## Unresolved issues"] + [f"- {i}" for i in issues]
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run prep_ibm_data.py.")

    llm = LLM(model=MODEL_NAME)
    stats = compute_stats()
    feedback = ""
    draft, issues = None, ["No draft produced."]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n=== Attempt {attempt} of {MAX_ATTEMPTS} ===")
        draft = draft_summary(llm, feedback)
        issues = (
            verify_draft(draft, stats) if draft
            else ["Output was not valid structured JSON."]
        )
        print("Verification issues:", issues or "none")
        if not issues:
            break
        feedback = (
            "\n\nYour previous draft failed automated verification. Fix "
            "these problems and return a new complete draft:\n- "
            + "\n- ".join(issues)
        )

    report = build_report(draft, issues, attempt)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("\n=== FINAL REPORT ===\n")
    print(report)
    print(f"\nSaved to {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()