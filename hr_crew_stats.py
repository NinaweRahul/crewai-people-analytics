"""
hr_crew_stats.py

Purpose:
    People Analytics crew with a custom statistics tool. Instead of
    reading raw CSV rows (which the baseline crew did inaccurately),
    agents call department_stats(), which computes metrics in Python.
    The LLM interprets and writes; the code does the counting.
    Data: public, fictional IBM HR Analytics dataset (Kaggle).

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: crewai, python-dotenv (OPENAI_API_KEY in .env)

Usage:
    python prep_ibm_data.py   (once, to create the trimmed data)
    python hr_crew_stats.py
"""

import csv
import json
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
DATA_FILE = Path("hr_employees_trimmed.csv")
REPORT_FILE = Path("hr_report_stats.md")


@tool("department_stats")
def department_stats() -> str:
    """Compute per-department workforce statistics from the HR dataset.

    Returns JSON with, for each department: headcount, leavers,
    attrition_rate_pct, avg_years_at_company, and overtime_pct, plus 
    a rankings section naming the department with the highest and 
    lowest attrition, overtime, and headcount. Values are
    calculated in code over every row of the file, so they are exact.
    """
    # Guardrail: this tool can only ever read one fixed, public file.
    with DATA_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    result = {}
    for dept in sorted({r["Department"] for r in rows}):
        group = [r for r in rows if r["Department"] == dept]
        n = len(group)
        leavers = sum(1 for r in group if r["Attrition"] == "Yes")
        overtime = sum(1 for r in group if r["OverTime"] == "Yes")
        years = sum(int(r["YearsAtCompany"]) for r in group)
        result[dept] = {
            "headcount": n,
            "leavers": leavers,
            "attrition_rate_pct": round(leavers / n * 100, 1),
            "avg_years_at_company": round(years / n, 2),
            "overtime_pct": round(overtime / n * 100, 1),
        }
    # Rankings computed in code so agents never derive them by eye.
    rankings = {
        "highest_attrition": max(result, key=lambda d: result[d]["attrition_rate_pct"]),
        "lowest_attrition": min(result, key=lambda d: result[d]["attrition_rate_pct"]),
        "highest_overtime": max(result, key=lambda d: result[d]["overtime_pct"]),
        "lowest_overtime": min(result, key=lambda d: result[d]["overtime_pct"]),
        "largest_headcount": max(result, key=lambda d: result[d]["headcount"]),
        "smallest_headcount": min(result, key=lambda d: result[d]["headcount"]),
    }
    return json.dumps({"departments": result, "rankings": rankings}, indent=2)


def build_crew(llm: LLM) -> Crew:
    """Create the three agents, their tasks, and the sequential crew."""
    analyst = Agent(
        role="Senior People Analytics Analyst",
        goal="Report accurate per-department workforce metrics",
        backstory=(
            "You are a careful people analytics analyst. You always use the "
            "department_stats tool for numbers and never calculate by hand."
        ),
        tools=[department_stats],
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Executive Summary Writer",
        goal="Turn analysis results into a clear, concise executive summary",
        backstory=(
            "You write for senior leaders. You use only the numbers you are "
            "given and keep the message short and decision-oriented."
        ),
        llm=llm,
        verbose=True,
    )

    checker = Agent(
        role="Head of Data Quality and Fact-Checking",
        goal="Verify every number in the summary against the tool output",
        backstory=(
            "You are a skeptical reviewer. You call department_stats "
            "yourself and compare every figure in the summary to it. You "
            "also check every comparison or ranking claim (such as "
            "'highest', 'lowest', or 'correlates with') against the "
            "numbers, and you rewrite any claim the numbers do not support."
        ),
        tools=[department_stats],
        llm=llm,
        verbose=True,
    )

    analysis_task = Task(
        description=(
            "Call department_stats and report, for each department: "
            "headcount, leavers, attrition_rate_pct, avg_years_at_company, "
            "and overtime_pct, exactly as the tool returns them."
        ),
        expected_output="A table with one row per department and the five metrics.",
        agent=analyst,
    )

    writing_task = Task(
        description=(
            "Write a 150-word executive summary: which departments have the "
            "highest attrition, how overtime relates to it, and one "
            "recommendation. Quote exact numbers from the analysis."
        ),
        expected_output="A short markdown executive summary with exact figures.",
        agent=writer,
        context=[analysis_task],
    )

    checking_task = Task(
        description=(
            "Call department_stats yourself and verify every number in the "
            "executive summary. Output two sections: (1) 'CORRECTED "
            "SUMMARY' with all numbers fixed, and (2) 'CORRECTIONS MADE' "
            "listing each number you changed, old value and new value. If "
            "nothing was wrong, say so."
        ),
        expected_output="Two markdown sections: CORRECTED SUMMARY and CORRECTIONS MADE.",
        agent=checker,
        context=[analysis_task, writing_task],
    )

    return Crew(
        agents=[analyst, writer, checker],
        tasks=[analysis_task, writing_task, checking_task],
        process=Process.sequential,
        verbose=True,
    )


def main() -> None:
    load_dotenv()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run prep_ibm_data.py first."
        )

    result = build_crew(LLM(model=MODEL_NAME)).kickoff()

    REPORT_FILE.write_text(result.raw, encoding="utf-8")
    print("\n=== FINAL REPORT ===\n")
    print(result.raw)
    print(f"\nSaved to {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()