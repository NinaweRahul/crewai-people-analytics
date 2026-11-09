"""
hr_crew_raw.py

Purpose:
    Baseline People Analytics crew. Three agents (analyst, writer,
    fact-checker) work from the RAW trimmed CSV using only a file reader
    tool, with no calculation tools. Used to measure how accurate LLMs
    are at counting and averaging over raw rows, before a stats tool is
    added. Data: public, fictional IBM HR Analytics dataset (Kaggle).

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: crewai, crewai-tools, python-dotenv (OPENAI_API_KEY in .env)

Usage:
    python prep_ibm_data.py   (once, to create the trimmed data)
    python hr_crew_raw.py
"""

from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import FileReadTool
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
DATA_FILE = Path("hr_employees_trimmed.csv")
REPORT_FILE = Path("hr_report_raw.md")

# Shared so every agent uses identical metric definitions.
DEFINITIONS = (
    "Data columns: EmployeeNumber, Department, JobRole, OverTime (Yes/No), "
    "YearsAtCompany (whole years), Attrition (Yes/No). Each row is one "
    "employee. Definitions per Department: headcount = number of rows; "
    "leavers = rows with Attrition = Yes; attrition_rate_pct = leavers / "
    "headcount * 100; avg_years_at_company = mean YearsAtCompany over all "
    "rows; overtime_pct = rows with OverTime = Yes / headcount * 100."
)


def build_crew(llm: LLM, data_file: Path) -> Crew:
    """Create the three agents, their tasks, and the sequential crew."""
    # Guardrail: tools can only read this one trimmed, public file.
    reader = FileReadTool(file_path=str(data_file))

    analyst = Agent(
        role="Senior People Analytics Analyst",
        goal="Compute per-department workforce metrics from the CSV",
        backstory=(
            "You are a careful people analytics analyst. You count rows "
            "methodically and never estimate numbers you can calculate."
        ),
        tools=[reader],
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
        goal="Verify every number in the summary against the raw data",
        backstory=(
            "You are a skeptical reviewer. You independently recompute each "
            "figure from the source data and correct anything that is wrong."
        ),
        tools=[reader],
        llm=llm,
        verbose=True,
    )

    analysis_task = Task(
        description=(
            f"{DEFINITIONS} Read the CSV and, for each department, report: "
            "headcount, leavers, attrition_rate_pct (1 decimal), "
            "avg_years_at_company (2 decimals), and overtime_pct (1 decimal)."
        ),
        expected_output="A table with one row per department and the five metrics.",
        agent=analyst,
    )

    writing_task = Task(
        description=(
            "Write a 150-word executive summary of the department metrics: "
            "which departments have the highest attrition, how overtime "
            "relates to it, and one recommendation. Quote exact numbers "
            "from the analysis."
        ),
        expected_output="A short markdown executive summary with exact figures.",
        agent=writer,
        context=[analysis_task],   # receives the analyst's output
    )

    checking_task = Task(
        description=(
            f"{DEFINITIONS} Re-read the CSV yourself and verify every number "
            "in the executive summary. Output two sections: (1) "
            "'CORRECTED SUMMARY' with all numbers fixed, and (2) "
            "'CORRECTIONS MADE' listing each number you changed, old value "
            "and new value. If nothing was wrong, say so."
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

    crew = build_crew(LLM(model=MODEL_NAME), DATA_FILE)
    result = crew.kickoff()

    REPORT_FILE.write_text(result.raw, encoding="utf-8")
    print("\n=== FINAL REPORT ===\n")
    print(result.raw)
    print(f"\nSaved to {REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()