"""
test_checker.py

Purpose:
    Tests whether the fact-checker agent actually catches errors. It is
    given an executive summary with two deliberately wrong numbers and the
    department_stats tool. A working checker should flag both errors.

Planted errors (true values from answer_key.json):
    - Sales attrition: summary says 25.6%, truth is 20.6%
    - HR overtime: summary says 40.0%, truth is 27.0%

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: crewai, python-dotenv, hr_crew_stats.py in the same folder

Usage:
    python test_checker.py
"""

from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv

from hr_crew_stats import MODEL_NAME, department_stats

TAMPERED_SUMMARY = (
    "Sales has the highest attrition at 25.6% (92 leavers of 446). "
    "Human Resources has 19.0% attrition (12 leavers of 63) and the "
    "highest overtime rate at 40.0%. Research & Development has 13.8% "
    "attrition (133 leavers of 961) with 28.2% overtime."
)


def main() -> None:
    load_dotenv()
    llm = LLM(model=MODEL_NAME)

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

    task = Task(
        description=(
            "Call department_stats and verify the summary below. Verify "
            "every number AND every comparison or ranking claim.\n\n"
            f"SUMMARY TO VERIFY:\n{TAMPERED_SUMMARY}\n\n"
            "Output two sections: (1) 'CORRECTED SUMMARY' with all numbers "
            "fixed and any unsupported claim rewritten, and (2) "
            "'CORRECTIONS MADE' listing each number or claim you changed, "
            "old and new. If nothing was wrong, say so."
        ),
        expected_output="Two markdown sections: CORRECTED SUMMARY and CORRECTIONS MADE.",
        agent=checker,
    )

    result = Crew(agents=[checker], tasks=[task], verbose=True).kickoff()
    print("\n=== CHECKER OUTPUT ===\n")
    print(result.raw)


if __name__ == "__main__":
    main()