"""
file_summarizer.py

Purpose:
    Demonstrates giving a CrewAI agent a tool. A document analyst agent
    uses FileReadTool to read a local file and summarize it. Reads
    article.md, which first_crew.py produced.

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: crewai, crewai-tools, python-dotenv (OPENAI_API_KEY in .env)

Usage:
    python file_summarizer.py
"""

from pathlib import Path

from crewai import Agent, Crew, LLM, Task
from crewai_tools import FileReadTool
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
INPUT_FILE = Path("article.md")


def build_crew(llm: LLM, input_file: Path) -> Crew:
    """Create an analyst agent equipped with a file reader tool.

    Args:
        llm: The language model the agent will use.
        input_file: Path of the file the agent is allowed to read.

    Returns:
        A Crew ready to be started with kickoff().
    """
    # The tool is locked to one file path. Restricting what a tool can reach
    # is a basic guardrail: the agent cannot wander through the file system.
    file_reader = FileReadTool(file_path=str(input_file))

    analyst = Agent(
        role="Senior Document Analyst",
        goal="Read the provided file and summarize its key points accurately",
        backstory=(
            "You are a meticulous analyst who reads documents closely and "
            "reports only what the text actually says, never guessing."
        ),
        tools=[file_reader],  # only the tools this job needs
        llm=llm,
        verbose=True,
    )

    summary_task = Task(
        description=(
            "Use the file reader tool to read the file, then write a "
            "5-bullet summary. Use only information found in the file."
        ),
        expected_output="A markdown list of exactly 5 concise bullet points.",
        agent=analyst,
    )

    return Crew(agents=[analyst], tasks=[summary_task], verbose=True)


def main() -> None:
    """Validate the input file, run the crew, and print the summary."""
    load_dotenv()

    # Fail early with a clear message instead of a confusing agent error.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run first_crew.py first to create it."
        )

    crew = build_crew(LLM(model=MODEL_NAME), INPUT_FILE)
    result = crew.kickoff()

    print("\n=== SUMMARY ===\n")
    print(result.raw)


if __name__ == "__main__":
    main()