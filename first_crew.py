"""
first_crew.py

Purpose:
    Minimal CrewAI example: one agent, one task, one crew.
    A "Senior Technical Writer" agent writes a short beginner-friendly
    article on a topic passed in at runtime, and the result is saved
    to article.md.

Author: Rahul Ninawe
Created: 2025-11-09
Python: 3.12
Requires: crewai, python-dotenv (OPENAI_API_KEY set in .env)

Usage:
    python first_crew.py
"""

from pathlib import Path

from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"           # cheap and good enough for learning
TOPIC = "AI agents"                  # change this to reuse the same crew
OUTPUT_FILE = Path("article.md")     # where the final article is saved


def build_crew(llm: LLM) -> Crew:
    """Create the agent, its task, and the crew that runs them.

    Args:
        llm: The language model the agent will use.

    Returns:
        A Crew ready to be started with kickoff().
    """
    # Agent: role, goal, and backstory become the agent's system prompt,
    # so they control most of its behavior. {topic} is filled in at kickoff.
    writer = Agent(
        role="Senior Technical Writer",
        goal="Craft a clear, publication-ready article on {topic}",
        backstory=(
            "You are a seasoned technical writer who explains complex AI "
            "topics in plain language. You structure articles with a short "
            "intro, clear sections, and a practical takeaway."
        ),
        llm=llm,
        verbose=True,  # print the agent's reasoning while it works
    )

    # Task: the actual assignment. expected_output tells the agent what
    # "done" looks like, which keeps results consistent between runs.
    writing_task = Task(
        description=(
            "Write a short article (about 300 words) explaining {topic} "
            "to a beginner."
        ),
        expected_output=(
            "A well-structured markdown article with a title, "
            "3 sections, and a conclusion."
        ),
        agent=writer,
    )

    # Crew: ties agents and tasks together and controls execution order.
    return Crew(agents=[writer], tasks=[writing_task], verbose=True)


def main() -> None:
    """Load config, run the crew, print the result, and save it to disk."""
    # Reads OPENAI_API_KEY from the .env file into environment variables,
    # so the key never appears in the source code.
    load_dotenv()

    llm = LLM(model=MODEL_NAME)
    crew = build_crew(llm)

    # inputs fills the {topic} placeholders in the agent and task text.
    result = crew.kickoff(inputs={"topic": TOPIC})

    print("\n=== FINAL OUTPUT ===\n")
    print(result.raw)

    # Save the output; the file reader tool in the next step will use it.
    OUTPUT_FILE.write_text(result.raw, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_FILE.resolve()}")


# Only run when executed directly (python first_crew.py),
# not when this file is imported by another script.
if __name__ == "__main__":
    main()