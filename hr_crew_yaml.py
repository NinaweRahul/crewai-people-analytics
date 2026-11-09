"""
hr_crew_yaml.py

Purpose:
    People Analytics crew configured through YAML. Agent and task
    definitions live in config/agents.yaml and config/tasks.yaml; this
    script only loads them, attaches tools, and runs the crew. Reuses the
    department_stats tool from hr_crew_stats.py, so all numbers are
    computed in code. Data: public, fictional IBM HR Analytics dataset.

Author: Rahul Ninawe
Created: 2026-11-09
Python: 3.12
Requires: crewai, python-dotenv, pyyaml, hr_crew_stats.py in this folder

Usage:
    python prep_ibm_data.py   (once, to create the trimmed data)
    python hr_crew_yaml.py
"""

from pathlib import Path

import yaml
from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv

from hr_crew_stats import MODEL_NAME, DATA_FILE, department_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent / "config"
REPORT_FILE = Path("hr_report_yaml.md")

# Tools stay in code because they are Python objects, not text. Only the
# agents listed here get a tool; the writer deliberately has none.
AGENT_TOOLS = {
    "analyst": [department_stats],
    "checker": [department_stats],
}


def load_yaml(name: str) -> dict:
    """Load one YAML config file from the config folder."""
    with (CONFIG_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_crew(llm: LLM) -> Crew:
    """Build agents and tasks from YAML and return the sequential crew."""
    agent_cfg = load_yaml("agents.yaml")
    task_cfg = load_yaml("tasks.yaml")

    agents = {
        name: Agent(
            **cfg,                       # role, goal, backstory from YAML
            tools=AGENT_TOOLS.get(name, []),
            llm=llm,
            verbose=True,
        )
        for name, cfg in agent_cfg.items()
    }

    # Tasks are created in file order, so any task listed in "context"
    # already exists by the time a later task refers to it.
    tasks = {}
    for name, cfg in task_cfg.items():
        tasks[name] = Task(
            description=cfg["description"],
            expected_output=cfg["expected_output"],
            agent=agents[cfg["agent"]],
            context=[tasks[c] for c in cfg.get("context", [])],
        )

    return Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
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