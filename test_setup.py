from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

llm = LLM(model="gpt-4o-mini")
print(llm.call("Reply with exactly : setup works"))