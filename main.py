from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# 2️⃣ Define structured output using Pydantic
# -----------------------------
class Analyzer(BaseModel):
    market_potential: str
    competitors: List[str]
    monetization_strategy: str
    mvp_features: List[str]
    risks: List[str]

# -----------------------------
# 3️⃣ Create PydanticOutputParser
# -----------------------------
parser = PydanticOutputParser(pydantic_object=Analyzer)
format_instructions = parser.get_format_instructions()

# -----------------------------
# 4️⃣ Create Mistral AI model instance
# -----------------------------
model = ChatMistralAI(
    model="mistral-large-2512",
    temperature=0.7,
)

# -----------------------------
# 5️⃣ Create PromptTemplate
# -----------------------------
prompt_template = """
You are a startup analyst. Based on the following inputs:

Startup Idea: {startup_idea}
Target User: {target_user}
Country: {country}

Provide a structured JSON output with the following fields:
- market_potential: short description of market opportunity
- competitors: list of main competitors
- monetization_strategy: how the startup can earn revenue
- mvp_features: main features for MVP
- risks: list of key risks

Return ONLY valid JSON that matches the structure.

{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["startup_idea", "target_user", "country"],
    partial_variables={"format_instructions": format_instructions},
    template=prompt_template
)

# -----------------------------
# 6️⃣ User input
# -----------------------------
startup_idea = input("Enter Your Startup Idea: ")
target_users = input("Enter Target Audience: ")
cont = input("Enter Country: ")

# -----------------------------
# 7️⃣ Generate AI output
# -----------------------------
filled_prompt = prompt.format(
    startup_idea=startup_idea,
    target_user=target_users,
    country=cont
)

response = model.invoke(filled_prompt)

# -----------------------------
# 8️⃣ Parse output (strip markdown fences if present)
# -----------------------------
try:
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()
    structured_output = parser.parse(raw)
except Exception as e:
    print("Parsing error:", e)
    structured_output = None

# -----------------------------
# 9️⃣ Print structured output
# -----------------------------
if structured_output:
    print("\n=== Structured Startup Analysis ===")
    print(structured_output.model_dump_json(indent=2))