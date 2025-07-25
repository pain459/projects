# ================================
# 🧠 STEP 1: Imports and Setup
# ================================
import os
import json
import re
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================
# 🎯 STEP 2: User Goal
# ================================
user_goal = "Create a report summarizing the top 3 most talked-about AI startups in July 2025"

# ================================
# 🗂️ STEP 3: Planner Agent (LLM)
# ================================
def plan_task_with_llm(goal: str) -> str:
    prompt = f"""
You are a task planner agent. Break the following high-level goal into 3–5 clear, ordered subtasks.

Goal: {goal}

Format:
[
  {{ "step": "1", "action": "Describe the task to perform" }},
  ...
]
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful planner agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

plan_output = plan_task_with_llm(user_goal)
print("📋 Plan Output:\n", plan_output)

# ================================
# 🧾 STEP 4: Parse Plan Output
# ================================
def parse_plan_output(plan_text: str) -> List[Dict[str, str]]:
    try:
        return json.loads(plan_text)
    except json.JSONDecodeError:
        steps = []
        lines = plan_text.strip().split('\n')
        for line in lines:
            if re.match(r"^\d+\.", line):
                step_num, action = line.split(".", 1)
                steps.append({
                    "step": step_num.strip(),
                    "action": action.strip()
                })
        return steps

parsed_steps = parse_plan_output(plan_output)
print("✅ Parsed Steps:\n", parsed_steps)

# ================================
# 🛠️ STEP 5: Tools + Registry
# ================================
# def search_trending_startups() -> str:
#     # Static simulated result for now
#     return """
#     1. SynthMind – building low-latency multi-modal AI chips.
#     2. QuantaFlow – offers self-healing infrastructure for LLMs.
#     3. NeuralForge – language agents that build software end-to-end.
#     """
import requests
from bs4 import BeautifulSoup

# def search_trending_startups() -> str:
#     url = "https://techcrunch.com/latest/"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     articles = soup.select("h2.post-block__title a")[:3]
#     results = "\n".join(f"{i+1}. {a.text.strip()}" for i, a in enumerate(articles))
#     return results or "No results found."

import requests

def search_trending_startups() -> str:
    # Example using Tavily
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": "top AI startups July 2025",
        "search_depth": "basic"
    }
    response = requests.post(url, json=payload)
    data = response.json()
    
    results = data.get("results", [])
    return "\n".join(f"{i+1}. {r['title']}: {r['url']}" for i, r in enumerate(results[:3])) or "No results."


def summarize_startups(raw_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a startup analyst who summarizes startup news for executive briefings."},
            {"role": "user", "content": f"Summarize the following startup news in 3 points:\n\n{raw_text}"}
        ],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()

def write_markdown(summary: str) -> str:
    filename = "ai_startup_report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Top AI Startups – July 2025\n\n")
        f.write(summary)
    return f"✅ Report written to `{filename}`"


TOOL_REGISTRY = {
    "search": search_trending_startups,
    "summarize": summarize_startups,
    "write": write_markdown
}

# ================================
# 🔁 STEP 6: Executor Loop
# ================================
def pick_tool(action_text: str):
    action_lower = action_text.lower()
    if any(kw in action_lower for kw in ["search", "find", "research", "identify"]):
        return "search"
    elif any(kw in action_lower for kw in ["summarize", "extract", "gather", "analyze"]):
        return "summarize"
    elif any(kw in action_lower for kw in ["write", "report", "format", "compile"]):
        return "write"
    elif any(kw in action_lower for kw in ["review", "edit", "finalize"]):
        return "summarize"  # Use summarizer again or create a "review" tool if needed
    return None


intermediate_memory = None

for step in parsed_steps:
    step_id = step["step"]
    action = step["action"]
    print(f"\n🧭 Step {step_id}: {action}")

    tool_key = pick_tool(action)
    if not tool_key:
        print("❌ No matching tool found.")
        continue

    tool_fn = TOOL_REGISTRY.get(tool_key)
    if not tool_fn:
        print("⚠️ Tool function not implemented.")
        continue

    # Call with or without memory
    if tool_key in ["search"]:
        result = tool_fn()
    else:
        result = tool_fn(intermediate_memory)

    intermediate_memory = result
    print("✅ Output:\n", result)
