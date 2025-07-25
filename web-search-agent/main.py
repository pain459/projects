import os
import json
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import requests
import google.generativeai as genai

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# -------------------------
# 🧠 Step 1: Planner Agent
# -------------------------
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful planner agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


# -------------------------
# 🧾 Step 2: Plan Parser
# -------------------------
def parse_plan_output(plan_text: str) -> List[Dict[str, str]]:
    try:
        return json.loads(plan_text)
    except json.JSONDecodeError:
        steps = []
        lines = plan_text.strip().split('\n')
        for line in lines:
            if re.match(r"^\d+\.", line):
                step_num, action = line.split(".", 1)
                steps.append({"step": step_num.strip(), "action": action.strip()})
        return steps


# -------------------------
# 🔍 Step 3: Tavily Search
# -------------------------
def search_trending_startups() -> str:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "query": "top AI startups July 2025",
        "search_depth": "basic"
    }
    response = requests.post(url, json=payload)
    data = response.json()

    results = data.get("results", [])
    formatted = "\n".join(
        f"{i+1}. {r['title']}\nURL: {r['url']}\nSnippet: {r.get('content', '')}"
        for i, r in enumerate(results[:3])
    )
    return formatted or "No results found."


# -------------------------
# 📝 Step 4: GPT Summarizer
# -------------------------
def summarize_startups(raw_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a startup analyst who summarizes startup news for executive briefings."},
            {"role": "user", "content": f"Summarize the following startup news in 3 points:\n\n{raw_text}"}
        ],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()


# -------------------------
# 📄 Step 5: Markdown Writer
# -------------------------
def write_markdown(summary: str) -> str:
    filename = "ai_startup_report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Top AI Startups – July 2025\n\n")
        f.write(summary)
    return f"✅ Report written to `{filename}`"


# -------------------------
# 🧠 Step 6: Gemini Evaluator
# -------------------------
def evaluate_with_gemini(report: str, goal: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
You are a quality reviewer AI. Evaluate the following report based on this user goal:

Goal: {goal}

Report:
{report}

Return a short review: Is it relevant, accurate, clear, and complete? Rate it 1–5 and explain your reasoning.
"""
    response = model.generate_content(prompt)
    return response.text.strip()


# -------------------------
# 🔁 Step 7: Tool Registry
# -------------------------
def pick_tool(action_text: str):
    action_lower = action_text.lower()
    if any(kw in action_lower for kw in ["search", "find", "research", "identify"]):
        return "search"
    elif any(kw in action_lower for kw in ["summarize", "extract", "gather", "analyze", "review"]):
        return "summarize"
    elif any(kw in action_lower for kw in ["write", "report", "format", "compile"]):
        return "write"
    return None

TOOL_REGISTRY = {
    "search": search_trending_startups,
    "summarize": summarize_startups,
    "write": write_markdown
}


# -------------------------
# 🚀 Step 8: Agent Runner
# -------------------------
def run_agent(goal: str) -> str:
    plan_output = plan_task_with_llm(goal)
    parsed_steps = parse_plan_output(plan_output)

    intermediate_memory = None
    report_text = ""

    for step in parsed_steps:
        tool_key = pick_tool(step["action"])
        tool_fn = TOOL_REGISTRY.get(tool_key)

        if not tool_fn:
            continue

        if tool_key == "search":
            result = tool_fn()
        else:
            result = tool_fn(intermediate_memory)

        intermediate_memory = result
        report_text = result

    review = evaluate_with_gemini(report_text, goal)

    return f"{report_text}\n\n---\n🔍 Gemini Review:\n{review}"


# -------------------------
# 🌐 Step 9: Gradio UI
# -------------------------
gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(label="Goal", placeholder="e.g., Summarize top AI startups in July 2025"),
    outputs=gr.Textbox(label="Final Report with Gemini Review", lines=20),
    title="Agentic AI Research Assistant",
    description="Planner → Executor → Summarizer → Gemini Validator"
).launch()
