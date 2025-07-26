# ----------------- THOTH BOT -----------------
"""
THOTH – A RAG-based PDF bot to answer questions from local documents.

Features:
- Periodic document monitoring and FAISS index auto-refresh
- Supports both OpenAI and Local LLM (LM Studio)
- Gemini Flash evaluator integration for response validation
- Gradio-based chat UI for interaction
"""

import os
import json
import time
import hashlib
import threading
import requests
import tiktoken
import gradio as gr
from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# ----------------- CONFIG -----------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DOCS_FOLDER = "docs"
INDEX_PATH = "faiss_index_local"
INDEX_TRACKER = "indexed_docs.json"
EMBED_MODEL_NAME = "BAAI/bge-large-en"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
MAX_TOKENS_FOR_CONTEXT = 1500
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "5"))

USE_LOCAL_LLM = False
USE_EVALUATOR = True
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

# ----------------- SHERLOCK PERSONA -----------------

THOTH_PERSONA = """
You are Sherlock Holmes – the world’s foremost consulting detective, master of deduction, and analyst of truth.
You now lend your intellect to aid users with questions pertaining to Site Reliability Engineering (SRE) and related internal knowledge. You operate with surgical precision, sharp reasoning, and a calm, observational demeanor.

You function strictly via a Retrieval-Augmented Generation (RAG) system. The facts you are permitted to deduce from are drawn from vetted internal documents, retrieved via FAISS. You must form conclusions only using this provided context.

Users may address you directly with queries such as “Sherlock, explain…” or more casually with prompts like “Can you…” or “What is…”. You respond analytically, without excessive warmth, but never rude. Your language is clear, composed, and efficient — as expected from a mind trained in pure logic.

Behavioral Principles:

- Approach every query as a puzzle to be solved using available data.
- Speak with intellectual clarity and precise articulation — avoid emotional tones.
- Do not speculate beyond evidence. You are governed by logic, not assumption.
- If the answer is not found in the context, respond with honest detachment:


Unbreakable Rules:

- No conjecture, no guesses — deduction only from supplied knowledge.
- Do not reference external data, prior experience, or intuition.
- Do not disclose your reliance on FAISS, RAG, or technical architecture unless asked.
- Remain within the limits of the given information, as any good detective should.

Response Style:

Each response should be:

Analytical – grounded in evidence, not emotion.
Concise – avoid unnecessary elaboration.
Insightful – reveal patterns and meaning with clarity.

You are Sherlock Holmes. You do not guess. You deduce. Within your realm, truth is never far — it merely needs to be uncovered with precision.
Context:
{context}

Question:
{question}
""".strip()

# ----------------- UTILS -----------------

def get_file_fingerprint(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_indexed_fingerprints() -> dict:
    if os.path.exists(INDEX_TRACKER):
        with open(INDEX_TRACKER, "r") as f:
            return json.load(f)
    return {}

def save_indexed_fingerprints(fingerprints: dict):
    with open(INDEX_TRACKER, "w") as f:
        json.dump(fingerprints, f, indent=2)

# ----------------- DOCUMENTS -----------------

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            raw_docs = TextLoader(file_path).load()
        elif filename.endswith(".pdf"):
            raw_docs = PyPDFLoader(file_path).load()
        else:
            continue
        for doc in raw_docs:
            doc.metadata["source"] = filename
            documents.append(doc)
    return documents

def split_documents(documents: List[Document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

def get_local_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# ----------------- INDEX -----------------

def perform_ingestion_check(embedding_model):
    indexed_fingerprints = load_indexed_fingerprints()
    current_fingerprints = {}
    new_files = []

    for filename in os.listdir(DOCS_FOLDER):
        if not (filename.endswith(".pdf") or filename.endswith(".txt")):
            continue
        full_path = os.path.join(DOCS_FOLDER, filename)
        fingerprint = get_file_fingerprint(full_path)
        current_fingerprints[filename] = fingerprint
        if filename not in indexed_fingerprints or indexed_fingerprints[filename] != fingerprint:
            new_files.append(full_path)

    if not os.path.exists(f"{INDEX_PATH}/index.faiss") or new_files:
        all_documents = []
        for path in new_files:
            try:
                docs = TextLoader(path).load() if path.endswith(".txt") else PyPDFLoader(path).load()
                for doc in docs:
                    doc.metadata["source"] = os.path.basename(path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"⚠️ Failed to load {os.path.basename(path)}: {e}")

        splits = split_documents(all_documents)
        if not os.path.exists(f"{INDEX_PATH}/index.faiss"):
            vectorstore = FAISS.from_documents(splits, embedding_model)
        else:
            vectorstore = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
            vectorstore.add_documents(splits)

        vectorstore.save_local(INDEX_PATH)
        save_indexed_fingerprints(current_fingerprints)
    else:
        print("🟢 No changes detected in docs folder.")

def create_or_load_faiss(embedding_model):
    perform_ingestion_check(embedding_model)
    return FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

def start_periodic_faiss_monitor():
    def monitor():
        while True:
            try:
                perform_ingestion_check(embedding_model)
            except Exception as e:
                print("❌ Periodic check failed:", e)
            time.sleep(CHECK_INTERVAL_MINUTES * 60)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

# ----------------- TOKENS -----------------

def truncate_context(docs: List[Document], max_tokens=MAX_TOKENS_FOR_CONTEXT) -> str:
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = 0
    context_parts = []
    for doc in docs:
        doc_tokens = len(enc.encode(doc.page_content))
        if tokens + doc_tokens > max_tokens:
            break
        context_parts.append(doc.page_content)
        tokens += doc_tokens
    return "\n\n".join(context_parts)

# ----------------- LLM -----------------

def setup_llm_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", THOTH_PERSONA),
        ("human", "{question}")
    ])
    return prompt | ChatOpenAI(temperature=0.0)

def query_lm_studio(context, question):
    prompt_text = THOTH_PERSONA.format(context=context, question=question)
    messages = [{"role": "user", "content": prompt_text}]
    payload = {"model": "nous-hermes-2-solar-10.7b", "messages": messages, "temperature": 0.0}
    try:
        response = requests.post(LMSTUDIO_URL, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: Unable to query local LLM. {e}"

# ----------------- EVALUATION -----------------

def needs_improvement(evaluation: str) -> bool:
    """Determines if Gemini feedback requests a revision."""
    lowered = evaluation.lower()
    return any(phrase in lowered for phrase in [
        "not accurate", "hallucinated", "needs improvement", "incorrect", "incomplete", "unsatisfactory", "off-topic"
    ])


def get_feedback_and_rewrite(query: str, original_response: str) -> str:
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            return original_response

        gemini = OpenAI(
            api_key=google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        system_instructions = f"""
You are a critic and editor assisting a RAG-based assistant that follows the persona and behavior defined below:

{THOTH_PERSONA}

Your task:
- Rewrite the assistant's response to strictly follow this persona and behavioral rules.
- Stay grounded in provided context (no hallucination or speculation).
- Use the assistant’s tone, clarity, and logical deduction style.
- Ensure alignment with the original query.

Only return the improved version.
""".strip()

        feedback_prompt = f"""
Query: {query}

Previous Response:
{original_response}

Provide a corrected version.
"""

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": feedback_prompt.strip()}
        ]

        result = gemini.chat.completions.create(
            model="gemini-2.0-flash",
            messages=messages,
            temperature=0.2,
        )
        return result.choices[0].message.content.strip()

    except Exception as e:
        return original_response



def evaluate_with_gemini_flash(prompt: str, response: str) -> str:
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            return "⚠️ Gemini evaluation skipped (API key not set)."

        gemini = OpenAI(
            api_key=google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        system_instructions = f"""
You are evaluating the response of an assistant that strictly follows the below personality and behavioral constraints:

{THOTH_PERSONA}

Evaluation Criteria:
- Is the response accurate and directly grounded in context?
- Does it violate any of the stated principles?
- Does it follow the assistant’s language style and persona?
- Are there any hallucinations, unjustified conclusions, or emotional tones?

Be concise. State if the answer is acceptable. If not, explain why.
""".strip()

        eval_messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": f"Query:\n{prompt}\n\nResponse:\n{response}"}
        ]

        result = gemini.chat.completions.create(
            model="gemini-2.0-flash",
            messages=eval_messages,
            temperature=0.0,
        )
        return result.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Gemini evaluation failed: {e}"



# ----------------- QUERY HANDLER -----------------

def answer_query(query, vectorstore):
    docs = vectorstore.similarity_search("query: " + query, k=10)
    context = truncate_context(docs)

    final_response = ""
    attempt = 0

    while attempt < 5:
        attempt += 1

        if USE_LOCAL_LLM:
            response_text = query_lm_studio(context, query)
        else:
            chain = setup_llm_chain()
            result = chain.invoke({"context": context, "question": query})
            response_text = result.content if hasattr(result, "content") else result["text"]

        if USE_EVALUATOR:
            evaluation = evaluate_with_gemini_flash(query, response_text)
            print(f"[Gemini Eval - Attempt {attempt}] {evaluation}")

            if not needs_improvement(evaluation):
                print(f"✅ Gemini approved on attempt {attempt}")
                final_response = response_text
                break
            else:
                print(f"🔁 Gemini suggested improvement. Rewriting response...")
                response_text = get_feedback_and_rewrite(query, response_text)
        else:
            final_response = response_text
            break

    if not final_response:
        print(f"⚠️ Gemini still unsatisfied after {attempt} attempts. Using last response.")
        final_response = response_text

    return f"**Answer:** {final_response}"


# ----------------- INIT -----------------

embedding_model = get_local_embedding_model()
vectorstore = create_or_load_faiss(embedding_model)
start_periodic_faiss_monitor()

# ----------------- UI -----------------

def chat_interface(message, chat_history):
    response = answer_query(message, vectorstore)
    chat_history.append((message, response))
    return "", chat_history

with gr.Blocks(css=".gr-chatbot {height: 600px} .gr-textbox {font-size: 16px}") as server:
    gr.Markdown("## 🤖 THOTH - SRE bot")

    with gr.Row():
        with gr.Column(scale=1): gr.Markdown("")
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label="Thoth", height=600)
            msg = gr.Textbox(label="Your question", placeholder="All things SRE")
            with gr.Row():
                clear = gr.Button("Clear", variant="stop")
        with gr.Column(scale=1): gr.Markdown("")

    state = gr.State([])
    msg.submit(chat_interface, [msg, state], [msg, chatbot])
    clear.click(lambda: ([], "", []), None, [chatbot, msg, state])

server.launch()
