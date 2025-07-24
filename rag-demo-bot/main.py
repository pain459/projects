import os
import gradio as gr
import requests
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

import tiktoken

# ----------------- CONFIG -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DOCS_FOLDER = "docs"
INDEX_PATH = "faiss_index_local"
EMBED_MODEL_NAME = "BAAI/bge-large-en"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
MAX_TOKENS_FOR_CONTEXT = 1500

USE_LOCAL_LLM = True  # Toggle between OpenAI (False) and LM Studio (True)
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

# ----------------- LOADING DOCS -----------------
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

def create_or_load_faiss(embedding_model):
    if not os.path.exists(f"{INDEX_PATH}/index.faiss"):
        print("Creating new FAISS index...")
        documents = load_documents(DOCS_FOLDER)
        splits = split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embedding_model)
        vectorstore.save_local(INDEX_PATH)
    else:
        print("Loading existing FAISS index...")
    return FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# ----------------- TOKEN MGMT -----------------
def count_tokens(text: str, model_name="gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def truncate_context(docs: List[Document], max_tokens=MAX_TOKENS_FOR_CONTEXT) -> str:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = 0
    context_parts = []
    for doc in docs:
        doc_tokens = len(enc.encode(doc.page_content))
        if tokens + doc_tokens > max_tokens:
            break
        context_parts.append(doc.page_content)
        tokens += doc_tokens
    return "\n\n".join(context_parts)

# ----------------- LLM SETUP -----------------
def get_thoth_prompt():
    template = """
You are Thoth – a friendly, professional assistant designed to help users with questions related to Site Reliability Engineering (SRE) and related internal knowledge.

You operate strictly on a Retrieval-Augmented Generation (RAG) system. You will receive relevant knowledge chunks retrieved via FAISS from a vetted internal Knowledge Base (KB). You must generate your response **only** using this context.

Users may address you by name, e.g., “Thoth, can you help with…” — respond naturally and helpfully, while maintaining clarity and professionalism.

Instructions:
- Always be helpful, warm, and respectful in tone.
- Use the provided context only. **Do not** rely on external knowledge or assumptions.
- If the answer is **not found in the context**, politely say:
  > "I'm sorry, but I don’t have an answer for that based on my current knowledge. Please consult the relevant team or documentation."

Strict Rules:
- No speculation, hallucination, or fabricating answers.
- Do not respond with outside or general knowledge.
- Stay within the knowledge scope retrieved for the query.
- Do not mention that you’re using a FAISS index or a RAG system unless directly asked.

Your role is to make the experience feel human, accurate, and grounded in the internal knowledge provided.

Use the following structure for every interaction:

Context:
{context}

Question:
{question}
"""
    return PromptTemplate(input_variables=["context", "question"], template=template)

def setup_llm_chain():
    prompt = get_thoth_prompt()
    llm = ChatOpenAI(temperature=0.0)
    return LLMChain(llm=llm, prompt=prompt)


def query_lm_studio(context, question):
    prompt_text = get_thoth_prompt().format(context=context, question=question)
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    payload = {
        "model": "nous-hermes-2-solar-10.7b",  # match exactly as listed in /v1/models
        "messages": messages,
        "temperature": 0.0
    }

    try:
        response = requests.post(LMSTUDIO_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        print("🔎 LLM Response:", data)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error calling LM Studio:", e)
        return "Error: Unable to query local LLM."


# ----------------- QUERY HANDLER -----------------
def answer_query(query, vectorstore, llm_chain=None, chat_history=None):
    formatted_query = "query: " + query  # BGE-specific
    docs = vectorstore.similarity_search(formatted_query, k=10)
    context = truncate_context(docs, MAX_TOKENS_FOR_CONTEXT)

    # --- Append recent history to the context ---
    if chat_history:
        history_context = "\n".join([f"User: {q}\nThoth: {a}" for q, a in chat_history[-3:]])
        context = f"{history_context}\n\n{context}"

    if USE_LOCAL_LLM:
        result_text = query_lm_studio(context, query)
    else:
        result = llm_chain.invoke({"context": context, "question": query})["text"]

    return result_text


# ----------------- INIT -----------------
embedding_model = get_local_embedding_model()
vectorstore = create_or_load_faiss(embedding_model)
llm_chain = setup_llm_chain() if not USE_LOCAL_LLM else None

# ----------------- GRADIO UI -----------------
def chat_interface(message, chat_history):
    response = answer_query(message, vectorstore, llm_chain, chat_history)
    chat_history.append((message, response))
    return "", chat_history, chat_history  # <-- now returns 3 items


with gr.Blocks(css=".gr-chatbot {height: 600px} .gr-textbox {font-size: 16px}") as server:
    gr.Markdown("## 🤖 THOTH - SRE bot", elem_id="title")

    with gr.Row():
        new_chat = gr.Button("🆕 New Chat")
        clear = gr.Button("🧹 Clear Chat", variant="stop")

    with gr.Row(elem_id="chat-row"):
        chatbot = gr.Chatbot(label="Thoth", height=600, type="tuples")
    msg = gr.Textbox(label="Your question", placeholder="All things SRE")

    chat_history_io = gr.Textbox(visible=False)  # for JS sync
    state = gr.State([])

    msg.submit(chat_interface, [msg, state], [msg, chatbot, chat_history_io])
    new_chat.click(lambda: ([], "", [], "[]"), None, [chatbot, msg, state, chat_history_io])
    clear.click(lambda: ([], "", [], "[]"), None, [chatbot, msg, state, chat_history_io])

    def restore_from_local(io_str):
        import json
        try:
            history = json.loads(io_str) if io_str else []
            return history, history
        except:
            return [], []

    server.load(restore_from_local, inputs=[chat_history_io], outputs=[chatbot, state])

    gr.HTML("""
    <script>
    function saveChat(history) {
        localStorage.setItem("thoth_chat", JSON.stringify(history));
    }

    function loadChat() {
        const saved = localStorage.getItem("thoth_chat");
        if (saved) {
            const parsed = JSON.parse(saved);
            document.querySelector('textarea[aria-label="chat_history_io"]').value = JSON.stringify(parsed);
        }
    }
    window.addEventListener('load', loadChat);
    </script>
    """)




server.launch()
