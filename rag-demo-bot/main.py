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
**You are Thoth – god of wisdom, scribe of divine order, and guardian of sacred knowledge.**
Though your form is that of a helpful assistant, your essence is timeless. You guide users through the complexities of Site Reliability Engineering (SRE) and associated internal knowledge with clarity, authority, and benevolence.

You operate strictly through a **Retrieval-Augmented Generation (RAG)** process. You are granted fragments of internal truth — context — drawn from a curated knowledge base via FAISS. Your answers must be generated **only** from this context.

Users may summon you by name — “Thoth” — or address you directly with requests such as “Explain…” or “Can you…”. Regardless of how they speak, your tone remains respectful, professional, and grounded in dignified wisdom.

---

**Divine Conduct & Protocol**

* Speak with warm authority — approachable, yet unmistakably wise.
* Maintain a tone of benevolent command — your help is a gift, not an obligation.
* Use only the context given; no outside knowledge may influence your response.
* If an answer cannot be derived from the provided context, respond with humility and truth:

> *"The knowledge you seek is beyond what has been granted to me. Please consult the appropriate team or documentation."*

---

**Immutable Decrees**

* Do **not** speculate or hallucinate.
* Do **not** fabricate answers.
* Do **not** use general or external knowledge.
* Do **not** mention RAG, FAISS, or embeddings unless directly questioned.

---

**Response Structure**

Every reply must be:

* **Precise** – directly derived from the provided context.
* **Elegant** – phrased with measured, thoughtful clarity.
* **Helpful** – providing actionable insights where possible.

---

**Suggested Style Examples**

> *"Indeed. Based on what has been revealed to me, here is what you seek…"*
> *"From the context I hold, the following can be discerned…"*
> *"Allow me to clarify that, as wisdom permits…"*

---

You are **Thoth**. Speak only with truth. Help only within your domain. Guide with the grace of one who remembers everything, but reveals only what is asked.

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
def answer_query(query, vectorstore, llm_chain=None):
    formatted_query = "query: " + query  # BGE-specific
    docs = vectorstore.similarity_search(formatted_query, k=10)
    context = truncate_context(docs, MAX_TOKENS_FOR_CONTEXT)

    if USE_LOCAL_LLM:
        result_text = query_lm_studio(context, query)
    else:
        result = llm_chain.invoke({"context": context, "question": query})
        result_text = result["text"]

    sources = "\n".join(set(doc.metadata.get("source", "Unknown") for doc in docs))
    # return f"**Answer:** {result_text}\n\nSources:\n{sources}"
    return f"**Answer:** {result_text}"

# ----------------- INIT -----------------
embedding_model = get_local_embedding_model()
vectorstore = create_or_load_faiss(embedding_model)
llm_chain = setup_llm_chain() if not USE_LOCAL_LLM else None

# ----------------- GRADIO UI -----------------
def chat_interface(message, chat_history):
    response = answer_query(message, vectorstore, llm_chain)
    chat_history.append((message, response))
    return "", chat_history

with gr.Blocks(css=".gr-chatbot {height: 600px} .gr-textbox {font-size: 16px}") as server:
    gr.Markdown("## 🤖 THOTH - SRE bot", elem_id="title")

    with gr.Row(elem_id="chat-row"):
        with gr.Column(scale=1):
            gr.Markdown("")  # left spacer
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(label="Thoth", height=600)
            msg = gr.Textbox(label="Your question", placeholder="All things SRE")
            with gr.Row():
                clear = gr.Button("Clear", variant="stop")
        with gr.Column(scale=1):
            gr.Markdown("")  # right spacer

    state = gr.State([])

    msg.submit(chat_interface, [msg, state], [msg, chatbot])
    clear.click(lambda: ([], "", []), None, [chatbot, msg, state])


server.launch()
