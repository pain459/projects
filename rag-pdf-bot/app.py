import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

import faiss
import numpy as np

from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS

import gradio as gr

# Load API key from .env
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("❌ OPENAI_API_KEY is missing in the .env file.")

# Global cache
VECTORSTORE = None
CHUNKS = []
PDF_NAME = None

# --- PDF & Chunking ---
def extract_pdf_text(file_path: str) -> str:
    return extract_text(file_path)

def split_text_to_chunks(text: str, chunk_size=1000, chunk_overlap=200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# --- Embedding ---
def create_faiss_index(chunks: list, model_name: str = "text-embedding-3-small") -> FAISS:
    embedding_model = OpenAIEmbeddings(model=model_name)
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    return vectorstore

# --- Retrieval & Prompt ---
def retrieve_relevant_chunks(vectorstore, query: str, k: int = 4) -> list:
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def build_rag_prompt(query: str, context_chunks: list) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""You are an expert assistant. Use the following context from a document to answer the user's question. If unsure, say so.

Context:
{context_text}

Question:
{query}

Answer:"""

# --- LLM ---
def get_llm_response(prompt: str, model_name: str = "gpt-4o", temperature: float = 0.2) -> str:
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    response = llm([HumanMessage(content=prompt)])
    return response.content

# --- Gradio UI ---
def process_pdf(file_obj, embedding_model: str = "text-embedding-3-small"):
    global VECTORSTORE, CHUNKS, PDF_NAME

    if not file_obj:
        return "❗ No file provided."

    file_path = file_obj.name
    PDF_NAME = Path(file_path).stem

    try:
        text = extract_pdf_text(file_path)
        CHUNKS = split_text_to_chunks(text)
        VECTORSTORE = create_faiss_index(CHUNKS, model_name=embedding_model)
        return f"✅ Processed {len(CHUNKS)} chunks from: {PDF_NAME}"
    except Exception as e:
        return f"❌ Failed to process PDF: {str(e)}"

def handle_question(question: str, model: str = "gpt-4o"):
    if VECTORSTORE is None:
        return "❗ Please upload and process a PDF first."

    relevant = retrieve_relevant_chunks(VECTORSTORE, question, k=4)
    prompt = build_rag_prompt(question, relevant)
    answer = get_llm_response(prompt, model_name=model)
    return answer

# UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 RAG-based PDF QA Bot (OpenAI + FAISS)")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        embedding_model = gr.Textbox(label="Embedding Model", value="text-embedding-3-small")
        process_btn = gr.Button("📚 Process PDF")

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question = gr.Textbox(label="Ask a question")
        model_choice = gr.Dropdown(choices=["gpt-4o", "gpt-4", "gpt-3.5-turbo"], value="gpt-4o", label="LLM Model")
        ask_btn = gr.Button("🔍 Get Answer")

    answer_output = gr.Textbox(label="Answer", lines=8)

    process_btn.click(process_pdf, inputs=[pdf_input, embedding_model], outputs=status)
    ask_btn.click(handle_question, inputs=[question, model_choice], outputs=answer_output)

# Run the app
if __name__ == "__main__":
    demo.launch()
