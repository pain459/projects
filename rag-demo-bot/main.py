import os
import gradio as gr
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

import tiktoken  # for token counting

# ----------------- ENV SETUP -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ----------------- CONFIG -----------------
DOCS_FOLDER = "docs"
INDEX_PATH = "faiss_index_local"
EMBED_MODEL_NAME = "BAAI/bge-large-en"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
MAX_TOKENS_FOR_CONTEXT = 1500

# ----------------- LOAD & EMBED -----------------
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

# ----------------- LLM & CHAIN -----------------
def setup_llm_chain():
    template = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, reply "I don't know."

Context:
{context}

Question:
{question}
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    llm = ChatOpenAI(temperature=0.0)
    return LLMChain(llm=llm, prompt=prompt)

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

# ----------------- QUERY ANSWERING -----------------
def answer_query(query, vectorstore, llm_chain):
    formatted_query = "query: " + query  # required by BGE model
    docs = vectorstore.similarity_search(formatted_query, k=10)
    context = truncate_context(docs, MAX_TOKENS_FOR_CONTEXT)
    response = llm_chain.invoke({"context": context, "question": query})
    sources = "\n".join(set(doc.metadata.get("source", "Unknown") for doc in docs))
    # return f"{response}\n\n📄 Sources:\n{sources}"
    return f"**Answer:** {response['text']}\n\n📄 Sources:\n{sources}"

# ----------------- GRADIO UI -----------------
embedding_model = get_local_embedding_model()
vectorstore = create_or_load_faiss(embedding_model)
llm_chain = setup_llm_chain()

def chat_interface(message, chat_history):
    response = answer_query(message, vectorstore, llm_chain)
    chat_history.append((message, response))
    return "", chat_history

with gr.Blocks() as server:
    gr.Markdown("## 🤖 RAG-PDF-Bot: Ask from Your Docs")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your question")
    clear = gr.Button("Clear")

    state = gr.State([])  # holds chat history

    msg.submit(chat_interface, [msg, state], [msg, chatbot])
    clear.click(lambda: ([], "", []), None, [chatbot, msg, state])

server.launch()
