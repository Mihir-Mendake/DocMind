"""Chat with your documents: a minimal RAG app.

Flow: upload -> extract text -> split into chunks -> store in a vector DB
      question -> find the most similar chunks -> send them + question to the LLM
"""
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()  # reads GEMINI_API_KEY from the .env file

MODEL = "gemini-3.8-flash"  # free-tier model; if it errors, check the name in Google AI Studio
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # overlap so sentences aren't cut off between chunks
TOP_K = 4             # how many chunks to retrieve per question

st.set_page_config(page_title="Chat with your docs", page_icon="📄")
st.title("📄 Chat with your documents")

# ---------- Setup ----------
@st.cache_resource
def get_collection():
    db = chromadb.PersistentClient(path="chroma_db")  # saved on disk
    return db.get_or_create_collection("docs")        # uses a built-in local embedding model

collection = get_collection()
llm = genai.Client()  # reads GEMINI_API_KEY from your environment / .env


# ---------- Step 1: read and chunk ----------
def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest(file):
    """Extract text page by page, chunk it, and store it with metadata."""
    file.seek(0)
    if file.name.lower().endswith(".pdf"):
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(file).pages)]
    else:
        pages = [(1, file.read().decode("utf-8", errors="ignore"))]

    ids, docs, metas = [], [], []
    for page_no, text in pages:
        for j, chunk in enumerate(chunk_text(text)):
            if chunk.strip():
                ids.append(f"{file.name}-p{page_no}-c{j}")
                docs.append(chunk)
                metas.append({"source": file.name, "page": page_no})
    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)  # embeds automatically
    return len(docs)


# ---------- Step 2: retrieve + generate ----------
def answer(question):
    results = collection.query(query_texts=[question], n_results=TOP_K)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n\n".join(
        f"[{i+1}] ({m['source']}, page {m['page']})\n{c}"
        for i, (c, m) in enumerate(zip(chunks, metas))
    )
    system = (
        "You answer questions using ONLY the numbered sources provided. "
        "Cite sources like [1] or [2] after each claim. "
        "If the sources don't contain the answer, say you couldn't find it in the documents. "
        "Never make up information."
    )
    try:
        response = llm.models.generate_content(
            model=MODEL,
            contents=f"Sources:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(system_instruction=system),
        )
        text = response.text
    except Exception as e:  # e.g. free-tier rate limit (429)
        text = f"Sorry, the AI service returned an error: {e}\n\nIf it mentions a limit, wait a minute and try again."
    return text, list(zip(chunks, metas))


# ---------- UI ----------
with st.sidebar:
    st.header("Your documents")
    files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Process documents") and files:
        with st.spinner("Reading and indexing..."):
            results = [(f.name, ingest(f)) for f in files]
        total = sum(count for _, count in results)
        if total:
            st.success(f"Indexed {total} chunks")
        else:
            st.warning("No readable text was found. Check that the file contains selectable text.")
        empty_files = [name for name, count in results if count == 0]
        if empty_files:
            st.info(
                f"No text could be extracted from: {', '.join(empty_files)}. "
                "Scanned or image-only PDFs need OCR, which this app does not currently include."
            )
    st.caption(f"Chunks in database: {collection.count()}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if collection.count() == 0:
        reply, sources = "Please upload and process a document first.", []
    else:
        with st.spinner("Thinking..."):
            reply, sources = answer(question)

    with st.chat_message("assistant"):
        st.markdown(reply)
        if sources:
            with st.expander("Sources used"):
                for i, (chunk, m) in enumerate(sources):
                    st.markdown(f"**[{i+1}] {m['source']}, page {m['page']}**")
                    st.caption(chunk)
    st.session_state.messages.append({"role": "assistant", "content": reply})
