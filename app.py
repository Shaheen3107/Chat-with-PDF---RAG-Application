import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

# ── TITLE ──
st.title("📄 Chat with PDF")
st.markdown("Upload any PDF and ask questions — powered by RAG + Groq LLM")
st.divider()

# ── SIDEBAR — API KEY ──
with st.sidebar:
    st.subheader("⚙️ Settings")
    groq_api_key = st.text_input(
        "Enter Groq API Key",
        type="password",
        placeholder="gsk_..."
    )
    st.markdown("Get free API key at [console.groq.com](https://console.groq.com)")
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. App chunks and embeds it")
    st.markdown("3. Ask any question")
    st.markdown("4. RAG retrieves relevant chunks")
    st.markdown("5. Groq LLM generates answer")

# ── FILE UPLOAD ──
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    help="Upload any PDF — resume, research paper, notes, report"
)

# ── PROCESS PDF ──
if uploaded_file and groq_api_key:

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF — this may take a moment..."):

        # Step 1 — Load
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Step 2 — Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        # Step 3 — Embed
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Step 4 — Store in ChromaDB
        vectorstore = Chroma.from_documents(chunks, embeddings)

        # Step 5 — Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Step 6 — LLM
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.1-8b-instant"
        )

        # Step 7 — Prompt
        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

        # Step 8 — Chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    # Success message
    st.success(f"✅ PDF processed — {len(documents)} pages, {len(chunks)} chunks ready")
    st.divider()

    # ── CHAT INTERFACE ──
    st.subheader("💬 Ask a Question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if question := st.chat_input("Ask anything about your PDF..."):

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(question)
            st.markdown(answer)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── SHOW RETRIEVED CHUNKS ──
    with st.expander("🔍 View Retrieved Chunks (what RAG found)"):
        if st.session_state.messages:
            last_question = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else None
            if last_question:
                relevant_docs = retriever.invoke(last_question)
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

    # Clean up temp file
    os.unlink(tmp_path)

elif uploaded_file and not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue")

elif not uploaded_file and groq_api_key:
    st.info("📤 Please upload a PDF to get started")

else:
    st.info("👈 Enter your Groq API key in the sidebar, then upload a PDF")
