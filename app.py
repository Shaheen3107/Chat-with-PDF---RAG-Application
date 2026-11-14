import os
import tempfile

import streamlit as st

from rag_core import ask, build_answer_chain, build_retriever, source_summary

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="centered")

st.title("📄 Chat with PDF")
st.markdown("Upload any PDF and ask questions — hybrid RAG (semantic + keyword search) with page citations")
st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ Settings")
    groq_api_key = st.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("Get free API key at [console.groq.com](https://console.groq.com)")
    search_mode = st.radio("Retrieval mode", ["hybrid", "semantic", "keyword"])
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. App chunks it (each chunk keeps its page number)")
    st.markdown("3. Ask any question")
    st.markdown("4. Semantic + keyword search find the best chunks")
    st.markdown("5. Groq LLM answers and cites the pages")

# ---------- Upload ----------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    help="Upload any PDF — resume, research paper, notes, report",
)

if uploaded_file and groq_api_key:

    # ---------- Build the pipeline (only when a NEW file is uploaded) ----------
    # Streamlit re-runs this whole script on every click, so we save the
    # retriever in st.session_state and rebuild it only when the file changes.
    file_key = (uploaded_file.name, uploaded_file.size)

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Processing PDF — this may take a moment..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            retriever, n_pages, n_chunks = build_retriever(tmp_path, uploaded_file.name)
            os.unlink(tmp_path)

        st.session_state.file_key = file_key
        st.session_state.retriever = retriever
        st.session_state.n_pages = n_pages
        st.session_state.n_chunks = n_chunks
        st.session_state.messages = []

    st.success(f"✅ PDF processed — {st.session_state.n_pages} pages, {st.session_state.n_chunks} chunks ready")
    st.divider()

    answer_chain = build_answer_chain(groq_api_key)

    # ---------- Chat ----------
    st.subheader("💬 Ask a Question")

    # Show earlier messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                pages_used = ", ".join(str(p) for p in message["sources"]["pages"])
                with st.expander(f"🔍 Sources — page(s) {pages_used}"):
                    for i, doc in enumerate(message["sources"]["docs"], start=1):
                        st.markdown(f"**Chunk {i} — page {doc['page']}**")
                        st.text(doc["text"])

    # New question
    if question := st.chat_input("Ask anything about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = ask(st.session_state.retriever, answer_chain, question, mode=search_mode)
            st.markdown(answer)

            sources = {
                "pages": source_summary(docs),
                "docs": [{"page": d.metadata["page"], "text": d.page_content} for d in docs],
            }
            pages_used = ", ".join(str(p) for p in sources["pages"])
            with st.expander(f"🔍 Sources — page(s) {pages_used}"):
                for i, doc in enumerate(sources["docs"], start=1):
                    st.markdown(f"**Chunk {i} — page {doc['page']}**")
                    st.text(doc["text"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

elif uploaded_file and not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue")
elif not uploaded_file and groq_api_key:
    st.info("📤 Please upload a PDF to get started")
else:
    st.info("👈 Enter your Groq API key in the sidebar, then upload a PDF")
