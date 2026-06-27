# 📄 Chat with PDF — RAG Application

An AI-powered application that lets you upload any PDF and ask questions about it using **Retrieval Augmented Generation (RAG)** with LangChain and Groq LLM.

## 🚀 Live Demo
[Click here to try it](YOUR_STREAMLIT_LINK_HERE)

## 📸 Screenshot
![Chat with PDF App](screenshot.png)

---

## 🧠 How It Works

This application uses a RAG (Retrieval Augmented Generation) pipeline:

```
PDF Upload
→ Split into chunks (RecursiveCharacterTextSplitter)
→ Convert to embeddings (sentence-transformers)
→ Store in vector database (ChromaDB)
→ User asks question
→ Retrieve top 3 similar chunks (cosine similarity)
→ Pass chunks + question to LLM (Groq)
→ Generate grounded answer
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Framework | Streamlit |
| RAG Framework | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | Groq (LLaMA 3.1 8B) |
| PDF Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |

---

## ✨ Features

- Upload any PDF — resume, research paper, notes, report
- Ask natural language questions about the document
- Answers grounded in actual document content — no hallucination
- View retrieved chunks — see exactly what RAG found to answer your question
- Conversation history — remembers previous questions in session
- Free to use — powered by Groq's free API

---

## 🏃 Run Locally

**Clone the repo:**
```bash
git clone https://github.com/Shaheen3107/chat-with-pdf.git
cd chat-with-pdf
```

**Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the app:**
```bash
python -m streamlit run app.py
```

**Get your free Groq API key:**
- Go to [console.groq.com](https://console.groq.com)
- Sign up free
- Copy your API key
- Paste in the app sidebar

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-chroma
langchain-groq
chromadb
sentence-transformers
pypdf
```

---

## 📁 Project Structure

```
chat-with-pdf/
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
├── README.md          # Project documentation
└── screenshot.png     # App screenshot
```

---

## 💡 Key Concepts Demonstrated

- **RAG Architecture** — two-stage pipeline (ingestion + retrieval/generation)
- **Document Loading** — PDF text extraction using PyPDFLoader
- **Chunking** — splitting documents into overlapping chunks for precise retrieval
- **Semantic Embeddings** — converting text to dense vectors using sentence-transformers
- **Vector Database** — storing and searching embeddings with ChromaDB
- **Retrieval** — cosine similarity search to find relevant chunks
- **Generation** — grounded answer generation using Groq LLM
- **Hallucination Prevention** — answers only from retrieved document context

---

## 👤 Author

**Shaheen Akram**
- GitHub: [@Shaheen3107](https://github.com/Shaheen3107)

---

## 📝 Related Projects

- [ATS Resume Checker](https://github.com/Shaheen3107/ATS-Resume-Checker) — NLP + BERT semantic matching
- [Fake News Detector](https://github.com/Shaheen3107/fake-news-detector) — ML + NLP classification