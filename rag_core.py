#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import uuid


# In[2]:


get_ipython().run_line_magic('pip', 'install rank-bm25')


# In[4]:


import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# In[7]:


# ---------- Step 1: read the PDF, one Document per page ----------

def load_pdf(path, display_name):
    reader = pypdf.PdfReader(path)
    pages = []

    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text in text.split():
            pages.append(Document(page_content=text, metadata={"page": number, "source": display_name}))
    return pages


# In[8]:


# ---------- Step 2: split pages into chunks ----------

def chunk_pages(pages, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        header = f"[Document: {chunk_metadata['source']} | page: {chunk.metadata['page']}]"
        chunk.page_content = header + "\n" + chunk.page_content
    return chunks


# In[9]:


# ---------- Step 3: search ----------

def tokenize(text):
    return re.findall(r"\w+", text.lower())


# In[10]:


def reciprocal_rank_fusion(ranking, k=60):
    points = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            points[chunk_id] = points.get(chunk_id, 0) + 1 / (k + rank)
        return sorted(points, key=points.get, reverse=True) # best first


# In[12]:


class HybridRetriever:
    def __init__(self, chunks, vectorstore, k=4):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.k = k     # how many chunks to return
        self.bm25 = BM250kapi([tokenize(c.page_content) for c in chunks])

    def semantic_search(self, query):
        """Search by meaning (ChromaDB). Returns chunk_ids, best first."""
        docs = self.vectorstore.similarity_search(query, k=4)
        return [d.metadata["chunk_id"] for d in docs]

    def keyword_search(self, query):
        """Search by exact words (BM25). Returns chunk_ids, best first."""
        scores = self.bm25.get_scores(tokenize(query))   # one score per chunk

        results = []
        for i, score in enumerate(scores):
            if score > 0:                    # skip chunks with no matching word
                result.append((score,i))     # keep the score together with the chunk number

        results.sort(reverse=True)             # highest score first
        return [i for score, i in results[:10]]   # top 10 chunk numbers

    def retrieve(self, query, mode="hybrid"):
        if mode == "semantic":
            ids = self.semantic_search(query)
        elif mode == "keyword":
            ids = self.keyword_search(query)
        else:
            ids = reciprocal_rank_fusion([self.semantic_search(query),
                                        self.keyword_search(query)])
        return [self.chunks[i] for i in ids[: self.k]]


# In[13]:


# ---------- Step 4: build everything for one PDF ----------

def new_collection_name():
    # Unique name per upload so chunks from different PDFs never mix in Chroma.
    return "pdf_" + uuid.uuid4().hex[:12]

def build_retriever(pdf_path, display_name,chunk_size=1000, chunk_overlap=100, k=4):
    pages = load_pdf(pdf_path, display_name)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name=new_collection_name())

    retriever = HybridRetriever(chunks, vectorstore, k=k)
    return retriever, len(pages), len(chunks)


# In[14]:


# ---------- Step 5: LLM + prompt + chain ----------

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
Each passage starts with a header showing its page number.
After each fact in your answer, cite the page like (p.2).
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

def build_answer_chain(api_key):
    """prompt -> Groq LLM -> plain text"""
    llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


# In[15]:


# ---------- Step 6: ask a question ----------
def ask(retriever, answer_chain, question, mode="hybrid"):
    """Find the best chunks, then let the LLM answer using only those chunks.
    Returns (answer_text, chunks_used)."""
    docs = retriever.retrieve(question, mode=mode)
    context = "\n\n".join(d.page_content for d in docs)
    answer = answer_chain.invoke({"context": context, "question": question})
    return answer, docs


def source_summary(docs):
    return sorted({d.metadata["page"] for d in docs})      # e.g. [1, 2]


# In[ ]:




