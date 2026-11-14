{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4c33b456-5af6-45a8-8846-1c6a66602a9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import uuid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7c34f53d-0bc2-40a3-9e6a-59b0a9a980d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: rank-bm25 in c:\\users\\shahe\\anaconda3\\lib\\site-packages (0.2.2)\n",
      "Requirement already satisfied: numpy in c:\\users\\shahe\\anaconda3\\lib\\site-packages (from rank-bm25) (2.3.5)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "%pip install rank-bm25"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "af936509-a420-4cc6-b080-d6091281b39f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pypdf\n",
    "from langchain_chroma import Chroma\n",
    "from langchain_core.documents import Document\n",
    "from langchain_core.output_parsers import StrOutputParser\n",
    "from langchain_groq import ChatGroq\n",
    "from langchain_huggingface import HuggingFaceEmbeddings\n",
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "from rank_bm25 import BM25Okapi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2fa34e68-308f-43d1-9619-47a509dd716a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---------- Step 1: read the PDF, one Document per page ----------\n",
    "\n",
    "def load_pdf(path, display_name):\n",
    "    reader = pypdf.PdfReader(path)\n",
    "    pages = []\n",
    "\n",
    "    for number, page in enumerate(reader.pages, start=1):\n",
    "        text = page.extract_text()\n",
    "        if text in text.split():\n",
    "            pages.append(Document(page_content=text, metadata={\"page\": number, \"source\": display_name}))\n",
    "    return pages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "db3821c0-4275-4530-b132-fec931e7ef3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---------- Step 2: split pages into chunks ----------\n",
    "\n",
    "def chunk_pages(pages, chunk_size=1000, chunk_overlap=100):\n",
    "    splitter = RecursiveCharacterTextSplitter(\n",
    "        chunk_size=chunk_size, \n",
    "        chunk_overlap=chunk_overlap\n",
    "    )\n",
    "    chunks = splitter.split_documents(pages)\n",
    "\n",
    "    for i, chunk in enumerate(chunks):\n",
    "        chunk.metadata[\"chunk_id\"] = i\n",
    "        header = f\"[Document: {chunk_metadata[\"source\"]} | page: {chunk.metadata[\"page\"]}]\"\n",
    "        chunk.page_content = header + \"\\n\" + chunk.page_content\n",
    "    return chunks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "38440537-432f-432f-91a4-c51af2f7b709",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---------- Step 3: search ----------\n",
    "\n",
    "def tokenize(text):\n",
    "    return re.findall(r\"\\w+\", text.lower())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6ea33f43-b069-4723-b28b-513bf42f536d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def reciprocal_rank_fusion(ranking, k=60):\n",
    "    points = {}\n",
    "    for ranking in rankings:\n",
    "        for rank, chunk_id in enumerate(ranking, start=1):\n",
    "            points[chunk_id] = points.get(chunk_id, 0) + 1 / (k + rank)\n",
    "        return sorted(points, key=points.get, reverse=True) # best first"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c3901c4c-3bd4-4560-a6bb-83bb162f173f",
   "metadata": {},
   "outputs": [],
   "source": [
    "class HybridRetriever:\n",
    "    def __init__(self, chunks, vectorstore, k=4):\n",
    "        self.chunks = chunks\n",
    "        self.vectorstore = vectorstore\n",
    "        self.k = k     # how many chunks to return\n",
    "        self.bm25 = BM250kapi([tokenize(c.page_content) for c in chunks])\n",
    "\n",
    "    def semantic_search(self, query):\n",
    "        \"\"\"Search by meaning (ChromaDB). Returns chunk_ids, best first.\"\"\"\n",
    "        docs = self.vectorstore.similarity_search(query, k=4)\n",
    "        return [d.metadata[\"chunk_id\"] for d in docs]\n",
    "\n",
    "    def keyword_search(self, query):\n",
    "        \"\"\"Search by exact words (BM25). Returns chunk_ids, best first.\"\"\"\n",
    "        scores = self.bm25.get_scores(tokenize(query))   # one score per chunk\n",
    "\n",
    "        results = []\n",
    "        for i, score in enumerate(scores):\n",
    "            if score > 0:                    # skip chunks with no matching word\n",
    "                result.append((score,i))     # keep the score together with the chunk number\n",
    "\n",
    "        results.sort(reverse=True)             # highest score first\n",
    "        return [i for score, i in results[:10]]   # top 10 chunk numbers\n",
    "\n",
    "    def retrieve(self, query, mode=\"hybrid\"):\n",
    "        if mode == \"semantic\":\n",
    "            ids = self.semantic_search(query)\n",
    "        elif mode == \"keyword\":\n",
    "            ids = self.keyword_search(query)\n",
    "        else:\n",
    "            ids = reciprocal_rank_fusion([self.semantic_search(query),\n",
    "                                        self.keyword_search(query)])\n",
    "        return [self.chunks[i] for i in ids[: self.k]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7e2c871c-affd-4dd1-ae7c-3b8dc6f7f984",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---------- Step 4: build everything for one PDF ----------\n",
    "\n",
    "def new_collection_name():\n",
    "    # Unique name per upload so chunks from different PDFs never mix in Chroma.\n",
    "    return \"pdf_\" + uuid.uuid4().hex[:12]\n",
    "\n",
    "def build_retriever(pdf_path, display_name,chunk_size=1000, chunk_overlap=100, k=4):\n",
    "    pages = load_pdf(pdf_path, display_name)\n",
    "    chunks = chunk_pages(pages, chunk_size, chunk_overlap)\n",
    "\n",
    "    embeddings = HuggingFaceEmbeddings(model_name=\"all-MiniLM-L6-v2\")\n",
    "    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name=new_collection_name())\n",
    "\n",
    "    retriever = HybridRetriever(chunks, vectorstore, k=k)\n",
    "    return retriever, len(pages), len(chunks)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "88f90a61-3bc3-47ee-a71d-ed38aa17607d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---------- Step 5: LLM + prompt + chain ----------\n",
    "\n",
    "PROMPT_TEMPLATE = \"\"\"Answer the question using ONLY the context below.\n",
    "Each passage starts with a header showing its page number.\n",
    "After each fact in your answer, cite the page like (p.2).\n",
    "If the answer is not in the context, say \"I don't know based on the provided document.\"\n",
    "\n",
    "Context:\n",
    "{context}\n",
    "\n",
    "Question:\n",
    "{question}\n",
    "\n",
    "Answer:\n",
    "\"\"\"\n",
    "\n",
    "def build_answer_chain(api_key):\n",
    "    \"\"\"prompt -> Groq LLM -> plain text\"\"\"\n",
    "    llm = ChatGroq(api_key=api_key, model_name=\"llama-3.1-8b-instant\")\n",
    "    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)\n",
    "    return prompt | llm | StrOutputParser()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "82750d3e-0fb1-4271-acb4-d80b56284b3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ---------- Step 6: ask a question ----------\n",
    "def ask(retriever, answer_chain, question, mode=\"hybrid\"):\n",
    "    \"\"\"Find the best chunks, then let the LLM answer using only those chunks.\n",
    "    Returns (answer_text, chunks_used).\"\"\"\n",
    "    docs = retriever.retrieve(question, mode=mode)\n",
    "    context = \"\\n\\n\".join(d.page_content for d in docs)\n",
    "    answer = answer_chain.invoke({\"context\": context, \"question\": question})\n",
    "    return answer, docs\n",
    "\n",
    "\n",
    "def source_summary(docs):\n",
    "    return sorted({d.metadata[\"page\"] for d in docs})      # e.g. [1, 2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ac58e93-fcbe-414e-91e4-81b54f235456",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
