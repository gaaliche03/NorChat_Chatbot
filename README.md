# NorChat: Intelligent University Chatbot

## Overview

NorChat is a conversational AI assistant developed as part of the **Master Sciences des Données** program at the **Université de Rouen Normandie**. It allows students and prospective applicants to ask questions about the university and the Master SD program through a natural language interface, instead of navigating multiple administrative pages.

The system relies exclusively on official curated documents as its knowledge base, ensuring that every response is grounded in verified information and free from hallucinations.

---

## Objectives

- Answer questions about the Master SD program, admission procedures, academic regulations, and student life
- Support both French and international students throughout administrative processes
- Deliver a conversational and accessible interface via a Streamlit application
- Demonstrate a complete RAG pipeline applied to a real academic use case

---

## Architecture

NorChat is built on a **Retrieval-Augmented Generation (RAG)** pipeline that combines semantic search with instruction-tuned language model generation.

### Pipeline Steps

**1. Document Loading**
Markdown files stored in `data/` are loaded using LangChain's `TextLoader`. Each file covers a specific domain: program overview, enrollment procedures, and academic regulations.

**2. Chunking**
Documents are split using a two-stage strategy:
- `MarkdownHeaderTextSplitter` preserves structural context from H1, H2, and H3 headers
- `RecursiveCharacterTextSplitter` further splits content into chunks of 500 tokens with a 50-token overlap

A cleaning step removes redundant whitespace and irrelevant characters before splitting.

**3. Embedding and Indexing**
Each chunk is embedded using `OrdalieTech/Solon-embeddings-large-0.1`, a sentence embedding model optimized for French text. The resulting vectors are indexed and saved locally using **FAISS**.

**4. Retrieval**
At inference time, the top-k most semantically similar chunks are retrieved via cosine similarity search based on the user query.

**5. Generation**
The retrieved chunks and the conversation history are injected into a structured prompt. A response is then generated through the **HuggingFace Inference API** using `meta-llama/Llama-3.1-8B-Instruct`.

---

## Evaluation with RAGAS

The performance of NorChat is evaluated using **RAGAS** (Retrieval-Augmented Generation Assessment), a framework designed to assess RAG systems without requiring extensive human annotations.

A benchmark dataset of representative university-related questions was manually created, along with their expected answers. For each question, NorChat retrieves relevant document chunks and generates a response. RAGAS then evaluates the system using four complementary metrics:

* **Faithfulness**: measures whether the generated answer is supported by the retrieved context.
* **Answer Relevancy**: evaluates how well the answer addresses the user's question.
* **Context Precision**: measures the proportion of retrieved chunks that are actually useful for answering the question.
* **Context Recall**: evaluates whether the retrieval step captures all the information required to produce the expected answer.

This evaluation provides quantitative insights into both the retrieval and generation components of the RAG pipeline and helps identify potential weaknesses in the system.

---

## Project Structure

```
NorChat/
│
├── data/                     # Official .md documents (knowledge base)
│
├── src/
│   ├── loader.py             # Document loading from data/
│   ├── chunker.py            # Text cleaning and chunk splitting
│   ├── embedder.py           # Embedding generation and FAISS indexing
│   ├── retriever.py          # Similarity-based chunk retrieval
│   ├── chain.py              # Prompt construction and LLM inference
│   ├── test.py               # Terminal-based conversational loop
│   └── test_ragas.py         # RAGAS-based evaluation of retrieval and generation quality
│
├── app.py                    # Streamlit user interface
├── stockage/                 # Persisted FAISS vector store (generated locally)
├── .env                      # HuggingFace API token (excluded from version control)
├── .gitignore
├── icon.png
├── logo_univ.jpg
├── logo.png
└── README.md
```

---

## Installation

```bash
pip install langchain langchain-community langchain-huggingface langchain-text-splitters \
            langchain-core faiss-cpu sentence-transformers python-dotenv \
            streamlit huggingface-hub --break-system-packages
```

---

## Configuration

Create a `.env` file at the root of the project:

```
HF_TOKEN=your_huggingface_token
```

Make sure `.env` is listed in `.gitignore` and never committed to version control.

---

## Usage

### Step 1 — Build the vector store (first-time setup only)

```bash
python src/embedder.py
```

### Step 2 — Run the terminal test

```bash
python src/test.py
```

### Step 3 — Launch the Streamlit application

```bash
streamlit run app.py
```

---

## Knowledge Base

NorChat answers questions based on three official documents:

| Document | Content |
|---|---|
| `master_sd.md` | Program overview, tracks (A2IA, SIME, MINMACS), courses, internships, career paths |
| `inscription_administration.md` | Application platforms, enrollment steps, required documents, tuition fees, international student procedures |
| `reglements.md` | Academic regulations, grading rules, compensation, exam procedures, RSE, plagiarism policy |

---

## Tech Stack

| Component | Tool / Model |
|---|---|
| RAG Framework | LangChain |
| Embeddings | OrdalieTech/Solon-embeddings-large-0.1 (HuggingFace) |
| Vector Store | FAISS (local, CPU) |
| Language Model | meta-llama/Llama-3.1-8B-Instruct (HuggingFace Inference API) |
| User Interface | Streamlit |
| Language | Python 3.12.3 |

---

## Design Decisions

**Why RAG over fine-tuning?**
Fine-tuning requires labeled training data and significant compute resources. RAG allows the model to remain grounded in up-to-date source documents without any retraining, making it more practical and maintainable for an academic project of this scale.

**Why Solon embeddings?**
`Solon-embeddings-large-0.1` is specifically trained on French text. It consistently outperforms multilingual or English-only embedding models for retrieval over French administrative documents.

**Why Llama 3.1 8B?**
It provides strong instruction-following capability at a size that remains accessible through the free HuggingFace Inference API, without requiring local GPU resources.

---

## Limitations

- Responses are strictly bounded by the content of the ingested documents. Questions outside the knowledge base are redirected to the official university website.
- The system does not have access to real-time data such as current deadlines or live announcements.
- Retrieval quality depends on how well the query matches the indexed chunks. Vague or ambiguous questions may result in less relevant context being retrieved.

---

## Impact

NorChat was built with one simple goal in mind: making university life less overwhelming for students, especially those who are new to the program or arriving from abroad. Instead of spending hours searching through administrative documents or waiting for office hours, students can just ask their question and get a clear, reliable answer in seconds.

---

## Authors

**Mariem GAALICHE**
</br>
Master 1 - Sciences des Données
</br>
Université de Rouen Normandie, France
</br>
2025-2026
