# RAG Complaint Chatbot — CrediTrust Financial

An internal AI tool that turns raw, unstructured CFPB customer complaint data into a queryable knowledge base, enabling Product, Support, and Compliance teams at CrediTrust Financial to ask plain-English questions and get evidence-backed answers — without needing a data analyst.

> **Status:** Interim submission — Task 1 (EDA & Preprocessing) and Task 2 (Chunking, Embedding & Vector Store) complete. Task 3 (RAG core logic) and Task 4 (interactive UI) are in progress.

[![Unit Tests](https://github.com/sosena2/rag-complaint-chatbot/actions/workflows/unittests.yml/badge.svg)](https://github.com/sosena2/rag-complaint-chatbot/actions/workflows/unittests.yml)

---

## Project Overview

CrediTrust Financial is a digital finance company serving East African markets across credit cards, personal loans, savings accounts, and money transfers. With 500,000+ users, the company receives thousands of complaints per month, scattered across in-app channels, email, and regulatory portals — making it hard for internal teams to spot trends quickly.

This project builds a Retrieval-Augmented Generation (RAG) chatbot that:
- Lets internal users ask natural-language questions about customer complaints (e.g., *"Why are people unhappy with Credit Cards?"*)
- Retrieves the most relevant complaint narratives via semantic search over a vector database
- Feeds retrieved context into an LLM to generate concise, grounded answers
- Supports filtering/comparison across four product categories: **Credit Card, Personal Loan, Savings Account, Money Transfer**

---

## Project Structure

```
rag-complaint-chatbot/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                  # Original CFPB CSV (gitignored — not committed)
│   └── processed/             # Cleaned/filtered dataset (gitignored — not committed)
├── vector_store/               # Persisted ChromaDB index
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_chunking_embedding.ipynb
│   └── README.md
├── src/
├── tests/
├── app.py                      # Gradio/Streamlit interface (Task 4 — pending)
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** Raw and processed data files are intentionally excluded from version control (see `.gitignore`) to keep the repository lightweight. Run the notebooks locally to regenerate them — see [Setup](#setup--installation) below.

---

## Task 1: Exploratory Data Analysis & Preprocessing

**Notebook:** [`notebooks/task1_eda_preprocessing.ipynb`](notebooks/task1_eda_preprocessing.ipynb)

**What was done:**
- Loaded the full CFPB consumer complaints dataset
- Analyzed complaint distribution across all product categories
- Calculated and visualized narrative word-count distribution, flagging very short and very long entries
- Counted complaints with vs. without a consumer narrative
- Filtered the dataset down to four target categories: **Credit Card, Personal Loan, Savings Account, Money Transfer**
- Removed records with empty narrative fields
- Cleaned narrative text: lowercased, stripped boilerplate phrases (e.g. "I am writing to file a complaint..."), removed redaction placeholders (`XXXX`) and special characters
- Saved the cleaned, filtered dataset to `data/processed/filtered_complaints.csv`

**Key EDA findings:** *(fill in your actual numbers once finalized)*
- Total raw records: `[N]`
- Records with a narrative: `[N]` (`[X]%` of total)
- Records after filtering to the 4 target categories with non-empty narratives: `[N]`
- Median narrative length: `[N]` words (range: `[min]`–`[max]`)

---

## Task 2: Chunking, Embedding & Vector Store Indexing

**Notebook:** [`notebooks/task2_chunking_embedding.ipynb`](notebooks/task2_chunking_embedding.ipynb)

**What was done:**
- Drew a **stratified sample of ~12,000 complaints** from the cleaned dataset, proportionally split across the four product categories
- Chunked narratives using LangChain's `RecursiveCharacterTextSplitter` with **chunk size 500 / overlap 50**, balancing semantic coherence against embedding model limits
- Generated embeddings using **`sentence-transformers/all-MiniLM-L6-v2`** (384-dim) — chosen for its strong performance on semantic similarity tasks at a small, fast footprint, suitable for local development on limited hardware
- Indexed embeddings into a **ChromaDB** persistent vector store at `vector_store/`, with per-chunk metadata (`complaint_id`, `product_category`, `issue`, `company`, `chunk_index`, `total_chunks`) to trace retrieved chunks back to their source complaint

**Sampling strategy:** Proportional stratified sampling by `product_category`, preserving the relative frequency of each category from the cleaned dataset within a 12,000-record sample — keeping the sample representative while staying computationally feasible on local hardware.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A Hugging Face account + read-access token (for downloading the embedding model) — see [Hugging Face tokens](https://huggingface.co/settings/tokens)

### 1. Clone the repository
```bash
git clone https://github.com/sosena2/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

### 2. Create a virtual environment and install dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --default-timeout=200 -r requirements.txt
```

### 3. Set up your Hugging Face token
Create a `.env` file in the project root (already gitignored):
```
HF_TOKEN=hf_your_token_here
```

### 4. Add the raw data
Download the full CFPB complaints dataset and place it at:
```
data/raw/complaints.csv
```

### 5. Run the notebooks in order
```
notebooks/task1_eda_preprocessing.ipynb
notebooks/task2_chunking_embedding.ipynb
```

---

## Tech Stack

| Component | Choice |
|---|---|
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| UI (planned, Task 4) | Gradio or Streamlit |

---

## Roadmap

- [x] **Task 1:** EDA and data preprocessing
- [x] **Task 2:** Chunking, embedding, vector store indexing
- [ ] **Task 3:** RAG retrieval + generation pipeline, qualitative evaluation
- [ ] **Task 4:** Interactive chat interface (Gradio/Streamlit)

---

