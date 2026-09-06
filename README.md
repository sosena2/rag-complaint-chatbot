# RAG Complaint Chatbot — CrediTrust Financial

An internal AI tool that turns raw, unstructured CFPB customer complaint data into a queryable knowledge base, enabling Product, Support, and Compliance teams at CrediTrust Financial to ask plain-English questions and get evidence-backed answers — without needing a data analyst.

> **Status:** All tasks complete — EDA & Preprocessing, Chunking & Vector Store Indexing, RAG Core Logic & Evaluation, and the interactive chat interface with streaming responses.

[![Unit Tests](https://github.com/sosena2/rag-complaint-chatbot/actions/workflows/unittests.yml/badge.svg)](https://github.com/sosena2/rag-complaint-chatbot/actions/workflows/unittests.yml)

---

## Project Overview

CrediTrust Financial is a digital finance company serving East African markets across credit cards, personal loans, savings accounts, and money transfers. With 500,000+ users, the company receives thousands of complaints per month, scattered across in-app channels, email, and regulatory portals — making it hard for internal teams to spot trends quickly.

This project builds a Retrieval-Augmented Generation (RAG) chatbot that:
- Lets internal users ask natural-language questions about customer complaints (e.g., *"Why are people unhappy with Credit Cards?"*)
- Retrieves the most relevant complaint narratives via semantic search over a vector database
- Feeds retrieved context into an LLM to generate concise, grounded answers, streamed token-by-token
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
│   ├── raw/                  # complaints.csv, complaint_embeddings.parquet (gitignored)
│   └── processed/            # filtered_complaints.csv, task3_evaluation_raw.csv (gitignored)
├── vector_store/              # Persisted ChromaDB index (gitignored)
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb
│   ├── task2_chunking_embedding.ipynb
│   ├── task3_rag_pipeline.ipynb
│   └── README.md
├── src/
│   └── rag_pipeline.py        # Single source of truth for retrieval + generation logic
├── tests/
├── app.py                     # Gradio interface with streaming responses
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** Raw and processed data files, plus the vector store, are intentionally excluded from version control (see `.gitignore`) to keep the repository lightweight. Run the notebooks locally to regenerate them — see [Setup](#setup--installation) below.

---

## Architecture

`src/rag_pipeline.py` contains a single `RAGPipeline` class that owns all retrieval and generation logic — the embedding model, the ChromaDB connection, and the Hugging Face inference client. Both `app.py` and `notebooks/task3_rag_pipeline.ipynb` import and use this same class, so there is exactly one implementation of the RAG logic to maintain and debug, rather than duplicated copies drifting out of sync.

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
result = rag.answer("Why are people unhappy with Credit Cards?")
print(result["answer"])
print(result["sources"])
```

---

## Task 1: Exploratory Data Analysis & Preprocessing

**Notebook:** [`notebooks/task1_eda_preprocessing.ipynb`](notebooks/task1_eda_preprocessing.ipynb)

**What was done:**
- Loaded the full CFPB consumer complaints dataset in chunks (the raw file is large enough to exceed available memory if loaded in one pass — see [Lessons Learned](#lessons-learned--known-issues))
- Analyzed complaint distribution across all product categories
- Calculated and visualized narrative word-count distribution, flagging very short and very long entries
- Counted complaints with vs. without a consumer narrative
- Filtered the dataset down to four target categories: **Credit Card, Personal Loan, Savings Account, Money Transfer**
- Removed records with empty narrative fields
- Cleaned narrative text: lowercased, stripped boilerplate phrases (e.g. "I am writing to file a complaint..."), removed redaction placeholders (`XXXX`) and special characters
- Saved the cleaned, filtered dataset to `data/processed/filtered_complaints.csv`

---

## Task 2: Chunking, Embedding & Vector Store Indexing

**Notebook:** [`notebooks/task2_chunking_embedding.ipynb`](notebooks/task2_chunking_embedding.ipynb)

**What was done:**
- Drew a **stratified sample of ~12,000 complaints** from the cleaned dataset, proportionally split across the four product categories
- Chunked narratives using LangChain's `RecursiveCharacterTextSplitter` with **chunk size 500 / overlap 50**, balancing semantic coherence against embedding model limits
- Generated embeddings using **`sentence-transformers/all-MiniLM-L6-v2`** (384-dim) — chosen for its strong performance on semantic similarity tasks at a small, fast footprint, suitable for local development on limited hardware
- Indexed embeddings into a **ChromaDB** persistent vector store, with per-chunk metadata (`complaint_id`, `product_category`, `issue`, `company`, `chunk_index`, `total_chunks`) to trace retrieved chunks back to their source complaint

**Sampling strategy:** Proportional stratified sampling by `product_category`, preserving the relative frequency of each category from the cleaned dataset within the sample — keeping it representative while staying computationally feasible on local hardware.

---

## Task 3: RAG Core Logic and Evaluation

**Notebook:** [`notebooks/task3_rag_pipeline.ipynb`](notebooks/task3_rag_pipeline.ipynb) · **Module:** [`src/rag_pipeline.py`](src/rag_pipeline.py)

**What was done:**
- Rebuilt the vector store from the pre-built `complaint_embeddings.parquet` (covering the full filtered dataset, not just the 12k sample), indexing in batches via `PersistentClient`
- Implemented `RAGPipeline.retrieve()`: embeds the incoming question with the same `all-MiniLM-L6-v2` model and performs similarity search (`k=5` default) against the vector store, with optional metadata filtering by product category
- Designed a prompt template instructing the model to act as a financial analyst, answer only from the retrieved context, and explicitly say when the context is insufficient
- Implemented generation via the Hugging Face `InferenceClient`, using `chat_completion()` against **`openai/gpt-oss-120b`**
- Ran a qualitative evaluation across 5 representative questions, scoring each on a 1–5 scale with written analysis — see `data/processed/task3_evaluation_raw.csv`

### Evaluation table (summary)

| Question | Quality Score | Notes |
|---|---|---|
| Why are people unhappy with Credit Cards? | 4/5 | Grounded in retrieved context, no hallucination |
| What are the most common complaints about money transfers? | 3–4/5 | Relevant; initial run had markdown formatting artifacts, corrected in prompt |
| Are customers reporting unauthorized charges? | 4/5 | Correctly identifies theme from retrieved excerpts |
| What issues do customers have with personal loans? | 4/5 | Coherent summary from context |
| Are there complaints about savings account fees? | 3–4/5 | Confirms the pattern; could cite more specific figures from sources |

Full detail, including the exact generated answers and retrieved sources per question, is in `data/processed/task3_evaluation_raw.csv`.

---

## Task 4: Interactive Chat Interface

**Script:** [`app.py`](app.py)

**What was done:**
- Built a Gradio interface with a question input, product-category filter, adjustable top-k source count, and Ask/Clear buttons
- **Displays retrieved sources** below the generated answer (product, issue, complaint ID, and a text excerpt) for user trust and verification
- **Streaming responses**: the answer renders token-by-token instead of appearing all at once, via `RAGPipeline.answer_stream()` and a Gradio generator function
- Example questions pre-loaded for quick testing

### Running the app

```powershell
python -m app
```
Then open the printed local URL (typically `http://127.0.0.1:7860`) in your browser.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A Hugging Face account + access token with **"Make calls to Inference Providers"** permission enabled (see [Lessons Learned](#lessons-learned--known-issues) for a note on model availability)

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
Create a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
```

### 4. Add the data
```powershell
mkdir data\raw, data\processed
```
Place the following in `data/raw/`:
- `complaints.csv` — full CFPB complaints dataset (for Task 1)
- `complaint_embeddings.parquet` — pre-built embeddings (for Task 3)

### 5. Run the notebooks in order
```
notebooks/task1_eda_preprocessing.ipynb
notebooks/task2_chunking_embedding.ipynb
notebooks/task3_rag_pipeline.ipynb
```
Each rebuilds a piece of the pipeline; Task 3 is the one that produces the final vector store your app queries.

### 6. Launch the app
```powershell
python -m app
```

---

## Lessons Learned / Known Issues

- **Large CSV reads can exceed available memory** when loaded in a single `pd.read_csv()` call. Reading in chunks and filtering to the target product categories as you go keeps peak memory bounded to one chunk at a time.
- **Hugging Face serverless Inference Provider availability shifts over time.** Several well-known models (`mistralai/Mistral-7B-Instruct-v0.2`, `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-2-2b-it`) returned `model_not_supported` errors despite valid tokens and correctly enabled account permissions, because none of the enabled providers currently served them. The fix: check live provider support before committing to a model —
  ```python
  from huggingface_hub import model_info
  info = model_info("model/name", expand="inferenceProviderMapping")
  print(info.inference_provider_mapping)
  ```
  `openai/gpt-oss-120b` was chosen as the generator model specifically because it showed `status='live'` across the most providers simultaneously (Groq, Novita, Cerebras, Together AI, Fireworks, Featherless AI, Scaleway), making it the most resilient choice if any single provider becomes unavailable.
- **ChromaDB persistent stores can become corrupted** (`InternalError: Failed to apply logs to the metadata segment`) if multiple kernels/processes hold a connection to the same store path simultaneously, or if the directory is only partially deleted before a rebuild. Shut down all other notebook kernels before wiping and rebuilding `vector_store/`.

---

## Tech Stack

| Component | Choice |
|---|---|
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Generator model | `openai/gpt-oss-120b` via Hugging Face Inference Providers |
| UI | Gradio (with streaming responses) |

---

## Roadmap

- [x] **Task 1:** EDA and data preprocessing
- [x] **Task 2:** Chunking, embedding, vector store indexing
- [x] **Task 3:** RAG retrieval + generation pipeline, qualitative evaluation
- [x] **Task 4:** Interactive chat interface with streaming responses