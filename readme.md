# 📚 RAG Research Papers Chatbot

A complete **Retrieval-Augmented Generation (RAG)** project built around a curated collection of academic research papers *about RAG itself*.

The project demonstrates:
- Persistent vector indexing with **Chroma**
- Multiple interaction modes (console + Gradio UI)
- Rigorous **evaluation with LangSmith**
- **Hyperparameter tuning tracked in MLflow**

It is designed with **reproducibility, debuggability, and evaluation-first workflows** in mind.

---

## 🚀 What This Project Does

1. **Indexes research papers (PDFs)** into a persistent vector store
2. **Answers user questions** using a RAG pipeline grounded strictly in retrieved paper content
3. **Displays citations and retrieved evidence** for transparency
4. **Evaluates retrieval & answer quality** using LangSmith (LLM-as-judge + deterministic metrics)
5. **Tunes retrieval hyperparameters** and tracks results in MLflow

The system explicitly separates:
- **Indexing** (one-time, expensive)
- **Inference** (fast, load-only)
- **Evaluation** (load-only, traceable)

---

## 📂 Project Structure

```
research-papers-chatbot/
│
├── research-papers/              # 📄 Raw PDFs of research papers
│
├── rag_core/                     # 🔁 Core RAG logic (single source of truth)
│   ├── config.py                 # Central configuration (paths, chunking, retrieval defaults)
│   ├── rag_pipeline.py           # Indexing, retrieval, QA chain, formatting utilities
│   ├── app_runtime.py            # Runtime helpers (load-only vs build-only logic)
│   └── build_index.py            # 🚨 The ONLY script that builds the vector index
│
├── evaluation/
│   ├── evaluate.py               # LangSmith batch evaluation pipeline
│   └── rag_eval_questions.csv    # Evaluation dataset (questions + gold answers)
│
├── tuning/
│   ├── hyperparameter-tuning.py  # MLflow + LangSmith hyperparameter tuning
│   ├── mlruns/                   # MLflow runs (local)
│   └── mlflow.db                 # MLflow SQLite backend
│
├── console_chatbot.py             # Terminal-based chatbot
├── rag_research_papers_chatbot_app.py  # Gradio UI
├── .env                           # API keys & environment variables
└── README.md
```

---

## 🧠 How the RAG System Works

### 1️⃣ Indexing (one-time)
**Script:** `rag_core/build_index.py`

- Loads PDFs from `research-papers/`
- Splits them into overlapping chunks
- Embeds chunks using OpenAI embeddings
- Persists vectors + metadata to **Chroma**

```bash
python rag_core/build_index.py
```

⚠️ **Only this script builds the index**. All other components are load-only.

---

### 2️⃣ Runtime RAG Pipeline
**Core logic:** `rag_core/rag_pipeline.py`

At query time:
1. Retrieve top-k chunks (MMR or similarity)
2. Format retrieved chunks with citations
3. Pass context to an LLM with a strict grounding prompt
4. Stream the answer back to the user

The model **must not hallucinate** and will say *"I don't know"* if the answer is not supported.

---

### 3️⃣ Shared Runtime Helpers
**File:** `rag_core/app_runtime.py`

Ensures:
- UI and evaluation scripts **never rebuild** the index
- Clear errors if the index is missing
- Consistent retriever and QA chain creation

This guarantees reproducibility across UI, console, and evaluation.

---

## 💬 Interaction Modes

### 🖥️ Console Chatbot
**File:** `console_chatbot.py`

```bash
python console_chatbot.py
```

- Simple terminal-based interaction
- Prints retrieved sources for transparency

---

### 🌐 Gradio Web App
**File:** `rag_research_papers_chatbot_app.py`

```bash
python rag_research_papers_chatbot_app.py
```

Features:
- Streaming chat interface
- Adjustable retriever settings (k, search type)
- Evidence table showing retrieved chunks
- Example question overlay

The UI **loads the existing index only** and fails fast if missing.

---

## 📊 Evaluation with LangSmith
**File:** `evaluation/evaluate.py`

This script performs **batch evaluation** against a labeled dataset.

Metrics:
- **retrieval_hit** (deterministic)
- **correctness_0_10** (LLM-as-judge vs reference)
- **groundedness_0_10** (LLM-as-judge vs retrieved context)

```bash
python evaluation/evaluate.py
```

All runs are:
- Fully traced in LangSmith
- Linked to retrieved chunks and prompts

---

## 🔬 Hyperparameter Tuning (MLflow + LangSmith)
**File:** `tuning/hyperparameter-tuning.py`

What it does:
- Iterates over chunk sizes and retriever k values
- Runs LangSmith evaluation for each configuration
- Logs **aggregated metrics** to MLflow

Tracked in MLflow:
- chunk size
- retriever k
- search type
- retrieval_hit
- correctness_0_10
- groundedness_0_10

```bash
python tuning/hyperparameter-tuning.py
```

This enables systematic comparison of RAG configurations.

---

## 🔐 Environment Variables
Create a `.env` file:

```
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=rag-papers
```

Optional:
```
MLFLOW_TRACKING_URI=file:./mlruns
```

---

## ✅ Design Principles

- **Explicit indexing** (no accidental re-embedding)
- **Load-only inference & evaluation**
- **Single source of truth** for RAG logic
- **Full observability** (LangSmith + MLflow)
- **Reproducible experiments**

---

## 📌 Typical Workflow

```bash
# 1. Build index (once)
python rag_core/build_index.py

# 2. Chat via UI or console
python rag_research_papers_chatbot_app.py
# or
python console_chatbot.py

# 3. Evaluate
python evaluation/evaluate.py

# 4. Tune
python tuning/hyperparameter-tuning.py
```

---

## 🧠 Summary

This project is a **production-quality RAG research sandbox**:
- Transparent
- Evaluated
- Tunable
- Grounded

Perfect for:
- RAG experimentation
- Research paper QA
- Evaluation-driven LLM development

