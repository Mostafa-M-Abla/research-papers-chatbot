
"""
Shared RAG utilities used by BOTH:
- research_paper_rag_persist.py (chatbot)
- evaluate_langsmith_rag_persist.py (LangSmith evaluation)

Design goals:
- Persist Chroma vectorstore to disk.
- Avoid code duplication (single source of truth for indexing + retrieval + formatting).
- Evaluation script must NEVER (re)build the vector store; it only loads.
"""

from __future__ import annotations

import glob
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb


@dataclass(frozen=True)
class IndexConfig:
    pdf_dir: str
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Tuple[str, ...] = ("\n\n", "\n", " ", "")
    # retrieval defaults
    search_type: str = "mmr"
    k: int = 8
    fetch_k: int = 20
    lambda_mult: float = 0.5


def _list_pdfs(pdf_dir: str) -> List[str]:
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in folder: {pdf_dir}")
    return pdf_paths


def index_exists(persist_dir: str) -> bool:
    return os.path.isdir(persist_dir) and any(os.scandir(persist_dir))


def build_and_persist_vectorstore(cfg: IndexConfig, persist_dir: str) -> Tuple[Chroma, List[str]]:
    """
    Build vectorstore from PDFs and persist to disk.
    Use this ONLY from chatbot/indexing flow (not evaluation).
    """
    pdf_paths = _list_pdfs(cfg.pdf_dir)
    embeddings = OpenAIEmbeddings(model=cfg.embedding_model)

    pages = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        doc_pages = loader.load_and_split()
        for page in doc_pages:
            page.metadata["source"] = path
        pages.extend(doc_pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=list(cfg.separators),
    )
    chunks = splitter.split_documents(pages)

    client = chromadb.PersistentClient(path=persist_dir)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="rag_papers",  # optional but nice to name it
    )

    _write_index_config(persist_dir, cfg, pdf_paths)
    return vs, pdf_paths


def load_vectorstore(cfg: IndexConfig, persist_dir: str) -> Chroma:
    """
    Load an EXISTING persisted vectorstore.
    Evaluation should use this; it errors if missing.
    """
    if not index_exists(persist_dir):
        raise FileNotFoundError(
            f"Persisted vectorstore not found at '{persist_dir}'. "
            f"Build it first by running the chatbot (or a separate indexing step)."
        )
    # This line creates a client opject, stores configuration (model name, API key, etc.) and prepares a callable embedding function
    #i.e. it creates a handle that can call the OpenAI Embeddings API later. It doesn't make API calls
    embeddings = OpenAIEmbeddings(model=cfg.embedding_model)

    client = chromadb.PersistentClient(path=persist_dir)

    return Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="rag_papers",
    )

    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def make_retriever(vectorstore: Chroma, cfg: IndexConfig):
    # Only MMR supports fetch_k and lambda_mult
    if cfg.search_type == "mmr":
        search_kwargs = {"k": cfg.k, "fetch_k": cfg.fetch_k, "lambda_mult": cfg.lambda_mult}
    else:
        # similarity / similarity_score_threshold etc.
        search_kwargs = {"k": cfg.k}

    return vectorstore.as_retriever(
        search_type=cfg.search_type,
        search_kwargs=search_kwargs,
    )


def make_qa_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    template = """
SYSTEM: You are a question-answering assistant.
SYSTEM:
You are an AI research assistant.
You answer questions using a collection of academic research papers about
Retrieval-Augmented Generation (RAG), including surveys, system designs,
evaluation methods, and recent advancements.

You may:
- summarize information across multiple documents
- synthesize high-level insights
- compare approaches described in different papers
- summarize aa certain paper

You must:
- use ONLY the provided context
- NOT introduce external knowledge
- base all claims on the retrieved text

If the answer cannot be reasonably inferred from the context, say:
"I don't know based on the provided documents."

QUESTION:
{question}

CONTEXT:
{context}
"""
    prompt = PromptTemplate.from_template(template.strip())
    return prompt | llm | StrOutputParser()


def format_docs_for_llm(docs) -> str:
    out = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        file_name = os.path.basename(src) if src else "unknown"
        page = d.metadata.get("page", None)
        page_display = page + 1 if isinstance(page, int) else page
        out.append(f"[Source {i} | {file_name} | Page {page_display}]\n{d.page_content}")
    return "\n\n".join(out)


# Convert retrieved LangChain Document objects into structured citation records.
# Each record represents one retrieved chunk and includes:
# - retrieval rank (order returned by the retriever)
# - source PDF filename
# - human-readable page number (1-based indexing)
# - a short text snippet for preview/debugging
# - full source path for traceability
#
# This structure is used for:
# - displaying sources to users (citations)
# - logging and debugging retrieval behavior
# - exporting evaluation results
# - computing retrieval-based metrics during evaluation
#
# Keeping this logic centralized ensures consistent, explainable,
# and reproducible RAG behavior across the chatbot and evaluation pipeline.
def docs_to_citation_rows(docs, max_chars: int = 500) -> List[Dict[str, Any]]:
    rows = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        file_name = os.path.basename(src) if src else "unknown"
        page = d.metadata.get("page", None)
        page_display = page + 1 if isinstance(page, int) else page
        rows.append(
            {
                "rank": i,
                "file": file_name,
                "page": page_display,
                "snippet": (d.page_content or "")[:max_chars].replace("\n", " "),
                "source_path": src,
            }
        )
    return rows


# Return the canonical path where index metadata/configuration is stored.
# This file records how the vector index was built (chunking, embeddings, PDFs)
# and is used to ensure reproducibility and prevent silent mismatches between
# indexing, inference, and evaluation runs.
def _index_config_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "index_config.json")


# Persist index configuration and source document metadata alongside the vector store.
# This captures:
# - indexing parameters (chunk size, overlap, embedding model, retrieval defaults)
# - the list of PDF files used to build the index
#
# Storing this metadata enables:
# - reproducible experiments
# - auditability of evaluation results
# - detection of stale or incompatible indexes when configurations change
def _write_index_config(persist_dir: str, cfg: IndexConfig, pdf_paths: List[str]) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    payload = {"index_config": asdict(cfg), "pdf_files": [os.path.basename(p) for p in pdf_paths]}
    import json
    with open(_index_config_path(persist_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
