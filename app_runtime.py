"""
Runtime helpers shared by the console chatbot, Gradio UI, and evaluation.

Design:
- UI + eval are LOAD-ONLY (they never build the index).
- Indexing is performed explicitly via build_index.py.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from rag_pipeline import (
    IndexConfig,
    index_exists,
    build_and_persist_vectorstore,
    load_vectorstore,
    make_retriever,
    make_qa_chain,
    format_docs_for_llm,
    docs_to_citation_rows,
)


def require_vectorstore(cfg: IndexConfig, persist_dir: str):
    """Load an existing persisted vectorstore; raise a clear error if missing."""
    if not index_exists(persist_dir):
        raise FileNotFoundError(
            f"No persisted index found at '{persist_dir}'. "
            "Run: python build_index.py  (or python build_index.py --help)"
        )
    return load_vectorstore(cfg, persist_dir)


def build_vectorstore(cfg: IndexConfig, persist_dir: str):
    """Build + persist the vectorstore (explicit indexing step)."""
    return build_and_persist_vectorstore(cfg, persist_dir)


def make_runtime_retriever(vectorstore, base_cfg: IndexConfig, *, k: int, search_type: str):
    """Create a retriever from a persisted vectorstore with runtime retrieval overrides."""
    runtime_cfg = replace(base_cfg, k=int(k), search_type=search_type)
    return make_retriever(vectorstore, runtime_cfg)


def make_qa(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """Create the LLM QA chain (kept in one place for consistency)."""
    return make_qa_chain(model_name=model_name, temperature=temperature)
