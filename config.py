"""
Central configuration for the RAG Research Papers project.

Why this exists:
- single source of truth for PDF_DIR, PERSIST_DIR, chunking, embeddings, retriever defaults.
"""
from __future__ import annotations

from dataclasses import replace

from rag_pipeline import IndexConfig

# Folder containing your PDFs (10 research papers)
PDF_DIR: str = "research-papers"

# Chroma persistence directory
PERSIST_DIR: str = "chroma_db"

# Single source of truth for indexing + retrieval defaults
DEFAULT_CFG: IndexConfig = IndexConfig(
    pdf_dir=PDF_DIR,
    embedding_model="text-embedding-3-large",
    chunk_size=1000,
    chunk_overlap=200,
    search_type="mmr",
    k=8,
    fetch_k=20,
    lambda_mult=0.5,
)


def cfg_with(**overrides) -> IndexConfig:
    """Return a new IndexConfig derived from DEFAULT_CFG with overrides applied."""
    return replace(DEFAULT_CFG, **overrides)
