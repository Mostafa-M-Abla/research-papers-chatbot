"""
One-time (explicit) indexing script.

This is the ONLY script that should build/re-embed the vectorstore.
- Gradio UI and evaluation scripts are load-only.
- Keeps UI responsive and prevents accidental re-indexing.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
from dataclasses import replace

from config import DEFAULT_CFG, PERSIST_DIR, PDF_DIR
from app_runtime import build_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build and persist the Chroma index for the research papers.")
    parser.add_argument("--pdf_dir", default=PDF_DIR, help="Folder containing PDFs to index.")
    parser.add_argument("--persist_dir", default=PERSIST_DIR, help="Directory to persist Chroma index.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CFG.chunk_size, help="Chunk size (characters).")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CFG.chunk_overlap, help="Chunk overlap (characters).")
    args = parser.parse_args()

    cfg = replace(
        DEFAULT_CFG,
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    vectorstore, pdf_paths = build_vectorstore(cfg, args.persist_dir)
    print(f"✅ Built and persisted index to: {args.persist_dir}")
    print(f"📄 PDFs indexed: {len(pdf_paths)}")
    for p in pdf_paths:
        print(" -", p)


if __name__ == "__main__":
    main()
