"""
Console chatbot (load-only).

Usage:
  1) python build_index.py
  2) python console_chatbot.py
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from config import DEFAULT_CFG, PERSIST_DIR
from app_runtime import require_vectorstore, make_runtime_retriever, make_qa
from rag_pipeline import format_docs_for_llm, docs_to_citation_rows


def print_sources(docs, max_to_show: int = 5):
    rows = docs_to_citation_rows(docs, max_chars=200)[:max_to_show]
    if not rows:
        return
    print("\nSources:")
    for r in rows:
        print(f"  [{r['rank']}] {r['file']} — page {r['page']}")


def main():
    vectorstore = require_vectorstore(DEFAULT_CFG, PERSIST_DIR)
    qa_chain = make_qa(model_name="gpt-4o-mini", temperature=0.0)

    print("\n📚 RAG Research Papers Console Chatbot")
    print("Type a question (or 'exit' to quit).\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        retriever = make_runtime_retriever(vectorstore, DEFAULT_CFG, k=DEFAULT_CFG.k, search_type=DEFAULT_CFG.search_type)
        docs = retriever.invoke(q)
        ctx = format_docs_for_llm(docs)
        answer = qa_chain.invoke({"question": q, "context": ctx})

        print("\nBot:", answer)
        print_sources(docs)


if __name__ == "__main__":
    main()
