
import os
from dotenv import load_dotenv
load_dotenv()

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

PDF_DIR = "research-papers"
PERSIST_DIR = "chroma_db"

cfg = IndexConfig(
    pdf_dir=PDF_DIR,
    embedding_model="text-embedding-3-large",
    chunk_size=1000,
    chunk_overlap=200,
    search_type="mmr",
    k=8,
    fetch_k=20,
    lambda_mult=0.5,
)

# Persisted vector store behavior:
# - Build once if missing
# - Load on subsequent runs
if index_exists(PERSIST_DIR):
    vectorstore = load_vectorstore(cfg, PERSIST_DIR)
    print(f"Loaded persisted vectorstore from '{PERSIST_DIR}'.")
else:
    print(f"No persisted vectorstore found at '{PERSIST_DIR}'. Building index now...")
    vectorstore, pdf_paths = build_and_persist_vectorstore(cfg, PERSIST_DIR)
    print(f"Built and persisted vectorstore to '{PERSIST_DIR}'. PDFs indexed: {len(pdf_paths)}")

retriever = make_retriever(vectorstore, cfg)
qa_chain = make_qa_chain(model_name="gpt-4o-mini", temperature=0.0)

if __name__ == "__main__":
    print("\n📚 RAG Research Papers Chatbot (persistent index)")
    print("Ask me anything about the PDFs in 'research-papers/'.")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("Bot: Goodbye! 👋")
                break
            if not question:
                continue

            docs = retriever.invoke(question)
            context = format_docs_for_llm(docs)
            answer = qa_chain.invoke({"question": question, "context": context})

            print(f"\nBot: {answer}\n")

            # Show exactly which chunks were retrieved
            retrieved_rows = docs_to_citation_rows(docs, max_chars=200)
            print("Sources:")
            for r in retrieved_rows:
                print(f"[{r['rank']}] {r['file']}, Page {r['page']}: {r['snippet']}...")
            print()

        except KeyboardInterrupt:
            print("\nBot: Goodbye! 👋")
            break
