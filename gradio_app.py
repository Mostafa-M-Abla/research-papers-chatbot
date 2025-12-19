
import os
from dotenv import load_dotenv
import gradio as gr

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

# -----------------------
# Config
# -----------------------
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

# -----------------------
# Vectorstore (build once, then reuse)
# -----------------------
if index_exists(PERSIST_DIR):
    vectorstore = load_vectorstore(cfg, PERSIST_DIR)
    print(f"Loaded persisted vectorstore from '{PERSIST_DIR}'.")
else:
    print(f"No persisted vectorstore found at '{PERSIST_DIR}'. Building index now...")
    vectorstore, pdf_paths = build_and_persist_vectorstore(cfg, PERSIST_DIR)
    print(f"Built and persisted vectorstore to '{PERSIST_DIR}'. PDFs indexed: {len(pdf_paths)}")

# Build QA chain once; retriever is created per-request so UI can vary search params.
qa_chain = make_qa_chain(model_name="gpt-4o-mini", temperature=0.0)


def _make_runtime_retriever(k: int, search_type: str, fetch_k: int, lambda_mult: float):
    """Create a retriever backed by the same persisted vectorstore but with runtime search params."""
    runtime_cfg = IndexConfig(
        pdf_dir=cfg.pdf_dir,
        embedding_model=cfg.embedding_model,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
        search_type=search_type,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    return make_retriever(vectorstore, runtime_cfg)


def answer_question(
    message: str,
    history: list,
    show_chunks: bool,
    top_k_to_show: int,
    retrieval_k: int,
    search_type: str,
    fetch_k: int,
    lambda_mult: float,
):
    """Main chat callback: retrieve -> build context -> answer -> optionally show evidence."""
    message = (message or "").strip()
    if not message:
        return history, [], ""

    retriever = _make_runtime_retriever(
        k=int(retrieval_k),
        search_type=search_type,
        fetch_k=int(fetch_k),
        lambda_mult=float(lambda_mult),
    )

    docs = retriever.invoke(message)
    context = format_docs_for_llm(docs)
    answer = qa_chain.invoke({"question": message, "context": context})

    history = history or []
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    rows = docs_to_citation_rows(docs, max_chars=260)[: int(top_k_to_show)]
    evidence = rows if show_chunks else []

    # Short sources preview (always shown)
    sources_lines = [f"[{r['rank']}] {r['file']} — Page {r['page']}" for r in rows[: min(5, len(rows))]]
    sources_md = "\n".join(sources_lines) if sources_lines else ""

    return history, evidence, sources_md


def clear_all():
    """Clear chat + evidence outputs."""
    return [], [], ""


with gr.Blocks(title="RAG Research Papers Chatbot") as demo:
    gr.Markdown(
        "# 📚 RAG Research Papers Chatbot\n"
        "Ask questions about the research papers in `research-papers/`.\n\n"
        "**Tip:** Enable *Show retrieved chunks* to see the evidence used."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="Chat", height=520)
            sources_preview = gr.Markdown(label="Sources (top few)")
            msg = gr.Textbox(
                label="Your question",
                placeholder="e.g., What is GraphRAG? How does LightRAG differ from OpenRAG?",
                lines=2,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

        with gr.Column(scale=2):
            gr.Markdown("## Retrieval & Evidence")
            show_chunks = gr.Checkbox(value=False, label="Show retrieved chunks (evidence)")
            top_k_to_show = gr.Slider(1, 15, value=8, step=1, label="Top-k chunks to display")
            retrieval_k = gr.Slider(1, 20, value=8, step=1, label="Retriever k (chunks fetched)")
            search_type = gr.Dropdown(choices=["mmr", "similarity"], value="mmr", label="Search type")
            fetch_k = gr.Slider(5, 50, value=20, step=1, label="MMR fetch_k (candidate pool)")
            lambda_mult = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="MMR lambda_mult (diversity)")

            with gr.Accordion("Evidence table", open=False):
                evidence_df = gr.Dataframe(
                    headers=["rank", "file", "page", "snippet", "source_path"],
                    datatype=["number", "str", "str", "str", "str"],
                    label="Retrieved chunks",
                    wrap=True,
                    interactive=False,
                )

    send.click(
        answer_question,
        inputs=[msg, chat, show_chunks, top_k_to_show, retrieval_k, search_type, fetch_k, lambda_mult],
        outputs=[chat, evidence_df, sources_preview],
    )
    msg.submit(
        answer_question,
        inputs=[msg, chat, show_chunks, top_k_to_show, retrieval_k, search_type, fetch_k, lambda_mult],
        outputs=[chat, evidence_df, sources_preview],
    )
    clear.click(clear_all, outputs=[chat, evidence_df, sources_preview])

if __name__ == "__main__":
    demo.launch()
