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


def _make_runtime_retriever(k: int, search_type: str):
    """Create a retriever backed by the same persisted vectorstore but with runtime search params."""
    runtime_cfg = IndexConfig(
        pdf_dir=cfg.pdf_dir,
        embedding_model=cfg.embedding_model,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=cfg.separators,
        search_type=search_type,
        k=k,
        # Keep MMR tuning fixed (hidden in UI); can be exposed later if desired.
        fetch_k=cfg.fetch_k,
        lambda_mult=cfg.lambda_mult,
    )
    return make_retriever(vectorstore, runtime_cfg)


def _rows_to_table(rows):
    """Convert list[dict] citation rows to a flat table (list[list]) for Gradio Dataframe."""
    headers = ["rank", "file", "page", "snippet"]
    table = []
    for r in rows:
        table.append([r.get(h, "") for h in headers])
    return table


def answer_question(
    message: str,
    history: list,
    retrieval_k: int,
    search_type: str,
):
    """Main chat callback: retrieve -> build context -> answer -> show evidence."""
    message = (message or "").strip()
    if not message:
        return history, [], ""

    retriever = _make_runtime_retriever(
        k=int(retrieval_k),
        search_type=search_type,
    )

    docs = retriever.invoke(message)
    context = format_docs_for_llm(docs)
    answer = qa_chain.invoke({"question": message, "context": context})

    history = history or []
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    rows = docs_to_citation_rows(docs, max_chars=260)
    evidence = _rows_to_table(rows)
    return history, evidence, ""


def clear_all():
    """Clear chat + evidence outputs."""
    return [], [], ""


with gr.Blocks(title="RAG Research Papers Chatbot", css="""
footer {display:none !important;}
.footer {display:none !important;}
#footer {display:none !important;}
.gradio-footer {display:none !important;}
""") as demo:
    gr.Markdown(
        "# 📚 RAG Research Papers Chatbot\n"
        "Ask questions about the research papers in `research-papers/`.\n\n"
        "**Tip:** Retrieved chunks are always shown in the Evidence table."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="Chat", height=520)
            msg = gr.Textbox(
                label="Your question",
                placeholder="e.g., What is GraphRAG? How does LightRAG differ from OpenRAG?",
                lines=2,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

            gr.HTML(
                """
                <a href="https://mostafaabla.com/projects.html" target="_self" style="text-decoration:none;">
                    <button style="
                        width:100%;
                        padding:10px;
                        border-radius:6px;
                        border:1px solid #ccc;
                        background:#f5f5f5;
                        cursor:pointer;
                    ">
                        Back
                    </button>
                </a>
                """
            )

        with gr.Column(scale=2):
            gr.Markdown("## Retrieval & Evidence")
            retrieval_k = gr.Slider(1, 20, value=8, step=1, label="Retriever k (chunks fetched)")
            search_type = gr.Dropdown(choices=["mmr", "similarity"], value="mmr", label="Search type")
            evidence_df = gr.Dataframe(
                    headers=["rank", "file", "page", "snippet"],
                    datatype=["number", "str", "str", "str"],
                    label="Retrieved chunks",
                    wrap=True,
                    interactive=False,
                )

    send.click(
        answer_question,
        inputs=[msg, chat, retrieval_k, search_type],
        outputs=[chat, evidence_df, msg],
    )
    msg.submit(
        answer_question,
        inputs=[msg, chat, retrieval_k, search_type],
        outputs=[chat, evidence_df, msg],
    )
    clear.click(clear_all, outputs=[chat, evidence_df, msg])

if __name__ == "__main__":
    demo.launch()
