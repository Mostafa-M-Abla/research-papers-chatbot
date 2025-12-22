"""
Gradio UI (load-only) for the RAG Research Papers chatbot.

Important:
- This UI NEVER builds the vectorstore.
- Build the index first:  python build_index.py
"""
from __future__ import annotations

from dotenv import load_dotenv
import gradio as gr

load_dotenv()

from config import DEFAULT_CFG, PERSIST_DIR
from app_runtime import require_vectorstore, make_runtime_retriever, make_qa
from rag_pipeline import format_docs_for_llm, docs_to_citation_rows


def _rows_to_table(rows):
    headers = ["rank", "file", "page", "snippet"]
    return [[r.get(h, "") for h in headers] for r in rows]


def answer_question(message: str, history: list, retrieval_k: int, search_type: str):
    message = (message or "").strip()
    if not message:
        return history, [], ""

    retriever = make_runtime_retriever(vectorstore, DEFAULT_CFG, k=int(retrieval_k), search_type=search_type)
    docs = retriever.invoke(message)

    ctx = format_docs_for_llm(docs)
    answer = qa_chain.invoke({"question": message, "context": ctx})

    history = history or []
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    rows = docs_to_citation_rows(docs, max_chars=260)
    evidence = _rows_to_table(rows)

    # Clear the input textbox after send
    return history, evidence, ""


def clear_all():
    return [], [], ""


# Load-only: fail fast with a clear message if index missing
vectorstore = require_vectorstore(DEFAULT_CFG, PERSIST_DIR)
qa_chain = make_qa(model_name="gpt-4o-mini", temperature=0.0)

# Older Gradio versions accept css in Blocks; keep it here for broad compatibility.
CSS = """
footer {display:none !important;}
.footer {display:none !important;}
#footer {display:none !important;}
.gradio-footer {display:none !important;}
"""

with gr.Blocks(title="RAG Research Papers Chatbot", css=CSS) as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center;">📚 RAG Research Papers Chatbot</h1>

        This app demonstrates a Retrieval-Augmented Generation (RAG) system over a curated set of research papers about RAG itself!
        
        Ask questions about RAG to receive grounded answers generated from retrieved research paper sections. You can adjust the retriever settings in the right panel and inspect the evidence table to see exactly where each answer comes from.

        The original research papers can be found <a href="https://www.abc.com" target="_blank">here</a>.
        """
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

            # Back button (simple link)
            gr.HTML(
                """
                <a href="https://www.abc.com" target="_self" style="text-decoration:none;">
                    <button style="
                        width:100%;
                        padding:10px;
                        border-radius:6px;
                        border:1px solid #ccc;
                        background:#f5f5f5;
                        cursor:pointer;
                    ">
                        ⬅️ Back
                    </button>
                </a>
                """
            )

        with gr.Column(scale=2):
            gr.Markdown("## Retrieval & Evidence")
            retrieval_k = gr.Slider(1, 15, value=DEFAULT_CFG.k, step=1, label="Retriever k (chunks fetched)")
            search_type = gr.Dropdown(choices=["mmr", "similarity"], value=DEFAULT_CFG.search_type, label="Search type")

            gr.Markdown("### Evidence table")
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
