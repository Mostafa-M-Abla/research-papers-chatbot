"""
Gradio UI (load-only) for the RAG Research Papers chatbot.

Important:
- This UI NEVER builds the vectorstore.
- Build the index first:  python build_index.py
"""
from __future__ import annotations

from gradio import themes
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

from config import DEFAULT_CFG, PERSIST_DIR
from app_runtime import require_vectorstore, make_runtime_retriever, make_qa
from rag_pipeline import format_docs_for_llm, docs_to_citation_rows

# -----------------------------
# Example questions (overlay)
# -----------------------------
EXAMPLE_QUESTIONS = [
    "What is Graph RAG, and where is it mostly used?",
    "How does LightRAG differ from Open RAG?",
    "Tell me about two trending topics in the field of RAG.",
    "How does the paper 'CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation' solve the over-reliance on semantic similarity for retrieval?",
]


def _rows_to_table(rows):
    headers = ["rank", "file", "page", "snippet"]
    return [[r.get(h, "") for h in headers] for r in rows]


def answer_question(message: str, history: list, retrieval_k: int, search_type: str):
    """Streams the assistant response.

    Returns 4 outputs:
      (chat_history, evidence_table, textbox_value, examples_overlay_visibility_update)
    """
    message = (message or "").strip()
    if not message:
        # keep output shape the same
        return history, [], "", gr.update()

    retriever = make_runtime_retriever(vectorstore, DEFAULT_CFG, k=int(retrieval_k), search_type=search_type)
    docs = retriever.invoke(message)

    ctx = format_docs_for_llm(docs)

    rows = docs_to_citation_rows(docs, max_chars=260)
    evidence = _rows_to_table(rows)

    history = history or []
    history = history + [{"role": "user", "content": message}]
    # Add placeholder assistant message we will update while streaming
    history = history + [{"role": "assistant", "content": ""}]

    partial = ""
    # Stream answer token-by-token (or chunk-by-chunk)
    for chunk in qa_chain.stream({"question": message, "context": ctx}):
        partial += chunk
        history[-1]["content"] = partial
        # Clear textbox immediately; hide examples once user starts chatting
        yield history, evidence, "", gr.update(visible=False)

    # final flush
    yield history, evidence, "", gr.update(visible=False)


def clear_all():
    # show examples again after clearing
    return [], [], "", gr.update(visible=True)


# Load-only: fail fast with a clear message if index missing
vectorstore = require_vectorstore(DEFAULT_CFG, PERSIST_DIR)
qa_chain = make_qa(model_name="gpt-4o-mini", temperature=0.0)

# Older Gradio versions accept css in Blocks; keep it here for broad compatibility.
CSS = r"""
footer {display:none !important;}
.footer {display:none !important;}
#footer {display:none !important;}
.gradio-footer {display:none !important;}

/* Make the chat area a positioning context for the overlay */
#chat_container { position: relative; }

/* Overlay that sits inside the empty chat area */
#examples_overlay {
  position: absolute;
  inset: 52px 18px 18px 18px; /* leave room for Chatbot header */
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none; /* allow the chat to be selectable; buttons will override */
}

/* Inner panel */
#examples_overlay .examples_panel{
  width: min(900px, 96%);
  pointer-events: auto;
}

/* Grid layout similar to the screenshot */
#examples_overlay .examples_grid{
  display: grid;
  grid-template-columns: repeat(2, minmax(260px, 1fr));
  gap: 16px;
  margin-top: 12px;
}

/* Make Gradio buttons look like "cards" */
.example_card button{
  background: #f0f4f8 !important;
  border: 2px solid #d1dbe6 !important;
  border-radius: 8px !important;
  padding: 12px 16px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
  font-size: 14px !important;
  text-align: left !important;
}

/* Hover */
.example_card button:hover{
  background: #e3edf7 !important;
  border-color: #6b9bd1 !important;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
  transform: translateY(-2px) !important;
}

.example_card button:active {
  transform: translateY(0) !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}
      
/* Small title */
#examples_overlay .examples_title{
  font-weight: 600;
  opacity: 0.9;
}

/* Responsive: 1 column on narrow screens */
@media (max-width: 720px){
  #examples_overlay .examples_grid{ grid-template-columns: 1fr; }
}
"""


with gr.Blocks(
    title="RAG Research Papers Chatbot",
    theme=themes.Ocean(primary_hue="blue", secondary_hue="slate"),
    css=CSS,
) as demo:
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
            # Chat + overlay live in the same positioned container
            with gr.Group(elem_id="chat_container"):
                chat = gr.Chatbot(label="Chat", height=520)

                # Overlay: clickable example "cards" inside the empty chat area
                with gr.Column(visible=True, elem_id="examples_overlay") as examples_overlay:
                    with gr.Column(elem_classes=["examples_panel"]):
                        gr.Markdown("Try an example", elem_classes=["examples_title"])
                        with gr.Row(elem_classes=["examples_grid"]):
                            ex_btns = []
                            for q in EXAMPLE_QUESTIONS:
                                ex_btns.append(gr.Button(q, elem_classes=["example_card"]))

            msg = gr.Textbox(
                label="Your question",
                placeholder="e.g., What is GraphRAG? How does LightRAG differ from OpenRAG?",
                lines=1,
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
            gr.Markdown("## Retriever Stettings")
            retrieval_k = gr.Slider(1, 15, value=DEFAULT_CFG.k, step=1, label="Retriever k (chunks fetched)")
            search_type = gr.Dropdown(choices=["mmr", "similarity"], value=DEFAULT_CFG.search_type, label="Search type")

            gr.Markdown("## Evidence Table")
            evidence_df = gr.Dataframe(
                headers=["rank", "file", "page", "snippet"],
                datatype=["number", "str", "str", "str"],
                label="Retrieved chunks",
                wrap=True,
                interactive=False,
            )

    # Normal send / submit
    send.click(
        answer_question,
        inputs=[msg, chat, retrieval_k, search_type],
        outputs=[chat, evidence_df, msg, examples_overlay],
    )
    msg.submit(
        answer_question,
        inputs=[msg, chat, retrieval_k, search_type],
        outputs=[chat, evidence_df, msg, examples_overlay],
    )
    clear.click(clear_all, outputs=[chat, evidence_df, msg, examples_overlay])

    # Example buttons: click -> set textbox -> auto-run (and hide overlay)
    for btn, q in zip(ex_btns, EXAMPLE_QUESTIONS):
        btn.click(lambda qq=q: qq, outputs=[msg]).then(
            answer_question,
            inputs=[msg, chat, retrieval_k, search_type],
            outputs=[chat, evidence_df, msg, examples_overlay],
        )

if __name__ == "__main__":
    demo.launch()
