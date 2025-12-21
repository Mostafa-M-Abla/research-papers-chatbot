"""
LangSmith batch evaluation script for the RAG Research Papers chatbot.

What this script does:
- LOAD-ONLY: it NEVER builds / re-embeds your vector store.
  (Build your index explicitly first with: python build_index.py)
- Creates (or reuses) a LangSmith dataset from a CSV evaluation set.
- Runs batch evaluation with:
    1) retrieval_hit (deterministic; robust to punctuation/special chars)
    2) correctness_0_10 (LLM-as-judge vs reference answer)
    3) groundedness_0_10 (LLM-as-judge vs retrieved context)
- Logs full traces to LangSmith, including retrieved chunks + context sent to the LLM.

Expected CSV columns:
  - question
  - answer
  - source_papers   (semi-colon separated list of expected paper tokens)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from config import DEFAULT_CFG, PERSIST_DIR, PDF_DIR
from app_runtime import require_vectorstore, make_runtime_retriever, make_qa
from rag_pipeline import format_docs_for_llm, docs_to_citation_rows


# -----------------------------
# Dataset utilities
# -----------------------------
def upsert_langsmith_dataset_from_csv(client: Client, dataset_name: str, csv_path: str) -> str:
    """Create or reuse a LangSmith dataset from a CSV file and return the dataset_id."""
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
        dataset_id = ds.id
    except Exception:
        ds = client.create_dataset(
            dataset_name=dataset_name,
            description="RAG evaluation set (questions + gold answers).",
        )
        dataset_id = ds.id

    df = pd.read_csv(csv_path)
    required = {"question", "answer", "source_papers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Expected: {sorted(required)}")

    # Avoid duplicating questions already in the dataset
    existing_qs = set()
    try:
        for ex in client.list_examples(dataset_id=dataset_id):
            q = (ex.inputs or {}).get("question")
            if q:
                existing_qs.add(str(q).strip())
    except Exception:
        pass

    new_examples = []
    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        if not q or q in existing_qs:
            continue
        new_examples.append(
            {
                "inputs": {"question": q},
                "outputs": {
                    # Keep names stable for evaluators
                    "reference_answer": str(row["answer"]).strip(),
                    "source_papers": str(row["source_papers"]).strip(),
                },
            }
        )

    if new_examples:
        client.create_examples(dataset_id=dataset_id, examples=new_examples)

    return dataset_id


# -----------------------------
# Retrieval-hit utilities
# -----------------------------
def parse_expected_sources(source_papers: str) -> List[str]:
    """
    Parse expected source papers.
    Supports separators: ; , |
    """
    if not source_papers:
        return []

    s = source_papers
    for sep in [",", "|"]:
        s = s.replace(sep, ";")

    return [p.strip() for p in s.split(";") if p.strip()]


def retrieval_hit(retrieved: List[Dict[str, Any]], expected_sources: List[str]) -> float:
    """Return 1.0 if any expected source token matches retrieved file names (punctuation-insensitive)."""
    import re

    def normalize(s: str) -> str:
        # Lowercase and keep only letters/digits (drops punctuation, spaces, unicode dash variants, etc.)
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    if not expected_sources:
        return 0.0

    retrieved_files_norm = normalize(" | ".join([str(r.get("file", "")) for r in retrieved if isinstance(r, dict)]))

    for token in expected_sources:
        t_norm = normalize(token)
        if t_norm and t_norm in retrieved_files_norm:
            return 1.0

    return 0.0


# -----------------------------
# LLM-as-judge utilities
# -----------------------------
def llm_judge_score(judge, rubric_prompt: str, **kwargs) -> Tuple[float, str]:
    """Invoke judge model with rubric prompt and parse JSON {score, rationale}."""
    msg = rubric_prompt.format(**kwargs)
    resp = judge.invoke(msg)
    text = resp.content if hasattr(resp, "content") else str(resp)

    import json as _json
    import re

    # Extract first JSON object in the output
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return 0.0, text[:500]

    try:
        obj = _json.loads(m.group(0))
        return float(obj.get("score", 0.0)), str(obj.get("rationale", ""))[:800]
    except Exception:
        return 0.0, text[:500]


# IMPORTANT: braces are doubled so .format(...) doesn't treat them as placeholders
CORRECTNESS_RUBRIC = r"""
You are grading a RAG system answer against a reference answer.
Return ONLY valid JSON like: {{"score": 0-10, "rationale": "<short>"}}

Scoring:
- 10: fully correct and complete vs reference
- 7-9: mostly correct, minor omissions/wording differences
- 4-6: partially correct, misses key points or includes some wrong claims
- 1-3: mostly incorrect
- 0: refuses/empty/unrelated

Question: {question}
Reference answer: {reference_answer}
System answer: {prediction}
"""

GROUNDEDNESS_RUBRIC = r"""
You are grading whether the system answer is supported by the retrieved context.
Return ONLY valid JSON like: {{"score": 0-10, "rationale": "<short>"}}

Scoring:
- 10: every factual claim is supported by the context
- 7-9: mostly supported; a few small unsupported details
- 4-6: mixed; several unsupported or speculative claims
- 1-3: mostly unsupported/hallucinated
- 0: no context used / completely ungrounded

Question: {question}
Retrieved context:
{context}
System answer: {prediction}
"""


# -----------------------------
# Predictor (system under test)
# -----------------------------
def make_predictor(vectorstore, base_cfg, *, llm_model: str, temperature: float, judge_model: str):
    """Create predictor callable returning prediction + context + retrieved chunk metadata."""
    qa_chain = make_qa(model_name=llm_model, temperature=temperature)
    judge = ChatOpenAI(model=judge_model, temperature=0.0)

    def predictor(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]

        # Create retriever per-run from the persisted store (allows swapping cfg later if desired)
        retriever = make_runtime_retriever(vectorstore, base_cfg, k=base_cfg.k, search_type=base_cfg.search_type)
        docs = retriever.invoke(question)

        context = format_docs_for_llm(docs)
        answer = qa_chain.invoke({"question": question, "context": context})

        retrieved_rows = docs_to_citation_rows(docs, max_chars=260)
        return {
            "prediction": answer,
            "context": context,
            "retrieved_chunks": retrieved_rows,
        }

    predictor._judge = judge  # type: ignore[attr-defined]
    return predictor


# -----------------------------
# Evaluators
# -----------------------------
def make_evaluators(predictor):
    judge = predictor._judge  # type: ignore[attr-defined]

    def evaluator_retrieval_hit(run: Run, example: Example) -> Dict[str, Any]:
        """
        Checks whether at least one retrieved chunk comes from
        an expected source paper.
        Supports source_papers stored in inputs or outputs.
        """

        # 1. Read source_papers from outputs OR inputs
        source_papers = ""
        if example.outputs and example.outputs.get("source_papers"):
            source_papers = example.outputs["source_papers"]
        elif example.inputs and example.inputs.get("source_papers"):
            source_papers = example.inputs["source_papers"]

        expected_sources = parse_expected_sources(source_papers)

        retrieved_chunks = (run.outputs or {}).get("retrieved_chunks", [])

        return {
            "key": "retrieval_hit",
            "score": retrieval_hit(retrieved_chunks, expected_sources),
        }

    def evaluator_correctness(run: Run, example: Example) -> Dict[str, Any]:
        question = (example.inputs or {}).get("question", "")
        ref = (example.outputs or {}).get("reference_answer", "")
        pred = (run.outputs or {}).get("prediction", "")
        score, rationale = llm_judge_score(
            judge,
            CORRECTNESS_RUBRIC,
            question=question,
            reference_answer=ref,
            prediction=pred,
        )
        return {"key": "correctness_0_10", "score": score, "comment": rationale}

    def evaluator_groundedness(run: Run, example: Example) -> Dict[str, Any]:
        question = (example.inputs or {}).get("question", "")
        pred = (run.outputs or {}).get("prediction", "")
        ctx = (run.outputs or {}).get("context", "")
        score, rationale = llm_judge_score(
            judge,
            GROUNDEDNESS_RUBRIC,
            question=question,
            context=str(ctx)[:12000],
            prediction=pred,
        )
        return {"key": "groundedness_0_10", "score": score, "comment": rationale}

    return [evaluator_retrieval_hit, evaluator_correctness, evaluator_groundedness]


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate the RAG system using LangSmith (load-only).")
    parser.add_argument("--eval_csv", default="rag_eval_questions.csv")
    parser.add_argument("--dataset_name", default="rag-papers-eval-20")
    parser.add_argument("--project_name", default=os.getenv("LANGSMITH_PROJECT", "rag-papers-eval"))
    parser.add_argument("--pdf_dir", default=PDF_DIR)
    parser.add_argument("--persist_dir", default=PERSIST_DIR)

    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("Missing LANGSMITH_API_KEY. Set it in your environment or .env file.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in your environment or .env file.")

    client = Client()

    dataset_id = upsert_langsmith_dataset_from_csv(client, args.dataset_name, args.eval_csv)
    print(f"LangSmith dataset ready: {args.dataset_name} ({dataset_id})")

    # Shared config + allow overriding only the pdf_dir path
    cfg = replace(DEFAULT_CFG, pdf_dir=args.pdf_dir)

    # LOAD-ONLY: requires the persisted index to exist
    vectorstore = require_vectorstore(cfg, args.persist_dir)
    print(f"Loaded persisted vectorstore from '{args.persist_dir}'.")

    predictor = make_predictor(
        vectorstore,
        cfg,
        llm_model=args.llm_model,
        temperature=args.temperature,
        judge_model=args.judge_model,
    )
    evaluators = make_evaluators(predictor)

    results = evaluate(
        predictor,
        data=dataset_id,
        evaluators=evaluators,
        experiment_prefix=args.project_name,
        metadata={
            "llm_model": args.llm_model,
            "judge_model": args.judge_model,
            "persist_dir": args.persist_dir,
            "pdf_dir": args.pdf_dir,
            "retrieval": {
                "search_type": cfg.search_type,
                "k": cfg.k,
                "fetch_k": cfg.fetch_k,
                "lambda_mult": cfg.lambda_mult,
            },
            "chunking": {
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
            },
        },
        client=client,
    )

    print("✅ Evaluation submitted to LangSmith.")
    print("Experiment:", getattr(results, "experiment_name", args.project_name))
    if getattr(results, "url", None):
        print("URL:", results.url)


if __name__ == "__main__":
    main()
