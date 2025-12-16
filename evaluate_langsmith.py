
"""
LangSmith batch evaluation script for your multi-PDF RAG chatbot.

Guarantees:
- Evaluation script NEVER builds/re-embeds the vector store.
- It only loads the persisted Chroma index from --persist_dir (default: chroma_db).
- It logs full traces to LangSmith, including retrieved chunks + context sent to the LLM.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate

from rag_pipeline import (
    IndexConfig,
    load_vectorstore,
    make_retriever,
    make_qa_chain,
    format_docs_for_llm,
    docs_to_citation_rows,
)


def upsert_langsmith_dataset_from_csv(client: Client, dataset_name: str, csv_path: str) -> str:
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
        dataset_id = ds.id
    except Exception:
        ds = client.create_dataset(dataset_name=dataset_name, description="RAG evaluation set (questions + gold answers).")
        dataset_id = ds.id

    df = pd.read_csv(csv_path)
    required = {"question", "answer", "source_papers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Expected: {sorted(required)}")

    existing_qs = set()
    try:
        for ex in client.list_examples(dataset_id=dataset_id):
            q = (ex.inputs or {}).get("question")
            if q:
                existing_qs.add(q.strip())
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
                    "reference_answer": str(row["answer"]).strip(),
                    "source_papers": str(row["source_papers"]).strip(),
                },
            }
        )

    if new_examples:
        client.create_examples(dataset_id=dataset_id, examples=new_examples)

    return dataset_id


def parse_expected_sources(source_papers: str) -> List[str]:
    return [p.strip() for p in str(source_papers).split(";") if p.strip()]


def retrieval_hit(retrieved: List[Dict[str, Any]], expected_sources: List[str]) -> float:
    if not expected_sources:
        return 0.0
    retrieved_files = " | ".join([r.get("file", "") for r in retrieved]).lower()
    for token in expected_sources:
        t = token.lower()
        if t and t in retrieved_files:
            return 1.0
    return 0.0


def llm_judge_score(judge, rubric_prompt: str, **kwargs) -> Tuple[float, str]:
    msg = rubric_prompt.format(**kwargs)
    resp = judge.invoke(msg)
    text = resp.content if hasattr(resp, "content") else str(resp)

    import re, json as _json
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return 0.0, text[:500]
    try:
        obj = _json.loads(m.group(0))
        return float(obj.get("score", 0)), str(obj.get("rationale", ""))[:800]
    except Exception:
        return 0.0, text[:500]


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


def make_predictor(retriever, qa_chain, judge_model_name: str, judge_temperature: float = 0.0):
    judge = ChatOpenAI(model=judge_model_name, temperature=judge_temperature)

    def predictor(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        docs = retriever.invoke(question)
        context = format_docs_for_llm(docs)
        answer = qa_chain.invoke({"question": question, "context": context})
        retrieved_rows = docs_to_citation_rows(docs)
        return {"prediction": answer, "context": context, "retrieved_chunks": retrieved_rows}

    predictor._judge = judge  # type: ignore[attr-defined]
    return predictor


def make_evaluators(predictor):
    judge = predictor._judge  # type: ignore[attr-defined]

    def evaluator_retrieval_hit(run, example):
        expected = parse_expected_sources((example.outputs or {}).get("source_papers", ""))
        retrieved = (run.outputs or {}).get("retrieved_chunks", []) or []
        return {"key": "retrieval_hit", "score": retrieval_hit(retrieved, expected)}

    def evaluator_correctness(run, example):
        question = (example.inputs or {}).get("question", "")
        ref = (example.outputs or {}).get("reference_answer", "")
        pred = (run.outputs or {}).get("prediction", "")
        score, rationale = llm_judge_score(judge, CORRECTNESS_RUBRIC, question=question, reference_answer=ref, prediction=pred)
        return {"key": "correctness_0_10", "score": score, "comment": rationale}

    def evaluator_groundedness(run, example):
        question = (example.inputs or {}).get("question", "")
        pred = (run.outputs or {}).get("prediction", "")
        ctx = (run.outputs or {}).get("context", "")
        score, rationale = llm_judge_score(judge, GROUNDEDNESS_RUBRIC, question=question, context=ctx[:12000], prediction=pred)
        return {"key": "groundedness_0_10", "score": score, "comment": rationale}

    return [evaluator_retrieval_hit, evaluator_correctness, evaluator_groundedness]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", default="rag_eval_questions_20.csv")
    parser.add_argument("--pdf_dir", default="research-papers")
    parser.add_argument("--persist_dir", default="chroma_db")
    parser.add_argument("--dataset_name", default="rag-papers-eval-20")
    parser.add_argument("--project_name", default=os.getenv("LANGSMITH_PROJECT", "rag-papers-eval"))
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv()
    client = Client()

    dataset_id = upsert_langsmith_dataset_from_csv(client, args.dataset_name, args.eval_csv)
    print(f"LangSmith dataset ready: {args.dataset_name} ({dataset_id})")

    cfg = IndexConfig(
        pdf_dir=args.pdf_dir,
        embedding_model="text-embedding-3-large",
        chunk_size=1000,
        chunk_overlap=200,
        search_type="mmr",
        k=8,
        fetch_k=20,
        lambda_mult=0.5,
    )

    # IMPORTANT: eval only loads; will error if missing
    vectorstore = load_vectorstore(cfg, args.persist_dir)
    print(f"Loaded persisted vectorstore from '{args.persist_dir}'.")

    retriever = make_retriever(vectorstore, cfg)
    qa_chain = make_qa_chain(model_name=args.llm_model, temperature=args.temperature)

    predictor = make_predictor(retriever, qa_chain, judge_model_name=args.judge_model, judge_temperature=0.0)
    evaluators = make_evaluators(predictor)

    results = evaluate(
        predictor,
        data=args.dataset_name,
        evaluators=evaluators,
        experiment_prefix=args.project_name,  # ✅ use this to name the experiment
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

    # Local export for convenience
    rows = []
    results.wait()  # ensure the experiment finished before exporting

    for r in results:
        ex = r["example"]
        run = r["run"]

        question = (ex.inputs or {}).get("question", "")
        ref = (ex.outputs or {}).get("reference_answer", "")
        source_papers = (ex.outputs or {}).get("source_papers", "")

        outs = run.outputs or {}
        pred = outs.get("prediction", "")
        retrieved = outs.get("retrieved_chunks", []) or []

        eval_results = r.get("evaluation_results", []) or []

        scores = {}
        comments = {}

        # LangSmith versions differ:
        # - some return a LIST of dicts: [{"key":..., "score":..., "comment":...}, ...]
        # - others return a DICT keyed by metric name: {"metric": {"score":..., "comment":...}, ...}
        if isinstance(eval_results, dict):
            for key, val in eval_results.items():
                if isinstance(val, dict):
                    scores[key] = val.get("score")
                    if val.get("comment"):
                        comments[f"{key}_comment"] = val.get("comment")
                else:
                    # sometimes val might already be a number/string
                    scores[key] = val

        elif isinstance(eval_results, list):
            for e in eval_results:
                if not isinstance(e, dict):
                    continue
                key = e.get("key")
                if not key:
                    continue
                scores[key] = e.get("score")
                if e.get("comment"):
                    comments[f"{key}_comment"] = e.get("comment")

        top = retrieved[:3]
        flat_top = {}
        for i, t in enumerate(top, start=1):
            flat_top[f"top{i}_file"] = t.get("file")
            flat_top[f"top{i}_page"] = t.get("page")
            flat_top[f"top{i}_snippet"] = t.get("snippet")

        rows.append(
            {
                "question": question,
                "reference_answer": ref,
                "source_papers": source_papers,
                "prediction": pred,
                **scores,
                **comments,
                **flat_top,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = "eval_results_with_sources.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved local results -> {out_path}")
    print("Open LangSmith to inspect traces (retrieved chunks + prompts + judge scores).")


if __name__ == "__main__":
    main()
