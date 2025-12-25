import os
import mlflow
from dataclasses import replace
from pathlib import Path

from rag_core.config import DEFAULT_CFG, PERSIST_DIR
from rag_core.app_runtime import require_vectorstore, build_vectorstore

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

from evaluation.evaluate import (
    upsert_langsmith_dataset_from_csv,  # if you want to reuse LangSmith dataset creation
    make_predictor,
    make_evaluators,
)

from dotenv import load_dotenv
load_dotenv()

def persist_dir_for_chunk(chunk_size: int) -> str:
    return f"{PERSIST_DIR}_cs{chunk_size}"


def run_langsmith_eval(vectorstore, cfg, dataset_id: str, llm_model: str, judge_model: str):
    predictor = make_predictor(
        vectorstore,
        cfg,
        llm_model=llm_model,
        temperature=0.0,
        judge_model=judge_model,
    )
    evaluators = make_evaluators(predictor)

    results = ls_evaluate(
        predictor,
        data=dataset_id,
        evaluators=evaluators,
        experiment_prefix=f"tune_cs{cfg.chunk_size}_k{cfg.k}",
        metadata={
            "retrieval": {"search_type": cfg.search_type, "k": cfg.k},
            "chunking": {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap},
        },
        client=Client(),
    )

    results.wait()

    # Aggregate metrics from LangSmith results
    scores = {"retrieval_hit": [], "correctness_0_10": [], "groundedness_0_10": []}

    for r in results:
        er = r.get("evaluation_results") or {}

        # Newer LangSmith shape: {"results": [EvaluationResult(...), ...]}
        if isinstance(er, dict) and isinstance(er.get("results"), list):
            items = er["results"]

        # Some older shapes: list of results
        elif isinstance(er, list):
            items = er

        # Rare: dict keyed by metric name
        elif isinstance(er, dict):
            items = []
            for kk, vv in er.items():
                # vv could be dict-like
                if isinstance(vv, dict) and "score" in vv:
                    items.append({"key": kk, "score": vv.get("score")})
                else:
                    items.append(vv)
        else:
            items = []

        for item in items:
            # --- handle EvaluationResult objects ---
            if hasattr(item, "key") and hasattr(item, "score"):
                k = item.key
                s = item.score
            # --- handle dicts ---
            elif isinstance(item, dict):
                k = item.get("key")
                s = item.get("score")
            else:
                continue

            if k in scores and s is not None:
                scores[k].append(float(s))

    out = {k: (sum(v) / len(v) if v else None) for k, v in scores.items()}
    return out


def main():
    mlflow.set_experiment("rag_hparam_tuning")

    # Your tuning grid
    #chunk_sizes = [800, 1000, 1200]
    #ks = [4, 6, 8, 10]
    chunk_sizes = [800]
    ks = [4, 6]

    # LangSmith dataset (reuse your CSV)
    eval_csv: str = str(
        Path(__file__).resolve().parent.parent / "evaluation/rag_eval_questions.csv"
    )
    dataset_name = "rag-papers-eval-hyper"
    client = Client()
    dataset_id = upsert_langsmith_dataset_from_csv(client, dataset_name, eval_csv)

    llm_model = "gpt-4o-mini"
    judge_model = "gpt-4o-mini"

    for cs in chunk_sizes:
        # cfg for this chunk size
        base_cfg = replace(DEFAULT_CFG, chunk_size=cs)

        # build/load index for this cs
        persist_dir = persist_dir_for_chunk(cs)
        if os.path.exists(persist_dir):
            vectorstore = require_vectorstore(base_cfg, persist_dir)
        else:
            vectorstore, _pdf_paths = build_vectorstore(base_cfg, persist_dir)

        for k in ks:
            cfg = replace(base_cfg, k=k, search_type="mmr")

            with mlflow.start_run(run_name=f"cs={cs}_k={k}"):
                mlflow.log_params({
                    "chunk_size": cs,
                    "chunk_overlap": cfg.chunk_overlap,
                    "search_type": cfg.search_type,
                    "k": k,
                    "persist_dir": persist_dir,
                    "llm_model": llm_model,
                    "judge_model": judge_model,
                })

                metrics = run_langsmith_eval(vectorstore, cfg, dataset_id, llm_model, judge_model)
                print("DEBUG metrics to log:", metrics)

                for mk, mv in metrics.items():
                    if mv is not None:
                        mlflow.log_metric(mk, mv)

                print(f"[cs={cs}, k={k}] metrics:", metrics)


if __name__ == "__main__":
    main()
