"""Benchmark: DeepScout (run_research) vs naive baseline (run_baseline).

Metrics per system:
  accuracy          - fraction judged correct by an LLM-as-judge
  abstain_rate      - fraction where the system said it could not answer
  faithfulness      - fraction of claims with valid evidence citations
  hallucination_rate- fraction of answers that are incorrect and confidently stated
  mean_cost_usd     - average cost per query
  mean_latency_ms   - average end-to-end latency

Questions are evaluated concurrently (DS_BENCH_WORKERS, default 8), so 150
questions finish in minutes. Per-query latency is still measured inside each run,
so concurrency does not distort the reported numbers.

Run in mock mode (no API key required):
    PYTHONPATH=src python evals/benchmark/run_benchmark.py --limit 8

Run with a real provider:
    DS_PROVIDER=openai DS_MODEL=gpt-4o-mini DS_SEARCH=tavily \
    OPENAI_API_KEY=... SEARCH_API_KEY=... PYTHONPATH=src \
    python evals/benchmark/run_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from deepscout.baseline import run_baseline
from deepscout.llm import LLM
from deepscout.models import Report, RunTrace
from deepscout.orchestrator import run_research

HERE = pathlib.Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
RESULTS = HERE / "results.json"

_JUDGE_SYSTEM = (
    "You are a factual accuracy judge. Given a question, a ground-truth answer, and a "
    "system answer, classify the system answer as exactly one of:\n"
    '  "correct"   — the system answer contains the key facts from the ground truth\n'
    '  "incorrect" — the system answer contradicts or lacks the key facts\n'
    '  "abstained" — the system answer says it does not know or cannot answer\n'
    "Respond with JSON only: "
    '{"verdict": "correct" | "incorrect" | "abstained", "reason": "one sentence"}'
)


def judge_correctness(llm: LLM, question: str, system_answer: str, ground_truth: str) -> str:
    user = (
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"System answer: {system_answer[:800]}"
    )
    try:
        data, _ = llm.complete_json(_JUDGE_SYSTEM, user, max_tokens=256)
        verdict = str(data.get("verdict", "abstained")).lower()
        if verdict in ("correct", "incorrect", "abstained"):
            return verdict
    except Exception:
        pass
    return "abstained"


def faithfulness_score(report: Report) -> float:
    if not report.claims:
        return 0.0
    valid_ids = {e.source_id for e in report.sources}

    def has_valid_citation(c) -> bool:
        return bool(c.evidence_ids) and all(eid in valid_ids for eid in c.evidence_ids)

    return sum(1 for c in report.claims if has_valid_citation(c)) / len(report.claims)


def score_one(llm: LLM, row: dict, report: Report, trace: RunTrace) -> dict:
    verdict = judge_correctness(llm, row["question"], report.summary, row["answer"])
    return {
        "question": row["question"],
        "category": row.get("category", ""),
        "ground_truth": row["answer"],
        "correctness": verdict,
        "faithfulness": round(faithfulness_score(report), 4),
        "hallucinated": 1 if verdict == "incorrect" else 0,
        "cost_usd": trace.total_cost_usd,
        "latency_ms": trace.total_latency_ms,
    }


def evaluate_row(llm: LLM, row: dict) -> tuple[dict, dict]:
    ds_report, ds_trace = run_research(row["question"])
    bl_report, bl_trace = run_baseline(row["question"])
    return score_one(llm, row, ds_report, ds_trace), score_one(llm, row, bl_report, bl_trace)


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    verdicts = [r["correctness"] for r in results]
    return {
        "n": n,
        "accuracy": round(sum(1 for v in verdicts if v == "correct") / n, 4),
        "abstain_rate": round(sum(1 for v in verdicts if v == "abstained") / n, 4),
        "mean_faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
        "hallucination_rate": round(sum(r["hallucinated"] for r in results) / n, 4),
        "mean_cost_usd": round(sum(r["cost_usd"] for r in results) / n, 8),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 2),
        "per_question": results,
    }


def _save(ds_scores: list[dict], bl_scores: list[dict]) -> None:
    """Write a checkpoint after each question so a crash never loses progress."""
    summary = {"deepscout": aggregate(ds_scores), "baseline": aggregate(bl_scores)}
    RESULTS.write_text(json.dumps(summary, indent=2))


def print_table(ds: dict, bl: dict) -> None:
    width = 74
    print("\n" + "=" * width)
    print(f"  {'Benchmark Results  (n=' + str(ds.get('n', 0)) + ')':^{width - 4}}")
    print("=" * width)
    print(f"  {'Metric':<28}  {'DeepScout':>18}  {'Baseline':>18}")
    print("-" * width)
    for label, key, fmt in [
        ("Accuracy", "accuracy", ".4f"),
        ("Abstain Rate", "abstain_rate", ".4f"),
        ("Faithfulness", "mean_faithfulness", ".4f"),
        ("Hallucination Rate", "hallucination_rate", ".4f"),
        ("Mean Cost (USD)", "mean_cost_usd", ".6f"),
        ("Mean Latency (ms)", "mean_latency_ms", ".1f"),
    ]:
        print(f"  {label:<28}  {format(ds[key], fmt):>18}  {format(bl[key], fmt):>18}")
    print("=" * width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N questions")
    ap.add_argument("--workers", type=int, default=int(os.getenv("DS_BENCH_WORKERS", "8")))
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in DATASET.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if args.limit:
        rows = rows[: args.limit]

    llm = LLM()
    deepscout_scores: list[dict] = []
    baseline_scores: list[dict] = []
    done = failed = 0
    lock = threading.Lock()
    total = len(rows)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate_row, llm, row): row for row in rows}
        for fut in as_completed(futures):
            with lock:
                done += 1
                try:
                    ds_score, bl_score = fut.result()
                    deepscout_scores.append(ds_score)
                    baseline_scores.append(bl_score)
                    print(f"[{done:03d}/{total}] ok   {ds_score['question'][:52]}...")
                except Exception as e:
                    failed += 1
                    q = futures[fut]["question"][:48]
                    print(f"[{done:03d}/{total}] FAIL ({type(e).__name__}) {q}...")
                _save(deepscout_scores, baseline_scores)  # checkpoint

    summary = {
        "deepscout": aggregate(deepscout_scores),
        "baseline": aggregate(baseline_scores),
    }
    RESULTS.write_text(json.dumps(summary, indent=2))
    print_table(summary["deepscout"], summary["baseline"])
    if failed:
        print(f"\n{failed}/{total} question(s) failed after retries and were skipped.")
    print(f"\nScored {len(deepscout_scores)}/{total}. Results written to {RESULTS}")


if __name__ == "__main__":
    main()
