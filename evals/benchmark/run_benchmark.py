"""Benchmark: DeepScout (run_research) vs naive baseline (run_baseline).

Metrics per system:
  correctness       - fraction judged correct by an LLM-as-judge
  faithfulness      - fraction of claims with valid evidence citations
  hallucination_rate- fraction of answers that are incorrect and confidently stated
  mean_cost_usd     - average cost per query
  mean_latency_ms   - average end-to-end latency

Run in mock mode (no API key required):
    PYTHONPATH=src python evals/benchmark/run_benchmark.py

Run with a real provider:
    DS_PROVIDER=anthropic DS_MODEL=claude-haiku-5-5 ... python evals/benchmark/run_benchmark.py
"""
from __future__ import annotations

import json
import pathlib
import sys

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
    """Call LLM-as-judge; returns 'correct', 'incorrect', or 'abstained'.

    Degrades gracefully in mock mode: parse failure → 'abstained'.
    """
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
    """Fraction of claims that carry at least one valid evidence citation."""
    if not report.claims:
        return 0.0
    valid_ids = {e.source_id for e in report.sources}

    def has_valid_citation(c) -> bool:
        return bool(c.evidence_ids) and all(eid in valid_ids for eid in c.evidence_ids)

    return sum(1 for c in report.claims if has_valid_citation(c)) / len(report.claims)


def score_one(
    llm: LLM, row: dict, report: Report, trace: RunTrace
) -> dict:
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


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    verdicts = [r["correctness"] for r in results]
    accuracy = sum(1 for v in verdicts if v == "correct") / n
    abstain_rate = sum(1 for v in verdicts if v == "abstained") / n
    mean_faith = sum(r["faithfulness"] for r in results) / n
    hallucination_rate = sum(r["hallucinated"] for r in results) / n
    mean_cost = sum(r["cost_usd"] for r in results) / n
    mean_latency = sum(r["latency_ms"] for r in results) / n
    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "abstain_rate": round(abstain_rate, 4),
        "mean_faithfulness": round(mean_faith, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "mean_cost_usd": round(mean_cost, 8),
        "mean_latency_ms": round(mean_latency, 2),
        "per_question": results,
    }


def print_table(ds: dict, bl: dict) -> None:
    width = 74
    print("\n" + "=" * width)
    print(f"  {'Benchmark Results':^{width - 4}}")
    print("=" * width)
    print(f"  {'Metric':<28}  {'DeepScout':>18}  {'Baseline':>18}")
    print("-" * width)
    metrics = [
        ("Accuracy", "accuracy", ".4f"),
        ("Abstain Rate", "abstain_rate", ".4f"),
        ("Faithfulness", "mean_faithfulness", ".4f"),
        ("Hallucination Rate", "hallucination_rate", ".4f"),
        ("Mean Cost (USD)", "mean_cost_usd", ".6f"),
        ("Mean Latency (ms)", "mean_latency_ms", ".1f"),
    ]
    for label, key, fmt in metrics:
        ds_val = format(ds[key], fmt)
        bl_val = format(bl[key], fmt)
        print(f"  {label:<28}  {ds_val:>18}  {bl_val:>18}")
    print("=" * width)


def main() -> None:
    rows = [
        json.loads(line)
        for line in DATASET.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    llm = LLM()
    deepscout_scores: list[dict] = []
    baseline_scores: list[dict] = []

    for i, row in enumerate(rows, 1):
        q_display = row["question"][:60]
        print(f"[{i:02d}/{len(rows)}] {q_display}...")

        ds_report, ds_trace = run_research(row["question"])
        bl_report, bl_trace = run_baseline(row["question"])

        deepscout_scores.append(score_one(llm, row, ds_report, ds_trace))
        baseline_scores.append(score_one(llm, row, bl_report, bl_trace))

    summary = {
        "deepscout": aggregate(deepscout_scores),
        "baseline": aggregate(baseline_scores),
    }

    RESULTS.write_text(json.dumps(summary, indent=2))

    print_table(summary["deepscout"], summary["baseline"])
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
