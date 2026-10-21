"""Eval harness. BUILD_PLAN: Thread G.

Scores three things per question:
  faithfulness      - fraction of claims the Critic marked supported (no uncited claims)
  coverage          - fraction of `must_mention` keywords present in the summary
  citation_validity - fraction of kept claims whose every evidence_id exists in report.sources
Run: python evals/run_evals.py
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from deepscout.orchestrator import run_research

HERE = pathlib.Path(__file__).resolve().parent


def faithfulness(report) -> float:
    if not report.claims:
        return 0.0
    return sum(1 for c in report.claims if c.supported and c.evidence_ids) / len(report.claims)


def coverage(report, keywords) -> float:
    if not keywords:
        return 1.0
    text = (report.summary + " " + report.to_markdown()).lower()
    return sum(1 for k in keywords if k.lower() in text) / len(keywords)


def citation_validity(report) -> float:
    """Fraction of kept claims where every evidence_id exists in report.sources."""
    kept = [c for c in report.claims if c.evidence_ids]
    if not kept:
        return 1.0
    valid_ids = {e.source_id for e in report.sources}
    def claim_valid(c) -> bool:
        return all(eid in valid_ids for eid in c.evidence_ids)
    return sum(1 for c in kept if claim_valid(c)) / len(kept)


def main():
    rows = [json.loads(l) for l in (HERE / "testset.jsonl").read_text().splitlines() if l.strip()]
    f_scores, c_scores, cv_scores, costs = [], [], [], []
    results = []

    col_q    = 60
    col_f    = 10
    col_c    = 10
    col_cv   = 10
    col_cost = 10

    header = (
        f"{'Question':<{col_q}}  {'faithful':>{col_f}}  {'coverage':>{col_c}}"
        f"  {'cite_valid':>{col_cv}}  {'cost_usd':>{col_cost}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in rows:
        report, trace = run_research(r["question"])
        f  = faithfulness(report)
        c  = coverage(report, r.get("must_mention", []))
        cv = citation_validity(report)
        cost = trace.total_cost_usd

        f_scores.append(f); c_scores.append(c); cv_scores.append(cv); costs.append(cost)

        q_trunc = r["question"][:col_q]
        print(
            f"{q_trunc:<{col_q}}  {f:>{col_f}.2f}  {c:>{col_c}.2f}"
            f"  {cv:>{col_cv}.2f}  {cost:>{col_cost}.4f}"
        )

        results.append({
            "question": r["question"],
            "faithfulness": round(f, 4),
            "coverage": round(c, 4),
            "citation_validity": round(cv, 4),
            "cost_usd": cost,
        })

    n = len(rows)
    mean_f    = sum(f_scores)  / n
    mean_c    = sum(c_scores)  / n
    mean_cv   = sum(cv_scores) / n
    mean_cost = sum(costs)     / n

    print(sep)
    print(
        f"{'MEAN':<{col_q}}  {mean_f:>{col_f}.2f}  {mean_c:>{col_c}.2f}"
        f"  {mean_cv:>{col_cv}.2f}  {mean_cost:>{col_cost}.4f}"
    )

    summary = {
        "n": n,
        "mean_faithfulness": round(mean_f, 4),
        "mean_coverage": round(mean_c, 4),
        "mean_citation_validity": round(mean_cv, 4),
        "mean_cost_usd": round(mean_cost, 6),
        "per_question": results,
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
