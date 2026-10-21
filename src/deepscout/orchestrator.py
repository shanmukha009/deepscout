"""Wire the agents into one run. BUILD_PLAN: Thread F.

planner -> (searcher -> reader) per sub-question [concurrent] -> claim-draft -> critic -> synthesizer -> guardrails
"""
from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agents.critic import CriticAgent
from .agents.planner import PlannerAgent
from .agents.reader import ReaderAgent
from .agents.searcher import SearcherAgent
from .agents.synthesizer import SynthesizerAgent
from .guardrails import check_citations
from .llm import LLM
from .models import AgentStep, Claim, Evidence, Report, RunTrace
from .tools.search import SearchTool
from .tracing import Tracer

_CLAIM_DRAFT_SYSTEM = (
    "You are a research assistant. Given a list of evidence passages, extract the key "
    "factual claims. Each claim must be a standalone factual statement directly supported "
    "by one or more of the provided evidence passages. Cite the evidence source_ids.\n\n"
    "Respond with valid JSON only: "
    '[{"text": "...", "evidence_ids": ["source-id-1", ...]}, ...]'
)


def _draft_claims(llm: LLM, tracer: Tracer, evidence: list[Evidence]) -> list[Claim]:
    """Ask the LLM to derive draft Claims from evidence before the Critic runs."""
    with tracer.step("orchestrator", "draft-claims") as step:
        if not evidence:
            return []

        evidence_text = "\n\n".join(
            f'[{e.source_id}] {e.title}: "{e.quote}"'
            for e in evidence
        )
        user = (
            f"Evidence passages:\n{evidence_text}\n\n"
            "Extract the key factual claims from this evidence. "
            "Each claim must cite at least one evidence source_id."
        )

        claims: list[Claim] = []
        try:
            data, usage = llm.complete_json(_CLAIM_DRAFT_SYSTEM, user, max_tokens=1024)
            items = data if isinstance(data, list) else []
            for item in items:
                if isinstance(item, dict) and item.get("text"):
                    claims.append(Claim(
                        text=str(item["text"]),
                        evidence_ids=[str(eid) for eid in item.get("evidence_ids", [])],
                    ))
            step.input_tokens = usage.input_tokens
            step.output_tokens = usage.output_tokens
            step.cost_usd = usage.cost_usd
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, Exception):
            pass

        if not claims:
            # Fallback: one claim per evidence piece
            claims = [
                Claim(text=e.quote[:200], evidence_ids=[e.source_id])
                for e in evidence
            ]

        return claims


def run_research(
    question: str,
    step_callback: Callable[[AgentStep], None] | None = None,
) -> tuple[Report, RunTrace]:
    llm = LLM()
    tracer = Tracer(question, step_callback=step_callback)
    search = SearchTool()

    planner = PlannerAgent(llm, tracer)
    searcher = SearcherAgent(llm, tracer, search)
    reader = ReaderAgent(llm, tracer)
    critic = CriticAgent(llm, tracer)
    synth = SynthesizerAgent(llm, tracer)

    plan = planner.run(question)

    # Run each sub-question's search+read concurrently
    all_evidence: list[Evidence] = []

    def _process_sub(sub):
        results = searcher.run(sub)
        return reader.run(sub, results)

    workers = max(1, min(len(plan.sub_questions), 5))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_sub, sub) for sub in plan.sub_questions]
        for future in as_completed(futures):
            try:
                all_evidence.extend(future.result())
            except Exception:
                # One sub-question failing (e.g. a search rate limit) must not
                # abort the whole run; proceed with the evidence we have.
                continue

    # LLM-derived claim drafting before the Critic runs
    draft_claims = _draft_claims(llm, tracer, all_evidence)

    verified = critic.run(draft_claims, all_evidence)

    report = synth.run(question, verified, all_evidence)
    report = check_citations(report)
    return report, tracer.trace


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval-augmented generation?"
    rep, tr = run_research(q)
    print(rep.to_markdown())
    print(f"\n--- cost ${tr.total_cost_usd} | latency {tr.total_latency_ms} ms | steps {len(tr.steps)}")
