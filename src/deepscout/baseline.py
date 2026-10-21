"""Naive single-shot baseline: answer from model memory, no search, no verification.

run_baseline() returns the same (Report, RunTrace) types as run_research() so the
two systems are directly comparable in the benchmark harness.
"""
from __future__ import annotations

from .llm import LLM
from .models import Claim, Report, RunTrace
from .tracing import Tracer

_SYSTEM = (
    "You are a knowledgeable assistant. Answer the question as accurately and concisely "
    "as possible using only your training knowledge — do not search the web. "
    "If you are not confident in the answer, say so explicitly."
)


def run_baseline(question: str) -> tuple[Report, RunTrace]:
    """Answer with a single LLM call — no search, no verification.

    Claims carry no evidence_ids, so faithfulness = 0.0 by design.
    Cost and latency are recorded on the trace for fair comparison.
    """
    llm = LLM()
    tracer = Tracer(question)

    with tracer.step("baseline", "single-shot") as step:
        text, usage = llm.complete(_SYSTEM, question, max_tokens=512)
        step.input_tokens = usage.input_tokens
        step.output_tokens = usage.output_tokens
        step.cost_usd = usage.cost_usd

    # One claim per answer, no citations — the baseline makes no verifiable claims.
    claim = Claim(
        text=text.strip(),
        evidence_ids=[],  # intentionally empty: no search was done
        supported=None,   # Critic never ran
    )

    report = Report(
        question=question,
        summary=text.strip(),
        claims=[claim],
        sources=[],
    )
    return report, tracer.trace
