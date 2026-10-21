"""Synthesizer - write the final cited report. BUILD_PLAN: Thread E."""
from __future__ import annotations

from ..models import Claim, Evidence, Report
from .base import BaseAgent

_SYSTEM = (
    "You are a research synthesis assistant. Given a research question, a list of "
    "verified claims (each tagged supported=true), and source evidence, write a "
    "concise, structured summary. Every sentence that makes a factual claim MUST "
    "cite the supporting evidence by its source_id in square brackets, e.g. [src-1]. "
    "Only cite evidence_ids that appear in the provided evidence list. "
    "Do not invent citations or make uncited assertions. "
    "Use plain prose — no markdown headers inside the summary itself."
)


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def run(self, question: str, claims: list[Claim], evidence: list[Evidence]) -> Report:
        with self.tracer.step(self.name, "write") as step:
            supported = [c for c in claims if c.supported]

            evidence_index = "\n".join(
                f"[{e.source_id}] {e.title} ({e.url}): \"{e.quote}\""
                for e in evidence
            )
            claims_text = "\n".join(
                f"- {c.text}  (evidence_ids: {', '.join(c.evidence_ids)})"
                for c in supported
            ) or "(none)"

            user = (
                f"Research question: {question}\n\n"
                f"Supported claims:\n{claims_text}\n\n"
                f"Evidence:\n{evidence_index}\n\n"
                "Write a concise synthesis (2-4 paragraphs) that covers all the "
                "supported claims and cites each by its source_id in [brackets]. "
                "Do not add claims beyond those listed above."
            )

            summary, usage = self.llm.complete(_SYSTEM, user, max_tokens=1024)

            step.input_tokens = usage.input_tokens
            step.output_tokens = usage.output_tokens
            step.cost_usd = usage.cost_usd

            return Report(
                question=question,
                summary=summary,
                claims=supported,
                sources=evidence,
            )
