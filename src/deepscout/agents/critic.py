"""Critic - verify each claim against evidence. BUILD_PLAN: Thread D."""
from __future__ import annotations

import json

from ..models import Claim, Evidence
from .base import BaseAgent

_SYSTEM = (
    "You are a rigorous fact-checker. "
    "Given a claim and one or more evidence passages, decide whether the evidence "
    "directly and explicitly supports the claim. "
    'Respond with valid JSON only: {"supported": true|false, "note": "<one-sentence reason>"}'
)


class CriticAgent(BaseAgent):
    name = "critic"

    def run(self, claims: list[Claim], evidence: list[Evidence]) -> list[Claim]:
        evidence_map = {e.source_id: e for e in evidence}

        with self.tracer.step(self.name, "verify") as step:
            total_input = total_output = 0
            total_cost = 0.0

            for c in claims:
                cited = [evidence_map[eid] for eid in c.evidence_ids if eid in evidence_map]

                if not cited:
                    c.supported = False
                    c.note = "no matching evidence found for cited ids"
                    continue

                evidence_text = "\n\n".join(
                    f"[{e.source_id}] {e.title}\n{e.quote}" for e in cited
                )
                user = f"Claim: {c.text}\n\nEvidence:\n{evidence_text}"

                try:
                    result, usage = self.llm.complete_json(_SYSTEM, user, max_tokens=256)
                    c.supported = bool(result.get("supported", False))
                    c.note = str(result.get("note", ""))
                    total_input += usage.input_tokens
                    total_output += usage.output_tokens
                    total_cost += usage.cost_usd
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError, Exception):
                    c.supported = False
                    c.note = "critic LLM call failed"

            step.input_tokens = total_input
            step.output_tokens = total_output
            step.cost_usd = total_cost

        return claims
