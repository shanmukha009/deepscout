"""Planner - decompose the question into sub-questions. BUILD_PLAN: Thread B."""
from __future__ import annotations

import json

from ..config import settings
from ..models import ResearchPlan, SubQuestion
from .base import BaseAgent

_SYSTEM = (
    "You are a research planning assistant. Given a research question, decompose it "
    "into focused, non-overlapping sub-questions that together fully cover the answer. "
    "Return ONLY a JSON array. Each element must have exactly two string fields: "
    "'question' and 'rationale'."
)


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, question: str) -> ResearchPlan:
        with self.tracer.step(self.name, "decompose") as step:
            n = settings.max_sub_questions
            user = (
                f"Research question: {question}\n\n"
                f"Produce between 3 and {n} sub-questions as a JSON array.\n"
                'Each item: {"question": "...", "rationale": "..."}.'
            )
            usage = None
            subs: list[SubQuestion] = []
            try:
                data, usage = self.llm.complete_json(_SYSTEM, user, max_tokens=1024)
                items: list = data if isinstance(data, list) else data.get("sub_questions", [])
                subs = [
                    SubQuestion(
                        id=f"sq{i}",
                        question=item["question"],
                        rationale=item.get("rationale", ""),
                    )
                    for i, item in enumerate(items, 1)
                    if isinstance(item, dict) and item.get("question")
                ]
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                pass

            if not subs:
                subs = [SubQuestion(id="sq1", question=question, rationale="fallback")]

            if usage is not None:
                step.input_tokens = usage.input_tokens
                step.output_tokens = usage.output_tokens
                step.cost_usd = usage.cost_usd

            return ResearchPlan(question=question, sub_questions=subs)
