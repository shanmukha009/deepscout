"""Shared data contracts. DO NOT change field names without updating BUILD_PLAN.

Every agent reads and writes these types only. This is the lock that lets five
agents be built in parallel on separate branches.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str
    question: str
    rationale: str = ""


class ResearchPlan(BaseModel):
    question: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class Evidence(BaseModel):
    """A single quote pulled verbatim from a source, tied to a sub-question."""
    source_id: str
    url: str
    title: str
    quote: str  # exact supporting text copied from the source
    sub_question_id: str


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    supported: Optional[bool] = None  # set by the Critic agent
    note: str = ""


class Report(BaseModel):
    question: str
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Evidence] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.question}", "", self.summary, "", "## Findings"]
        for i, c in enumerate(self.claims, 1):
            flag = "" if c.supported else "  _(unsupported - flagged by Critic)_"
            cites = ", ".join(c.evidence_ids) if c.evidence_ids else "no citation"
            lines.append(f"{i}. {c.text} [{cites}]{flag}")
        lines += ["", "## Sources"]
        for e in self.sources:
            lines.append(f"- `{e.source_id}` [{e.title}]({e.url})")
        if self.unanswered:
            lines += ["", "## Still open", *[f"- {q}" for q in self.unanswered]]
        return "\n".join(lines)


class AgentStep(BaseModel):
    agent: str
    action: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detail: str = ""


class RunTrace(BaseModel):
    question: str
    steps: list[AgentStep] = Field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.steps), 6)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(s.latency_ms for s in self.steps), 2)
