"""Lightweight observability: record each agent step with cost + latency.

Swap the sink for LangSmith / Arize Phoenix later; the interface stays the same.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager

from .models import AgentStep, RunTrace

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("deepscout")


class Tracer:
    def __init__(self, question: str, step_callback=None):
        self.trace = RunTrace(question=question)
        self._step_callback = step_callback

    @contextmanager
    def step(self, agent: str, action: str):
        step = AgentStep(agent=agent, action=action)
        t0 = time.perf_counter()
        try:
            yield step
        finally:
            step.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.trace.steps.append(step)
            if self._step_callback:
                self._step_callback(step)
            log.info(json.dumps({
                "agent": step.agent, "action": step.action,
                "cost_usd": step.cost_usd, "latency_ms": step.latency_ms,
            }))
