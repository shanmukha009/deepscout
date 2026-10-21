"""The contract every agent implements. Keep this stable."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..llm import LLM
from ..tracing import Tracer


class BaseAgent(ABC):
    name: str = "agent"

    def __init__(self, llm: LLM, tracer: Tracer):
        self.llm = llm
        self.tracer = tracer

    @abstractmethod
    def run(self, *args, **kwargs):
        """Each agent does one job and returns a typed object from models.py."""
        raise NotImplementedError
