import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("DS_PROVIDER", "mock")
os.environ.setdefault("DS_MODEL", "mock-model")

from unittest.mock import patch

from deepscout.llm import LLM, Usage
from deepscout.tracing import Tracer
from deepscout.agents.planner import PlannerAgent
from deepscout.models import ResearchPlan, SubQuestion


_MOCK_DATA = [
    {"question": "What is RAG?", "rationale": "Define the concept"},
    {"question": "How does RAG improve accuracy?", "rationale": "Explore benefits"},
    {"question": "What frameworks implement RAG?", "rationale": "Survey tooling"},
]
_MOCK_USAGE = Usage(input_tokens=50, output_tokens=100, cost_usd=0.001, latency_ms=200.0)


def _make_agent(question: str = "What is retrieval-augmented generation?"):
    llm = LLM()
    tracer = Tracer(question)
    return PlannerAgent(llm=llm, tracer=tracer), tracer


def test_planner_returns_research_plan_with_sub_questions():
    agent, tracer = _make_agent()
    with patch.object(LLM, "complete_json", return_value=(_MOCK_DATA, _MOCK_USAGE)):
        plan = agent.run("What is retrieval-augmented generation?")

    assert isinstance(plan, ResearchPlan)
    assert len(plan.sub_questions) >= 1


def test_planner_assigns_sequential_ids():
    agent, _ = _make_agent()
    with patch.object(LLM, "complete_json", return_value=(_MOCK_DATA, _MOCK_USAGE)):
        plan = agent.run("What is retrieval-augmented generation?")

    ids = [sq.id for sq in plan.sub_questions]
    assert ids == ["sq1", "sq2", "sq3"]


def test_planner_records_usage_on_trace_step():
    agent, tracer = _make_agent()
    with patch.object(LLM, "complete_json", return_value=(_MOCK_DATA, _MOCK_USAGE)):
        agent.run("What is retrieval-augmented generation?")

    step = tracer.trace.steps[0]
    assert step.input_tokens == 50
    assert step.output_tokens == 100
    assert step.cost_usd == 0.001


def test_planner_falls_back_on_bad_json():
    """If complete_json raises JSONDecodeError, planner returns a single-item fallback."""
    import json
    agent, _ = _make_agent("fallback question")
    with patch.object(LLM, "complete_json", side_effect=json.JSONDecodeError("bad", "", 0)):
        plan = agent.run("fallback question")

    assert isinstance(plan, ResearchPlan)
    assert len(plan.sub_questions) >= 1
    assert plan.sub_questions[0].question == "fallback question"
