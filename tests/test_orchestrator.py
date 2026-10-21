"""End-to-end test for run_research in mock mode."""
import sys
import os
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("DS_PROVIDER", "mock")
os.environ.setdefault("DS_MODEL", "mock-model")
os.environ.setdefault("DS_SEARCH", "mock")

from deepscout.orchestrator import run_research
from deepscout.models import Report, RunTrace


_QUESTION = "What is retrieval-augmented generation?"


def test_run_research_returns_report_and_trace():
    report, trace = run_research(_QUESTION)
    assert isinstance(report, Report), "run_research must return a Report"
    assert isinstance(trace, RunTrace), "run_research must return a RunTrace"


def test_run_research_report_has_question():
    report, _ = run_research(_QUESTION)
    assert report.question == _QUESTION


def test_run_research_trace_has_at_least_five_steps():
    _, trace = run_research(_QUESTION)
    assert len(trace.steps) >= 5, (
        f"Expected >= 5 trace steps, got {len(trace.steps)}: "
        + str([f"{s.agent}/{s.action}" for s in trace.steps])
    )


def test_run_research_trace_includes_expected_agents():
    _, trace = run_research(_QUESTION)
    agents_seen = {s.agent for s in trace.steps}
    for expected in ("planner", "searcher", "reader", "orchestrator", "critic", "synthesizer"):
        assert expected in agents_seen, f"Agent '{expected}' missing from trace"


def test_run_research_step_callback_called():
    collected = []
    run_research(_QUESTION, step_callback=lambda s: collected.append(s.agent))
    assert len(collected) >= 5, f"Callback called {len(collected)} times, expected >= 5"


def test_run_research_trace_cost_and_latency_are_non_negative():
    _, trace = run_research(_QUESTION)
    assert trace.total_cost_usd >= 0
    assert trace.total_latency_ms >= 0
