import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from deepscout.models import Claim, Evidence, Report


def test_report_markdown_runs():
    r = Report(
        question="q",
        summary="s",
        claims=[Claim(text="c", evidence_ids=["e1"], supported=True)],
        sources=[Evidence(source_id="e1", url="u", title="t", quote="q", sub_question_id="sq1")],
    )
    md = r.to_markdown()
    assert "Findings" in md and "Sources" in md


def test_trace_totals():
    from deepscout.models import AgentStep, RunTrace
    t = RunTrace(question="q", steps=[AgentStep(agent="a", action="x", cost_usd=0.01, latency_ms=5)])
    assert t.total_cost_usd == 0.01 and t.total_latency_ms == 5
