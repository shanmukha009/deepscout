"""Tests for the naive single-shot baseline in mock mode."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("DS_PROVIDER", "mock")
os.environ.setdefault("DS_MODEL", "mock-model")
os.environ.setdefault("DS_SEARCH", "mock")

from deepscout.baseline import run_baseline
from deepscout.models import Report, RunTrace

_QUESTION = "What is the speed of light?"


def test_baseline_returns_report_and_trace():
    report, trace = run_baseline(_QUESTION)
    assert isinstance(report, Report)
    assert isinstance(trace, RunTrace)


def test_baseline_report_question_matches():
    report, _ = run_baseline(_QUESTION)
    assert report.question == _QUESTION


def test_baseline_report_has_summary():
    report, _ = run_baseline(_QUESTION)
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0


def test_baseline_report_has_one_claim():
    report, _ = run_baseline(_QUESTION)
    assert len(report.claims) == 1


def test_baseline_claim_has_no_evidence_ids():
    """Faithfulness must be 0 by design — baseline makes no cited claims."""
    report, _ = run_baseline(_QUESTION)
    assert report.claims[0].evidence_ids == []


def test_baseline_has_no_sources():
    report, _ = run_baseline(_QUESTION)
    assert report.sources == []


def test_baseline_trace_has_single_step():
    _, trace = run_baseline(_QUESTION)
    assert len(trace.steps) == 1
    assert trace.steps[0].agent == "baseline"
    assert trace.steps[0].action == "single-shot"


def test_baseline_trace_cost_and_latency_non_negative():
    _, trace = run_baseline(_QUESTION)
    assert trace.total_cost_usd >= 0
    assert trace.total_latency_ms >= 0
