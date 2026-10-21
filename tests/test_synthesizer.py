import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("DS_PROVIDER", "mock")
os.environ.setdefault("DS_MODEL", "mock-model")

from unittest.mock import patch

from deepscout.llm import LLM, Usage
from deepscout.tracing import Tracer
from deepscout.agents.synthesizer import SynthesizerAgent
from deepscout.models import Claim, Evidence, Report

_QUESTION = "What is retrieval-augmented generation?"

_EVIDENCE = [
    Evidence(
        source_id="src-1",
        url="https://example.com/rag",
        title="RAG Overview",
        quote="RAG combines retrieval with generation to ground LLM outputs.",
        sub_question_id="sq1",
    ),
    Evidence(
        source_id="src-2",
        url="https://example.com/benefits",
        title="RAG Benefits",
        quote="RAG reduces hallucinations by anchoring responses to retrieved documents.",
        sub_question_id="sq2",
    ),
]

_SUPPORTED_CLAIM = Claim(
    text="RAG combines retrieval with generation.",
    evidence_ids=["src-1"],
    supported=True,
)
_UNSUPPORTED_CLAIM = Claim(
    text="RAG is useless.",
    evidence_ids=["src-2"],
    supported=False,
)
_UNREVIEWED_CLAIM = Claim(
    text="RAG was invented in 2020.",
    evidence_ids=[],
    supported=None,
)

_MOCK_SUMMARY = "RAG grounds LLM outputs in retrieved documents [src-1], reducing hallucinations [src-2]."
_MOCK_USAGE = Usage(input_tokens=80, output_tokens=40, cost_usd=0.002, latency_ms=150.0)


def _make_agent():
    llm = LLM()
    tracer = Tracer(_QUESTION)
    return SynthesizerAgent(llm=llm, tracer=tracer), tracer


def test_report_contains_only_supported_claims():
    """Only claims with supported=True must appear in the report."""
    agent, _ = _make_agent()
    claims = [_SUPPORTED_CLAIM, _UNSUPPORTED_CLAIM, _UNREVIEWED_CLAIM]
    with patch.object(LLM, "complete", return_value=(_MOCK_SUMMARY, _MOCK_USAGE)):
        report = agent.run(_QUESTION, claims, _EVIDENCE)

    assert isinstance(report, Report)
    for claim in report.claims:
        assert claim.supported is True, f"Non-supported claim leaked into report: {claim.text}"
    assert len(report.claims) == 1
    assert report.claims[0].text == _SUPPORTED_CLAIM.text


def test_report_returns_correct_question_and_sources():
    agent, _ = _make_agent()
    with patch.object(LLM, "complete", return_value=(_MOCK_SUMMARY, _MOCK_USAGE)):
        report = agent.run(_QUESTION, [_SUPPORTED_CLAIM], _EVIDENCE)

    assert report.question == _QUESTION
    assert report.sources == _EVIDENCE
    assert report.summary == _MOCK_SUMMARY


def test_usage_recorded_on_trace_step():
    agent, tracer = _make_agent()
    with patch.object(LLM, "complete", return_value=(_MOCK_SUMMARY, _MOCK_USAGE)):
        agent.run(_QUESTION, [_SUPPORTED_CLAIM], _EVIDENCE)

    step = tracer.trace.steps[0]
    assert step.input_tokens == 80
    assert step.output_tokens == 40
    assert step.cost_usd == 0.002


def test_to_markdown_renders_without_error():
    agent, _ = _make_agent()
    with patch.object(LLM, "complete", return_value=(_MOCK_SUMMARY, _MOCK_USAGE)):
        report = agent.run(_QUESTION, [_SUPPORTED_CLAIM], _EVIDENCE)

    md = report.to_markdown()
    assert _QUESTION in md
    assert _MOCK_SUMMARY in md
    assert "src-1" in md


def test_empty_claims_produces_empty_report():
    agent, _ = _make_agent()
    with patch.object(LLM, "complete", return_value=("No findings.", _MOCK_USAGE)):
        report = agent.run(_QUESTION, [], _EVIDENCE)

    assert report.claims == []
    assert isinstance(report, Report)
