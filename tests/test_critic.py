import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("DS_PROVIDER", "mock")
os.environ.setdefault("DS_MODEL", "mock-model")

from unittest.mock import patch

from deepscout.llm import LLM, Usage
from deepscout.tracing import Tracer
from deepscout.agents.critic import CriticAgent
from deepscout.models import Claim, Evidence, Report
from deepscout.guardrails import check_citations


_MOCK_USAGE = Usage(input_tokens=20, output_tokens=10, cost_usd=0.0, latency_ms=5.0)


def _make_agent():
    return CriticAgent(llm=LLM(), tracer=Tracer("test question"))


def _evidence(source_id="src-1"):
    return Evidence(
        source_id=source_id,
        url="https://example.com",
        title="Test Source",
        quote="RAG retrieves external documents and feeds them to the model.",
        sub_question_id="sq1",
    )


# ── CriticAgent.run tests ────────────────────────────────────────────────────

def test_critic_marks_supported():
    claims = [Claim(text="RAG improves accuracy.", evidence_ids=["src-1"])]
    with patch.object(LLM, "complete_json", return_value=({"supported": True, "note": "Confirmed."}, _MOCK_USAGE)):
        result = _make_agent().run(claims, [_evidence("src-1")])
    assert result[0].supported is True
    assert result[0].note == "Confirmed."


def test_critic_marks_unsupported():
    claims = [Claim(text="RAG was invented in 2010.", evidence_ids=["src-1"])]
    with patch.object(LLM, "complete_json", return_value=({"supported": False, "note": "No mention of 2010."}, _MOCK_USAGE)):
        result = _make_agent().run(claims, [_evidence("src-1")])
    assert result[0].supported is False


def test_critic_no_matching_evidence_is_unsupported():
    """Claim citing an ID absent from the evidence list is marked unsupported without an LLM call."""
    claims = [Claim(text="Something.", evidence_ids=["ghost-id"])]
    result = _make_agent().run(claims, evidence=[])
    assert result[0].supported is False
    assert "no matching evidence" in result[0].note


def test_critic_records_usage_on_trace_step():
    agent = _make_agent()
    claims = [Claim(text="RAG is useful.", evidence_ids=["src-1"])]
    with patch.object(LLM, "complete_json", return_value=({"supported": True, "note": "ok"}, _MOCK_USAGE)):
        agent.run(claims, [_evidence("src-1")])
    step = agent.tracer.trace.steps[0]
    assert step.input_tokens == 20
    assert step.output_tokens == 10


# ── check_citations (guardrails) tests ──────────────────────────────────────

def test_check_citations_drops_uncited_claim():
    """A claim with no evidence_ids must move to unanswered."""
    report = Report(
        question="test?",
        claims=[
            Claim(text="Cited claim.", evidence_ids=["src-1"], supported=True),
            Claim(text="Uncited claim.", evidence_ids=[], supported=True),
        ],
        sources=[_evidence("src-1")],
    )
    result = check_citations(report)
    assert len(result.claims) == 1
    assert result.claims[0].text == "Cited claim."
    assert "Uncited claim." in result.unanswered


def test_check_citations_drops_unsupported_claim():
    """A claim with supported=False must move to unanswered even if cited."""
    report = Report(
        question="test?",
        claims=[
            Claim(text="Good claim.", evidence_ids=["src-1"], supported=True),
            Claim(text="Bad claim.", evidence_ids=["src-1"], supported=False),
        ],
        sources=[_evidence("src-1")],
    )
    result = check_citations(report)
    assert len(result.claims) == 1
    assert result.claims[0].text == "Good claim."
    assert "Bad claim." in result.unanswered


def test_check_citations_drops_claim_with_phantom_evidence_id():
    """A claim whose evidence_id doesn't exist in report.sources must be dropped."""
    report = Report(
        question="test?",
        claims=[Claim(text="Phantom cite.", evidence_ids=["nonexistent"], supported=True)],
        sources=[_evidence("src-1")],
    )
    result = check_citations(report)
    assert result.claims == []
    assert "Phantom cite." in result.unanswered


def test_check_citations_keeps_fully_valid_claim():
    """A supported claim with all evidence_ids in sources must be kept."""
    report = Report(
        question="test?",
        claims=[Claim(text="Valid.", evidence_ids=["src-1"], supported=True)],
        sources=[_evidence("src-1")],
    )
    result = check_citations(report)
    assert len(result.claims) == 1
    assert result.unanswered == []


def test_check_citations_reason_in_note():
    """Dropped claim's note must contain the reason; the claim object is mutated in-place."""
    claim = Claim(text="No cite.", evidence_ids=[], supported=True)
    report = Report(question="test?", claims=[claim], sources=[])
    check_citations(report)
    # The claim is removed from report.claims but its .note was mutated before that
    assert "dropped" in claim.note
