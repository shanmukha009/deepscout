import sys
import os
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("DS_PROVIDER", "mock")
os.environ.setdefault("DS_MODEL", "mock-model")
os.environ.setdefault("DS_SEARCH", "mock")

from unittest.mock import patch

from deepscout.llm import LLM, Usage
from deepscout.tracing import Tracer
from deepscout.tools.search import SearchTool
from deepscout.agents.searcher import SearcherAgent
from deepscout.agents.reader import ReaderAgent
from deepscout.models import SubQuestion, SearchResult, Evidence


def _sub(id: str = "sq1", question: str = "What is RAG?") -> SubQuestion:
    return SubQuestion(id=id, question=question)


# --- SearchTool ---

def test_search_tool_mock_returns_list():
    results = SearchTool().query("test query")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert isinstance(results[0], SearchResult)


def test_search_tool_mock_snippet_contains_query():
    results = SearchTool().query("retrieval augmented generation")
    assert "retrieval augmented generation" in results[0].snippet


# --- SearcherAgent ---

def test_searcher_mock_returns_search_results():
    agent = SearcherAgent(llm=LLM(), tracer=Tracer("q"), search=SearchTool())
    results = agent.run(_sub())
    assert isinstance(results, list) and len(results) >= 1
    assert isinstance(results[0], SearchResult)


def test_searcher_skips_llm_rewrite_in_mock_mode():
    """In mock mode the LLM rewrite path is bypassed entirely."""
    called = []
    original_complete = LLM.complete

    def spy(self, *args, **kwargs):
        called.append(True)
        return original_complete(self, *args, **kwargs)

    agent = SearcherAgent(llm=LLM(), tracer=Tracer("q"), search=SearchTool())
    with patch.object(LLM, "complete", spy):
        agent.run(_sub())

    assert called == [], "LLM.complete should not be called in mock mode"


def test_searcher_records_no_usage_in_mock_mode():
    tracer = Tracer("q")
    agent = SearcherAgent(llm=LLM(), tracer=tracer, search=SearchTool())
    agent.run(_sub())
    step = tracer.trace.steps[0]
    # mock path records zero usage
    assert step.input_tokens == 0
    assert step.cost_usd == 0.0


# --- ReaderAgent ---

def test_reader_mock_returns_evidence_for_each_result():
    results = [
        SearchResult(title="Src A", url="https://a.com", snippet="RAG combines retrieval with generation."),
        SearchResult(title="Src B", url="https://b.com", snippet="Vector stores index embeddings."),
    ]
    agent = ReaderAgent(llm=LLM(), tracer=Tracer("q"))
    evidence = agent.run(_sub(), results)
    assert len(evidence) == 2
    assert all(isinstance(e, Evidence) for e in evidence)


def test_reader_mock_copies_snippet_verbatim():
    snippet = "RAG combines retrieval with generation."
    results = [SearchResult(title="T", url="https://t.com", snippet=snippet)]
    agent = ReaderAgent(llm=LLM(), tracer=Tracer("q"))
    evidence = agent.run(_sub(), results)
    assert evidence[0].quote == snippet


def test_reader_source_id_format():
    results = [
        SearchResult(title="A", url="https://a.com", snippet="a"),
        SearchResult(title="B", url="https://b.com", snippet="b"),
    ]
    agent = ReaderAgent(llm=LLM(), tracer=Tracer("q"))
    evidence = agent.run(_sub(id="sq3"), results)
    assert evidence[0].source_id == "sq3-s0"
    assert evidence[1].source_id == "sq3-s1"


def test_reader_sets_sub_question_id():
    results = [SearchResult(title="T", url="https://t.com", snippet="some text")]
    agent = ReaderAgent(llm=LLM(), tracer=Tracer("q"))
    evidence = agent.run(_sub(id="sq2"), results)
    assert evidence[0].sub_question_id == "sq2"


def test_reader_records_trace_step():
    tracer = Tracer("q")
    results = [SearchResult(title="T", url="https://t.com", snippet="text")]
    ReaderAgent(llm=LLM(), tracer=tracer).run(_sub(), results)
    assert len(tracer.trace.steps) == 1
    assert tracer.trace.steps[0].agent == "reader"


def test_reader_empty_results_returns_empty_list():
    agent = ReaderAgent(llm=LLM(), tracer=Tracer("q"))
    assert agent.run(_sub(), []) == []


# --- Integration: searcher feeds reader ---

def test_searcher_output_feeds_reader():
    tracer = Tracer("What is RAG?")
    sub = _sub()
    tool = SearchTool()
    searcher = SearcherAgent(llm=LLM(), tracer=tracer, search=tool)
    reader = ReaderAgent(llm=LLM(), tracer=tracer)

    search_results = searcher.run(sub)
    evidence = reader.run(sub, search_results)

    assert len(evidence) == len(search_results)
    for ev in evidence:
        assert ev.sub_question_id == sub.id
        assert ev.quote  # non-empty quote
