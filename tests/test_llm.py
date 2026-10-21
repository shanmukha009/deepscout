import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# Ensure mock mode is active before the config singleton is created.
os.environ.setdefault("DS_PROVIDER", "mock")
os.environ.setdefault("DS_MODEL", "mock-model")

import json
from unittest.mock import patch


def test_complete_mock_returns_nonempty_text_and_nonnegative_cost():
    from deepscout.llm import LLM, Usage

    llm = LLM()
    text, usage = llm.complete("you are helpful", "what is 2 + 2?")

    assert isinstance(text, str) and len(text) > 0, "complete() must return non-empty text"
    assert isinstance(usage, Usage)
    assert usage.cost_usd >= 0


def test_complete_json_retries_and_accumulates_usage():
    """complete_json retries once on bad JSON and merges usage from both calls."""
    from deepscout.llm import LLM, Usage

    calls = iter([
        ("not valid json {{", Usage(input_tokens=10, output_tokens=5, cost_usd=0.0, latency_ms=1.0)),
        ('{"answer": 42}',   Usage(input_tokens=10, output_tokens=5, cost_usd=0.0, latency_ms=1.0)),
    ])

    with patch.object(LLM, "complete", lambda self, *a, **kw: next(calls)):
        llm = LLM()
        result, usage = llm.complete_json("system", "user")

    assert result == {"answer": 42}
    assert usage.input_tokens == 20   # accumulated across both attempts
