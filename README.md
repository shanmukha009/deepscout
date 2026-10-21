# DeepScout — Multi-Agent Research Analyst

[![CI](https://github.com/shanmukha009/deepscout/actions/workflows/ci.yml/badge.svg)](https://github.com/shanmukha009/deepscout/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A multi-agent system that takes a research question, plans it, searches and reads
sources, **verifies every claim against the evidence**, and writes a cited report —
with retrieval-augmented reading, response caching, full cost/latency tracing, and a
benchmarked eval harness.

The idea: a plain LLM will answer a hard factual question confidently even when it has
no grounding, and that is exactly where it hurts you. DeepScout plans, retrieves, reads
real sources, **criticizes its own draft**, and only then answers. If a claim cannot be
grounded in evidence, it is dropped rather than guessed.

## Architecture

```
question
   │
   ▼
┌──────────┐   plan → sub-questions
│ Planner  │ ─────────────────────────────┐
└──────────┘                              │
                     sub-questions run CONCURRENTLY
                     ┌──────────┐     ┌─────────────────────────────┐
                     │ Searcher │ ──► │  Reader                     │
                     │ (cached) │     │  • fetch sources concurrently│
                     └──────────┘     │  • RAG: embed + retrieve the │
                                      │    most relevant passages    │
                                      │  • extract verbatim evidence │
                                      └─────────────────────────────┘
                                     │
                                     ▼
                              ┌──────────┐   marks each claim supported / unsupported
                              │  Critic  │
                              └──────────┘
                                     │
                                     ▼
                              ┌──────────────┐   cited report (supported claims only)
                              │ Synthesizer  │
                              └──────────────┘
                                     │
                                     ▼
                         guardrails.check_citations()   ← "cite or drop"
                                     │
                                     ▼
                           Report + RunTrace (cost, latency, per-step)
```

Everything flows through the typed Pydantic contracts in `src/deepscout/models.py`;
agents only read and write those types, which is what let the five agents be built in
parallel without colliding.

## What sets it apart

- **Verification agent (Critic)** — checks each claim against its evidence before it
  reaches the report.
- **Guardrail** — `guardrails.check_citations` drops or flags any claim that is
  unsupported or uncited ("cite or drop").
- **RAG retrieval** (`rag.py`) — each fetched source is chunked, embedded, and reduced
  to the passages most relevant to the sub-question before quote extraction, so the
  model grounds on the right span instead of a blind truncation. Pluggable embeddings
  (deterministic hashed vectors for tests/CI; `text-embedding-3-small` in production)
  behind a small `VectorStore` interface that swaps for FAISS / Chroma.
- **Latency & cost controls** — sub-questions and per-source reads run concurrently;
  identical LLM and search calls are served from a disk cache (`DS_CACHE=1`); the
  benchmark harness evaluates questions in parallel.
- **Benchmarked eval harness** (`evals/benchmark/`) — scores accuracy (LLM-as-judge),
  faithfulness, hallucination and abstain rate, cost and latency, on 150 verified
  fact-seeking questions from OpenAI's SimpleQA set.
- **Observability** — `tracing.py` records every agent step with tokens, USD cost, and
  latency; the Streamlit UI shows the trace live.
- **CI** — GitHub Actions runs the 57-test suite on every push across Python 3.11/3.12.

## Benchmark

Questions are drawn from **OpenAI's SimpleQA** benchmark — short, fact-seeking questions
with verified short answers, chosen because they are exactly where LLMs hallucinate.
150 questions, balanced across 8 topic categories. The judge is an LLM-as-judge that
classifies each answer correct / incorrect / abstained against the ground truth.

```bash
# full run (uses your provider + search keys from .env)
export $(grep -v '^#' .env | xargs)
export PYTHONPATH=src
python evals/benchmark/run_benchmark.py            # all 150, 8 workers
python evals/benchmark/run_benchmark.py --limit 25 # quick subset
python evals/benchmark/plot_results.py             # regenerate results.png
```

### Results

Real run, **n=150** SimpleQA questions, `gpt-4o-mini`, Tavily search
(reproduce with the command above):

| Metric | DeepScout | Baseline (LLM only) |
|---|---|---|
| Accuracy | **62.0%** | 8.0% |
| Faithfulness | **1.00** | 0.00 |
| Hallucination rate | **34.7%** | 88.7% |
| Abstain rate | 3.3% | 3.3% |
| Mean cost / query | $0.0030 | $0.00003 |
| Mean latency | 31.6s | 1.0s |

DeepScout answers **62%** of hard fact-seeking questions correctly versus **8%** for a
bare LLM, and every claim in its reports is backed by a cited source (faithfulness 1.00).
It trades cost and latency for that gain — it retrieves and verifies instead of answering
from memory — which is the right tradeoff for an agent trusted with research.

### Where it fails (honest limitations)

Of 150 questions, DeepScout got **93 correct, 52 incorrect, 5 abstained** (0 runtime
failures). Accuracy by category:

| Strongest | | Weakest | |
|---|---|---|---|
| Science & technology | 78% | TV shows | 37% |
| Sports / History | 68% | Geography | 58% |
| Music / Art | 63% | Politics | 61% |

- **TV-show trivia is the weak spot** (exact episode quotes, who directed a specific
  episode) — the facts are thin on the open web, so retrieval surfaces the wrong passage.
- When DeepScout is wrong it is almost always because **retrieval found the wrong fact**,
  not because it fabricated one: the cite-or-drop guardrail keeps faithfulness at 1.00, so
  wrong answers are still grounded in a (mis-selected) source rather than invented.
- It is **~100x more expensive and ~30x slower** than a single LLM call. For throwaway
  questions that is not worth it; for research you need to trust, it is.

## Run it (mock mode, no API key)

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python -m deepscout.orchestrator "What is retrieval-augmented generation?"
pytest -q
streamlit run app/streamlit_app.py     # the demo UI
```

## Go live

1. `cp .env.example .env` and set `DS_PROVIDER` / `DS_MODEL` / `DS_SEARCH` and your keys.
2. Optional: `DS_CACHE=1` (cache repeated calls), `DS_RAG=1` (default on),
   `DS_EMBED=openai` (real embeddings), `DS_SOURCES_PER_SUBQ`, `DS_BENCH_WORKERS`.
3. Run the benchmark and record the numbers for your write-up.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `DS_PROVIDER` | `mock` | `mock` \| `anthropic` \| `openai` |
| `DS_MODEL` | `mock-model` | model id |
| `DS_SEARCH` | `mock` | `mock` \| `tavily` \| `serper` |
| `DS_RAG` | `1` | retrieval-augmented reading on/off |
| `DS_EMBED` | `mock` | `mock` \| `openai` embeddings |
| `DS_CACHE` | `0` | cache identical LLM/search calls |
| `DS_SOURCES_PER_SUBQ` | `5` | sources fetched per sub-question |
| `DS_BENCH_WORKERS` | `8` | benchmark concurrency |

## Honesty note

Built as a portfolio project on public web search and a public benchmark. The results
table is from a real n=150 run of `run_benchmark.py` on OpenAI's SimpleQA set, numbers and
all, including the failure cases. Not affiliated with any lab whose agent pattern it
mirrors.

## Layout

```
src/deepscout/
  models.py          # locked Pydantic data contracts
  llm.py             # provider-agnostic client + cost/latency + cache
  cache.py           # disk cache for LLM + search calls
  rag.py             # chunk + embed + retrieve (vector store)
  config.py          # env-based settings
  tracing.py         # per-step observability
  guardrails.py      # cite-or-drop enforcement
  orchestrator.py    # wires the agents together (concurrent)
  baseline.py        # naive single-shot baseline for comparison
  tools/search.py    # pluggable search (cached)
  agents/            # planner, searcher, reader, critic, synthesizer
app/streamlit_app.py # demo UI
evals/benchmark/     # SimpleQA benchmark harness + dataset + plot
tests/               # 57 contract/unit tests (run in CI)
```
