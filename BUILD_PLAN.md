# Build Plan — driving this with T3 Code (parallel agents)

The scaffold already defines every data contract (`src/deepscout/models.py`) and the
agent interface (`src/deepscout/agents/base.py`). That means the agents can be built
**in parallel** — each on its own branch — without stepping on each other.

## How to run it in T3 Code

1. Put this folder in a git repo (`git init && git add -A && git commit -m "scaffold"`).
2. Open it in T3 Code.
3. Start **Thread A first** (everyone depends on the LLM client). When its PR is in,
   fire **Threads B, C, D, E, G in parallel** — they only touch their own files.
4. Finish with **Thread F** (integration), which ties the finished agents together.
5. Each thread = one T3 Code thread → its own branch → a pull request you review.

Dependency order:

```
A (foundation)  ──►  B, C, D, E, G  (parallel)  ──►  F (integration)
```

Paste each prompt below into its own T3 Code thread.

---

### Thread A — LLM client, config, tracing  (do first)
> In `src/deepscout/llm.py`, implement the real provider calls in `LLM.complete` for
> `provider="anthropic"` and `provider="openai"`, reading token usage from each SDK's
> response and keeping the returned `(text, Usage)` shape exactly as it is now. Update
> `PRICING` with current per-million-token rates for the models in `.env.example`. Add
> a small helper `complete_json()` that asks for and parses strict JSON (retry once on
> parse failure). Do not change `models.py` or any agent file. Add a unit test that
> `complete()` returns a non-empty string and a `Usage` with cost >= 0 in mock mode.

### Thread B — Planner agent
> Implement `PlannerAgent.run` in `src/deepscout/agents/planner.py`. Prompt the LLM to
> break the question into 3–`settings.max_sub_questions` focused sub-questions and return
> them as JSON; parse into `SubQuestion` objects (ids `sq1`, `sq2`, …). Record token usage
> and cost on the trace `step`. Only edit `planner.py`. Add a test that a stub/mock run
> returns a `ResearchPlan` with >= 1 sub-question.

### Thread C — Search + Reader + search tool
> Implement real web search in `src/deepscout/tools/search.py` behind the existing
> `SearchTool.query` interface (support Tavily or Serper via `settings.search_api_key`;
> keep the `mock` branch working). Then implement `SearcherAgent.run`
> (`agents/searcher.py`) to optionally rewrite the query with the LLM before searching,
> and `ReaderAgent.run` (`agents/reader.py`) to fetch each result and have the LLM pull
> **verbatim quotes** that answer the sub-question, returning `Evidence` objects (the
> `quote` must be copied exactly from the source, never paraphrased). Only edit those
> three files. Add a test using the mock search provider.

### Thread D — Critic + guardrails
> Implement `CriticAgent.run` (`agents/critic.py`): for each `Claim`, ask the LLM whether
> the cited `Evidence` actually supports it, and set `claim.supported` and `claim.note`.
> Then harden `guardrails.check_citations` (`guardrails.py`) so it keeps only claims that
> are `supported` AND whose `evidence_ids` all map to real `report.sources`; everything
> else moves to `report.unanswered` with a reason. Only edit those two files. Add a test
> that an uncited claim is dropped and an unsupported claim is dropped.

### Thread E — Synthesizer
> Implement `SynthesizerAgent.run` (`agents/synthesizer.py`). Prompt the LLM to write a
> concise, structured summary that cites only supported claims by their `evidence_id`,
> and return a `Report`. Keep `Report.to_markdown` working. Record usage on the trace
> `step`. Only edit `synthesizer.py`. Add a test that the report contains only supported
> claims.

### Thread F — Orchestrator + Streamlit UI  (do last)
> In `src/deepscout/orchestrator.py`, add a claim-drafting step: after reading evidence,
> prompt the LLM to turn evidence into draft `Claim`s (each citing its `evidence_ids`)
> before the Critic runs. Make the per-sub-question search+read loop run concurrently
> (thread pool). Then polish `app/streamlit_app.py` so it streams the trace as agents run
> and shows the final report, total cost, and latency. Do not change `models.py`. Add an
> end-to-end test in mock mode that `run_research` returns a `Report` and a `RunTrace`
> with >= 5 steps.

### Thread G — Eval harness
> Extend `evals/run_evals.py`: add a third metric, **citation validity** (every
> `evidence_id` in a kept claim exists in `report.sources`), and write results to
> `evals/results.json` as well as stdout. Add 4 more questions to `evals/testset.jsonl`
> across different domains. Only edit the `evals/` folder. Make `python evals/run_evals.py`
> print a clean per-question table plus the means.

---

## After the threads merge
- `cp .env.example .env`, set a real provider + search key.
- `python evals/run_evals.py` → record the faithfulness number.
- Record a 30-second screen capture of the Streamlit demo for the LinkedIn post.
- Ping me with what the numbers came out to and I'll draft the post in your voice.
